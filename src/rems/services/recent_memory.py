"""JShi adapter around Mem0 OSS. Mem0 owns all memory data and retrieval.

The adapter only adds stable source identities, source filters, disk retention,
and RecalledFragment conversion. No second recent-memory database or index.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from datetime import datetime, timezone

from ..models.interlocutor import InterlocutorAttribution
from ..port import MemoryBatch, MemoryExperience, RecalledFragment

logger = logging.getLogger(__name__)
AGENT = "jshi-recent-v1"


def timestamp(value: datetime | None, default: float) -> float:
    # Match JShi's local naive wall clock; aware datetimes retain their offset.
    return value.timestamp() if value is not None else default


class RecentCapacityError(RuntimeError):
    pass


class Mem0RecentMemory:
    def __init__(self, config, *, clock=time.time, memory_factory=None, background=True):
        from sqlalchemy.engine import make_url
        self.config = config
        self.policy = config.recent_memory
        database = make_url(config.storage.database_url).database
        base = Path(database).resolve().parent if database and database != ":memory:" else Path.cwd()
        self.root = Path(self.policy.path).resolve() if self.policy.path else base / "recent_memory"
        if database and self.root == Path(database).resolve():
            raise ValueError("Recent memory must use a dedicated directory")
        if database and database != ":memory:" and self.root in Path(database).resolve().parents:
            raise ValueError("Recent memory directory must not contain the long-term database")
        if config.storage.qdrant_path:
            long_vectors = Path(config.storage.qdrant_path).resolve()
            if (self.root == long_vectors or self.root in long_vectors.parents
                    or long_vectors in self.root.parents):
                raise ValueError("Recent and long-term Qdrant paths must be independent")
        self._clock, self._factory = clock, memory_factory
        self._memory = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._background = background
        self._last_maintenance = 0.0
        self.last_maintenance: dict = {}

    def _get_memory(self):
        if self._stop.is_set():
            raise RuntimeError("Recent memory is closed")
        if self._memory is not None:
            return self._memory
        marker = self.root / "rems-mem0.json"
        if self.root.exists() and any(self.root.iterdir()) and not marker.exists():
            raise ValueError(f"Refusing to adopt nonempty recent memory directory: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        embedder = self.policy.embedder or {
            "provider": "huggingface", "config": {"model": self.config.embedding.model_name},
        }
        fingerprint = hashlib.sha256(json.dumps(
            [embedder, self.policy.embedding_dims], sort_keys=True).encode()).hexdigest()
        if marker.exists():
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            if metadata.get("owner") != AGENT:
                raise ValueError("Recent memory directory has a different owner")
            if metadata.get("embedding") != fingerprint:
                raise ValueError("Mem0 embedding configuration changed; use a new recent path")
        else:
            marker.write_text(json.dumps({"owner": AGENT, "embedding": fingerprint}), encoding="utf-8")
        # Disable optional outbound telemetry before importing Mem0.
        os.environ["MEM0_TELEMETRY"] = "false"
        native_config = {
            "vector_store": {"provider": "qdrant", "config": {
                "path": str(self.root / "qdrant"), "collection_name": "recent",
                "embedding_model_dims": self.policy.embedding_dims, "on_disk": True}},
            "history_db_path": str(self.root / "history.db"),
            "embedder": embedder,
            # infer=False never calls this LLM; avoid needing another secret.
            "llm": {"provider": "openai", "config": {
                "api_key": "unused-raw-mode", "openai_base_url": "http://127.0.0.1:9/v1"}},
        }
        if self._factory is None:
            from mem0 import Memory
            self._factory = Memory.from_config
        if self.policy.reranker:
            native_config["reranker"] = self.policy.reranker
        # Offline loading is a runtime option, not a change to embedding identity.
        if embedder.get("provider") == "huggingface":
            import copy
            native_config["embedder"] = copy.deepcopy(embedder)
            native_config["embedder"].setdefault("config", {}).setdefault(
                "model_kwargs", {}).setdefault("local_files_only", True)
            # Device is a runtime choice. Keep the embedding identity and stored vectors.
            native_config["embedder"]["config"]["model_kwargs"].setdefault(
                "device", self.config.embedding.device)
        self._memory = self._factory(native_config)
        self._compact(initialize=True)
        if self._background:
            self._thread = threading.Thread(target=self._maintenance_loop,
                name="rems-mem0-retention", daemon=True)
            self._thread.start()
        return self._memory

    def _maintenance_loop(self):
        while not self._stop.wait(self.policy.maintenance_interval_seconds):
            try:
                self.maintain()
            except Exception:
                logger.exception("Mem0 retention maintenance failed")

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def _compact(self, *, initialize=False, clear_history=False):
        # Mem0 delete retains audit copies of the raw text. Its own history reset
        # is necessary for a bounded store. Active Qdrant memories are untouched.
        if clear_history:
            from mem0.memory.storage import SQLiteManager
            self._memory.db.reset()
            self._memory.db.close()
            self._memory.db = SQLiteManager(str(self.root / "history.db"))
        # Native SQLite maintenance only, no custom tables or memory writes.
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix not in {".db", ".sqlite"}:
                continue
            with sqlite3.connect(path, timeout=30) as db:
                if initialize and db.execute("PRAGMA auto_vacuum").fetchone()[0] == 0:
                    db.execute("PRAGMA auto_vacuum=INCREMENTAL")
                    db.execute("VACUUM")
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                # Incremental vacuum can free one page per call on some SQLite builds.
                while db.execute("PRAGMA freelist_count").fetchone()[0] > 0:
                    before = db.execute("PRAGMA freelist_count").fetchone()[0]
                    db.execute("PRAGMA incremental_vacuum")
                    if db.execute("PRAGMA freelist_count").fetchone()[0] >= before:
                        break
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _records(self):
        # Mem0 get_all has no cursor in 2.2.1. Use its native Qdrant cursor for
        # bounded maintenance enumeration, never for substitute retrieval.
        memory = self._get_memory()
        offset = None
        while True:
            rows, offset = memory.vector_store.client.scroll(
                collection_name=memory.collection_name,
                scroll_filter=memory.vector_store._create_filter({"agent_id": AGENT}),
                limit=self.policy.cleanup_batch_size, offset=offset,
                with_payload=True, with_vectors=False)
            yield from rows
            if offset is None:
                break

    def maintain(self, *, force=False) -> dict:
        with self._lock:
            self._get_memory()
            now = self._clock()
            size = self.disk_bytes()
            if not force and size < self.policy.cleanup_trigger_bytes and (
                    now - self._last_maintenance < self.policy.maintenance_interval_seconds):
                return self.last_maintenance
            cutoff = now - self.policy.retention_days * 86400
            deleted = 0
            # Delete in batches, restarting the native cursor after mutation.
            while True:
                expired = []
                for row in self._records():
                    if row.payload.get("rems_time", now) < cutoff:
                        expired.append(str(row.id))
                    if len(expired) >= self.policy.cleanup_batch_size:
                        break
                if not expired:
                    break
                for identity in expired:
                    self._memory.delete(identity)
                    deleted += 1
            if deleted or force or size >= self.policy.cleanup_trigger_bytes:
                self._compact(clear_history=True)
            capacity_cleanup = size >= self.policy.cleanup_trigger_bytes
            while capacity_cleanup and self.disk_bytes() > self.policy.cleanup_target_bytes:
                oldest = heapq.nsmallest(self.policy.cleanup_batch_size, self._records(),
                    key=lambda row: (row.payload.get("rems_time", 0), str(row.id)))
                if not oldest:
                    break
                for row in oldest:
                    self._memory.delete(str(row.id))
                    deleted += 1
                self._compact(clear_history=True)
            self._last_maintenance = now
            self.last_maintenance = {"deleted": deleted, "disk_bytes": self.disk_bytes(),
                "target_reached": self.disk_bytes() <= self.policy.cleanup_target_bytes,
                "capacity_cleanup": capacity_cleanup}
            return dict(self.last_maintenance)

    def ingest_batch(self, batch: MemoryBatch) -> None:
        with self._lock:
            memory = self._get_memory()
            self.maintain()
            now = self._clock()
            subject = hashlib.sha256(batch.subject_id.encode()).hexdigest()
            segment_ids = list(dict.fromkeys(
                sid for e in batch.experiences
                for sid in ([s.segment_id for s in e.sub_segments] or [e.segment_id]) if sid))
            # A replay batch need not rescan the local native store once per
            # message. Ask Mem0 for this batch's immutable segment IDs together.
            existing_segments = {}
            for offset in range(0, len(segment_ids), 256):
                wanted = segment_ids[offset:offset+256]
                rows = memory.get_all(filters={"user_id": subject, "agent_id": AGENT,
                    "rems_segment": {"in": wanted}}, top_k=len(wanted), show_expired=True)["results"]
                existing_segments.update(((row.get("metadata") or {}).get("rems_segment"), row["memory"])
                                         for row in rows)
            for experience in batch.experiences:
                if experience.subject_id != batch.subject_id:
                    raise ValueError("Mixed subjects in recent memory batch")
                if not experience.text.strip():
                    continue
                # Keep original subsegment ownership where supplied by a caller.
                parts = [MemoryExperience(subject_id=batch.subject_id, text=s.text,
                    objects=s.objects, interlocutor=s.interlocutor_object_id,
                    source_ids=s.source_ids, segment_id=s.segment_id,
                    occurred_at=s.occurred_at, origin=experience.origin)
                    for s in experience.sub_segments] or [experience]
                for item in parts:
                    moment = timestamp(item.occurred_at, now)
                    if moment < now - self.policy.retention_days * 86400:
                        continue
                    data = item.model_dump(mode="json")
                    identity = hashlib.sha256(json.dumps(data, sort_keys=True,
                        ensure_ascii=False).encode()).hexdigest()
                    # JShi segments are immutable. Use their stable identity for
                    # replay/live retries even when aware UTC becomes local naive
                    # time at the JShi adapter boundary.
                    if item.segment_id:
                        existing = ([{"memory": existing_segments[item.segment_id]}]
                                    if item.segment_id in existing_segments else [])
                    else:
                        existing = memory.get_all(filters={"user_id": subject, "agent_id": AGENT,
                            "rems_key": identity}, top_k=1, show_expired=True)["results"]
                    if existing:
                        if existing[0]["memory"] != item.text:
                            raise ValueError("Existing segment has different raw text; use an explicit correction")
                        continue
                    estimated = len(item.text.encode("utf-8")) * 4 + 65536
                    if self.disk_bytes() + estimated > self.policy.max_bytes:
                        self.maintain(force=True)
                        if self.disk_bytes() + estimated > self.policy.max_bytes:
                            raise RecentCapacityError("Mem0 disk budget exhausted; retry after cleanup")
                    metadata = {"rems_key": identity, "rems_time": moment,
                        "rems_subject": batch.subject_id, "rems_sources": list(item.source_ids),
                        "rems_origin": item.origin,
                        "rems_segment": item.segment_id, "rems_interlocutor": item.interlocutor,
                        "rems_objects": list(dict.fromkeys([*item.objects.values(),
                            *([item.interlocutor] if item.interlocutor else [])])),
                        "rems_occurred_at": item.occurred_at.isoformat() if item.occurred_at else None}
                    added = memory.add(item.text, user_id=subject, agent_id=AGENT,
                        metadata=metadata, infer=False)["results"]
                    if not added:
                        raise RuntimeError("Mem0 returned no stored memory")
                    if item.segment_id:
                        existing_segments[item.segment_id] = item.text
                    current_size = self.disk_bytes()
                    if current_size > self.policy.max_bytes:
                        for row in added:
                            memory.delete(row["id"])
                        self._compact(clear_history=True)
                        raise RecentCapacityError("Mem0 write exceeded disk budget; write rolled back")
                    if current_size >= self.policy.cleanup_trigger_bytes:
                        self.maintain(force=True)
            self.maintain()

    def recall(self, subject_id, query, *, object_id=None, object_ids=None,
               interlocutor_object_id=None, time_range=None, **kwargs):
        if not query.strip():
            return ()
        with self._lock:
            memory = self._get_memory()
            now = self._clock()
            filters = {"agent_id": AGENT,
                "user_id": hashlib.sha256(subject_id.encode()).hexdigest(),
                "rems_time": {"gte": now - self.policy.retention_days * 86400}}
            if time_range:
                filters["rems_time"]["gte"] = max(filters["rems_time"]["gte"],
                    timestamp(time_range[0], now))
                filters["rems_time"]["lte"] = timestamp(time_range[1], now)
            wanted = tuple(dict.fromkeys([*(object_ids or ()),
                *([object_id] if object_id else []),
                *([interlocutor_object_id] if interlocutor_object_id else [])]))
            if wanted:
                filters["rems_objects"] = {"in": list(wanted)}
            rows = memory.search(query, filters=filters,
                top_k=max(self.policy.search_limit, self.policy.rerank_candidate_limit)
                    if self.policy.reranker else self.policy.search_limit,
                threshold=self.policy.search_threshold,
                rerank=bool(self.policy.reranker))["results"][:self.policy.search_limit]
            result = []
            for row in rows:
                meta = row.get("metadata") or {}
                # Defense against provider/filter changes; subject and time are hard boundaries.
                moment = meta.get("rems_time", 0)
                if meta.get("rems_subject") != subject_id or moment < filters["rems_time"]["gte"]:
                    continue
                if time_range and moment > filters["rems_time"]["lte"]:
                    continue
                if wanted and not set(wanted).intersection(meta.get("rems_objects", [])):
                    continue
                text = row["memory"]
                date = datetime.fromtimestamp(moment, timezone.utc)
                who = meta.get("rems_interlocutor")
                origin = meta.get("rems_origin")
                # JShi's external origin also includes its own spoken replies.
                label = {"internal": "；内部记录", "external": "；外部经历", "dream": "；梦境记录"}.get(origin, "")
                result.append(RecalledFragment(event_id="mem0:" + row["id"], text=text,
                    content=f"〔近期原文 {date.astimezone():%Y-%m-%d %H:%M}{label}〕\n{text}",
                    kind="recent", summary_level="raw", representation_level="raw",
                    source_ids=meta.get("rems_sources", []), occurred_at=date,
                    object_id=next((x for x in wanted if x in meta.get("rems_objects", [])), who),
                    interlocutor=who, interlocutor_object_ids=[who] if who else [],
                    interlocutor_attributions=[InterlocutorAttribution(
                        segment_id=meta.get("rems_segment") or identity_for(row), object_id=who)] if who else [],
                    score=float(row.get("score") or 0), signals={"recent": 1.0,
                        "source_origin": {"external": 0.0, "internal": 1.0, "dream": 2.0}.get(origin, -1.0)}))
            return tuple(result)

    def prepare(self):
        with self._lock:
            embedder = self._get_memory().embedding_model
            embedder.embed("记忆检索准备", "search")
            model = getattr(embedder, "model", None)
            return {"embedding_device": str(getattr(model, "device", "unknown"))}

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        with self._lock:
            if self._memory is not None:
                self._memory.close()
                self._memory.vector_store.client.close()
                self._memory = None


def identity_for(row):
    return "mem0:" + row["id"]
