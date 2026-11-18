"""Read-only JShi sampling and reproducible real-content dual-memory evaluation.

prepare never invokes a model. run uses the real configured LLM for long-event
ingest and Mem0's native embedder/search. It requires --live explicitly.
Reports measure retrieved/delivered evidence, not subjective answer accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rems.config import REMSConfig
from rems.llm.provider import LLMProvider
from rems.pipeline import REMSPipeline, MemoryIngestError
from rems.port import MemoryBatch, MemoryExperience
from rems.recall.fusion import merge_memories
from rems.eval.jshi_snapshot import experience_mapper


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(source, output, *, subject, limit, max_chars, jshi_root=None):
    if limit <= 0 or max_chars <= 0:
        raise ValueError("Sampling limits must be positive")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory; snapshots are immutable")
    output.mkdir(parents=True)
    uri = Path(source).resolve().as_uri() + "?mode=ro"
    selected, size = [], 0
    convert = experience_mapper(jshi_root)
    with sqlite3.connect(uri, uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute("SELECT payload FROM experience_segments WHERE subject_id=? ORDER BY sequence DESC", (subject,))
        for (payload,) in rows:
            row = json.loads(payload)
            experience = convert(row)
            if experience is None:
                continue
            text = experience.text
            if size + len(text) > max_chars:
                break
            selected.append(experience)
            size += len(text)
            if len(selected) >= limit:
                break
    selected.reverse()
    if not selected:
        raise ValueError("No actual text found in the selected ledger")
    data = [item.model_dump(mode="json") for item in selected]
    save(output / "experiences.json", data)
    save(output / "manifest.json", {"source_db": str(Path(source).resolve()),
        "read_only": True, "mapping": "JShi memorycontrol native mapping",
        "subject": subject, "segments": len(data), "chars": size,
        "first": data[0]["occurred_at"], "last": data[-1]["occurred_at"],
        "sha256": hashlib.sha256((output / "experiences.json").read_bytes()).hexdigest()})
    save(output / "cases.example.json", [{"id": "replace-with-reviewed-case",
        "query": "请根据实际记录填写一个自然问法",
        "expected_sources": data[-1]["source_ids"], "expected_quotes": [data[-1]["text"]]}])
    return {"segments": len(data), "chars": size, "output": str(output)}


class BoundedLLM(LLMProvider):
    def __init__(self, config, max_calls):
        super().__init__(config)
        self.max_calls, self.calls = max_calls, 0
        self._budget_lock = threading.Lock()

    def complete(self, *args, **kwargs):
        with self._budget_lock:
            if self.calls >= self.max_calls:
                raise RuntimeError("Evaluation LLM call budget exhausted")
            self.calls += 1
        return super().complete(*args, **kwargs)


def score_case(fragments, case):
    expected = set(case.get("expected_sources", []))
    delivered = set(s for f in fragments for s in f.source_ids)
    text = "\n".join(f.content for f in fragments)
    quotes = case.get("expected_quotes", [])
    return {"source_recall": len(expected & delivered) / len(expected) if expected else None,
        "quote_coverage": sum(q in text for q in quotes) / len(quotes) if quotes else None,
        "chars": len(text), "count": len(fragments),
        "fragments": [f.model_dump(mode="json") for f in fragments]}


def run(snapshot, cases_path, output, *, env_file, max_calls, batch_chars,
        budget_chars, limit, repeats, flush_pending=False, store_run=None):
    if min(max_calls, batch_chars, budget_chars, limit, repeats) <= 0:
        raise ValueError("Evaluation budgets and repeats must be positive")
    if store_run and flush_pending:
        raise ValueError("Reusing stores cannot also force pending events closed")
    snapshot, output = Path(snapshot), Path(output)
    if output.exists():
        raise ValueError("Use a new run directory; evaluation does not overwrite results")
    output.mkdir(parents=True)
    items = [MemoryExperience.model_validate(x) for x in json.loads(
        (snapshot / "experiences.json").read_text(encoding="utf-8"))]
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256((snapshot / "experiences.json").read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Snapshot content changed since preparation")
    if not items or len({item.subject_id for item in items}) != 1:
        raise ValueError("Evaluation requires one nonempty subject snapshot")
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    if not cases or any(not c.get("query", "").strip() or not c.get("expected_sources")
                        or not c.get("expected_quotes") for c in cases):
        raise ValueError("Every case needs a query, reviewed gold sources and quotes")
    all_sources = {s for item in items for s in item.source_ids}
    for case in cases:
        if not set(case.get("expected_sources", [])).issubset(all_sources):
            raise ValueError(f"Gold source outside snapshot: {case['id']}")
        gold_text = "\n".join(item.text for item in items
            if set(item.source_ids).intersection(case.get("expected_sources", [])))
        if any(q not in gold_text for q in case.get("expected_quotes", [])):
            raise ValueError(f"Gold quote not in its source: {case['id']}")
    config = REMSConfig(_env_file=env_file)
    store_path = Path(store_run).resolve() if store_run else output.resolve()
    if store_run and not all((store_path / p).exists() for p in ("long.db", "long-vectors", "recent")):
        raise ValueError("Reuse requires an existing completed evaluation store")
    config.storage.database_url = "sqlite:///" + (store_path / "long.db").as_posix()
    config.storage.qdrant_path = str(store_path / "long-vectors")
    config.storage.qdrant_url = ":memory:"
    config.recent_memory.enabled = True
    config.recent_memory.path = str(store_path / "recent")
    config.memory_refinement.enabled = False
    config.context_window, config.chars_per_token = 88000, 1.5
    config.event_min_chars, config.event_len_factor = 2000, 1.5
    config.recall_event_index_level = "L1"
    config.recall_reinforce_enabled = False
    config.recall_query_compress_enabled = False
    config.ingest_llm_session_enabled = False
    config.llm.max_retries = 0
    llm = BoundedLLM(config, max_calls)
    pipe = REMSPipeline.from_config(config, llm=llm)
    save(output / "manifest.json", {"snapshot_sha256": manifest["sha256"],
        "stores": str(store_path), "reused_stores": bool(store_run),
        "cases": cases, "event_min_chars": config.event_min_chars,
        "event_len_factor": config.event_len_factor, "len_msg": config.len_msg,
        "index_level": config.recall_event_index_level})
    timings, results = [], []
    try:
        batch, size = [], 0
        batches = []
        for item in items:
            batch.append(item)
            size += len(item.text)
            if size >= batch_chars:
                batches.append(batch)
                batch, size = [], 0
        if batch:
            batches.append(batch)
        if store_run:
            previous = json.loads((store_path / "manifest.json").read_text(encoding="utf-8"))
            if previous["snapshot_sha256"] != manifest["sha256"]:
                raise ValueError("Reused stores belong to a different snapshot")
            batches = []
        for index, batch in enumerate(batches):
            t0 = time.perf_counter()
            try:
                result = pipe.ingest_batch(MemoryBatch(subject_id=items[0].subject_id, experiences=tuple(batch)))
            except MemoryIngestError as exc:
                result = exc.result  # Persist actual failure evidence before aborting below.
            timings.append({"batch": index, "seconds": time.perf_counter()-t0,
                "stores": pipe.last_ingest_timings,
                "chars": sum(len(x.text) for x in batch), "errors": result.errors,
                "sealed": len(result.sealed_event_ids), "pending": result.unclosed_count})
            save(output / "ingest.json", timings)
            print(f"batch {index+1}/{len(batches)}: {len(result.sealed_event_ids)} events, {len(result.errors)} errors", flush=True)
            if result.errors:
                raise RuntimeError("Ingestion failed; see ingest.json. No quality score fabricated.")
        if flush_pending:
            events = pipe.metabolism_service.process_input("", force_save=True,
                memory=MemoryExperience(subject_id=items[0].subject_id, text=""))
            for event in events:
                pipe.recall_pipeline.index_event(event)
        # Retrieval timing excludes provider/model cold start; report it separately.
        warm_start = time.perf_counter()
        pipe.recall(items[0].subject_id, cases[0]["query"], budget_chars=budget_chars, reinforce=False)
        warm_seconds = time.perf_counter() - warm_start
        for case in cases:
            subject, query = items[0].subject_id, case["query"]
            kwargs = {"object_id": case.get("object_id")}
            arms = {
                "recent_only": lambda: merge_memories([], pipe.recent_memory.recall(subject, query, **kwargs),
                    budget_chars=budget_chars, limit=limit),
                "long_l1_only": lambda: pipe.recall_pipeline.recall(subject, query,
                    budget_chars=budget_chars, limit=limit, reinforce=False, **kwargs),
                "dual": lambda: pipe.recall(subject, query, budget_chars=budget_chars,
                    limit=limit, reinforce=False, **kwargs),
            }
            for name, operation in arms.items():
                elapsed, output_fragments = [], ()
                for _ in range(repeats):
                    start = time.perf_counter()
                    output_fragments = operation()
                    elapsed.append((time.perf_counter()-start)*1000)
                results.append({"case_id": case["id"], "query": query, "arm": name,
                    "errors": list(pipe.last_recall_errors) if name == "dual" else [],
                    "latency_ms": elapsed, **score_case(output_fragments, case)})
            save(output / "results.json", results)
        summary = {}
        for name in ("recent_only", "long_l1_only", "dual"):
            rows = [r for r in results if r["arm"] == name]
            latencies = sorted(t for row in rows for t in row["latency_ms"])
            summary[name] = {key: statistics.mean(r[key] for r in rows if r[key] is not None)
                for key in ("source_recall", "quote_coverage")}
            summary[name].update({"p50_ms": statistics.median(latencies),
                "p95_ms": latencies[min(len(latencies)-1, int(len(latencies)*.95))]})
        summary["run"] = {"cases": len(cases), "segments": len(items),
            "sealed_events": len(pipe.event_repo.list_all()),
            "pending": len(pipe.meta_repo.get_unclosed_events()), "warmup_seconds": warm_seconds,
            "forced_flush": flush_pending, "budget_chars": budget_chars, "limit": limit,
            "stores": str(store_path), "reused_stores": bool(store_run),
            "recent_disk_bytes": pipe.recent_memory.disk_bytes(),
            "llm_calls": llm.calls, "model": config.llm.task_models.default,
            "mem0_version": "2.2.1", "native_embedder": config.recent_memory.embedder or config.embedding.model_name,
            "scope": "retrieved evidence only; no generated-answer accuracy claim"}
        save(output / "summary.json", summary)
        return summary
    finally:
        save(output / "llm-usage.json", [asdict(x) for x in llm.invocation_history()])
        pipe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source-db", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--subject", default="stone")
    p.add_argument("--jshi-root", help="JShi checkout; defaults to adjacent JShi")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=18000)
    r = sub.add_parser("run")
    r.add_argument("--snapshot", required=True)
    r.add_argument("--cases", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--env-file")
    r.add_argument("--live", action="store_true")
    r.add_argument("--max-llm-calls", type=int, default=30)
    r.add_argument("--batch-chars", type=int, default=3000)
    r.add_argument("--budget-chars", type=int, default=4000)
    r.add_argument("--limit", type=int, default=6)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--flush-pending", action="store_true")
    r.add_argument("--store-run", help="Reuse a prior test run's stores; skip ingestion/model calls")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source_db, args.out, subject=args.subject,
            limit=args.limit, max_chars=args.max_chars, jshi_root=args.jshi_root)
    else:
        if not args.live:
            parser.error("run requires --live: long-event ingestion calls the configured model")
        result = run(args.snapshot, args.cases, args.out, env_file=args.env_file,
            max_calls=args.max_llm_calls, batch_chars=args.batch_chars, budget_chars=args.budget_chars,
            limit=args.limit, repeats=args.repeats, flush_pending=args.flush_pending,
            store_run=args.store_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
