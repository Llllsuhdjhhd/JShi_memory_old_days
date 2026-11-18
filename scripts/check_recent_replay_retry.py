"""Compare native lookup methods and verify timezone-safe replay on a test copy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rems.config import REMSConfig
from rems.port import MemoryBatch, MemoryExperience
from rems.services.recent_memory import AGENT, Mem0RecentMemory


def check(root):
    root = Path(root).resolve()
    if not (root / "rehearsal.json").is_file():
        raise ValueError("Requires a completed isolated rehearsal, after its process closes")
    items = [MemoryExperience.model_validate(x) for x in json.loads(
        (root / "recent-replay.json").read_text(encoding="utf-8"))]
    sampled = items[::max(1, len(items)//25)][:25]
    config = REMSConfig(_env_file=None, recent_memory={"path": str(root / "recent")})
    recent = Mem0RecentMemory(config, background=False)
    try:
        memory = recent._get_memory()
        scope = {"agent_id": AGENT, "user_id": hashlib.sha256(items[0].subject_id.encode()).hexdigest()}
        begin = time.perf_counter()
        singles = {}
        for item in sampled:
            rows = memory.get_all(filters={**scope, "rems_segment": item.segment_id},
                                  top_k=1, show_expired=True)["results"]
            singles.update((r["metadata"]["rems_segment"], r["memory"]) for r in rows)
        single_ms = (time.perf_counter()-begin)*1000
        begin = time.perf_counter()
        rows = memory.get_all(filters={**scope, "rems_segment": {"in": [i.segment_id for i in sampled]}},
                              top_k=len(sampled), show_expired=True)["results"]
        batched = {r["metadata"]["rems_segment"]: r["memory"] for r in rows}
        batch_ms = (time.perf_counter()-begin)*1000
        if singles != batched or len(batched) != len(sampled):
            raise AssertionError("Native batched lookup changed the replay identity set")
        before = {str(r.id) for r in recent._records()}
        replay = [i.model_copy(update={"occurred_at": i.occurred_at.astimezone().replace(tzinfo=None)})
                  for i in sampled]
        recent.ingest_batch(MemoryBatch(subject_id=items[0].subject_id, experiences=tuple(replay)))
        after = {str(r.id) for r in recent._records()}
        if before != after:
            raise AssertionError("Live-style retry changed stored Mem0 identities")
        result = {"actual_records": len(after), "sampled_records": len(sampled),
            "native_single_lookup_ms": single_ms, "native_batch_lookup_ms": batch_ms,
            "same_results": True, "timezone_retry_same_ids": True,
            "scope": "one paired native lookup run, excludes embeddings and writes; not end-to-end speedup"}
        (root / "retry-and-lookup.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        recent.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    print(json.dumps(check(parser.parse_args().audit), indent=2))
