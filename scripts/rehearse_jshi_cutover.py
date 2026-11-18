"""Exercise both stores on audit copies with real embeddings, without any LLM.

No JShi configuration, cursor, live SQLite database or live Qdrant is opened.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rems.config import REMSConfig
from rems.pipeline import REMSPipeline
from rems.port import MemoryBatch, MemoryExperience


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rehearse(audit_dir, cases_path=None, *, resume=False):
    root = Path(audit_dir).resolve()
    manifest = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    if manifest.get("live_databases_modified") is not False:
        raise ValueError("Expected an isolated audit snapshot")
    if not resume and ((root / "recent").exists() or (root / "long-vectors").exists()):
        raise ValueError("Use a fresh audit snapshot; rehearsal stores already exist")
    progress = root / "rehearsal-progress.json"
    if resume and not progress.is_file():
        raise ValueError("Resume requires this audit's saved rehearsal progress")
    # Reject paths that could redirect the rehearsal into another directory.
    for name in ("long.db", "recent-replay.json"):
        if (root / name).resolve().parent != root:
            raise ValueError("Snapshot files must remain inside the audit directory")
    config = REMSConfig(_env_file=None,
        llm={"api_key": "unused-no-llm", "base_url": "http://127.0.0.1:9/v1"},
        storage={"database_url": "sqlite:///" + (root / "long.db").as_posix(),
                 "qdrant_path": str(root / "long-vectors"), "qdrant_url": ":memory:"},
        recent_memory={"enabled": True, "path": str(root / "recent"), "process_isolation": False},
        memory_refinement={"enabled": False}, recall_event_index_level="L1",
        recall_query_compress_enabled=False, recall_reinforce_enabled=False)
    class NoLLM:
        def complete(self, *a, **kw):
            raise AssertionError("Cutover rehearsal must not call an LLM")
        complete_json = complete
    pipe = REMSPipeline.from_config(config, llm=NoLLM())
    report = (json.loads(progress.read_text(encoding="utf-8")) if resume
              else {"llm_calls": 0, "live_qdrant_opened": False})
    start = time.perf_counter()
    try:
        if not resume:
            scanned, written = pipe.recall_pipeline.backfill_index(manifest["subject"])
        else:
            scanned, written = report["long_index"]["scanned"], report["long_index"]["points_written"]
        events = pipe.event_repo.list_all()
        missing, representations = [], Counter()
        for event in events:
            if event.is_tombstoned or event.refinement_hidden_by:
                continue
            point = pipe.recall_pipeline._vector_store.get(event.event_id)
            if point is None:
                missing.append(event.event_id)
            else:
                representations[point["representation_level"]] += 1
        report["long_index"] = {"scanned": scanned, "points_written": written,
            "event_representations": representations, "missing_events": missing,
            "seconds": report.get("long_index", {}).get("seconds", time.perf_counter() - start)}
        save(root / "rehearsal-progress.json", report)
        print("long index:", scanned, "events; missing:", len(missing), flush=True)
        items = [MemoryExperience.model_validate(x) for x in json.loads(
            (root / "recent-replay.json").read_text(encoding="utf-8"))]
        completed = report.get("recent_replay", {}).get("completed", 0) if resume else 0
        previous_seconds = report.get("recent_replay", {}).get("seconds", 0) if resume else 0
        if resume:
            report["resume"] = {"completed_before": completed, "elapsed_before": previous_seconds,
                                "reason": "Continue interrupted isolated rehearsal with batch deduplication"}
        resumed_at = time.perf_counter()
        start = resumed_at - previous_seconds
        for offset in range(completed, len(items), 100):
            batch = MemoryBatch(subject_id=manifest["subject"], experiences=tuple(items[offset:offset+100]))
            # Import historical text into Mem0 only. Do not re-ingest old long events.
            pipe.recent_memory.ingest_batch(batch)
            report["recent_replay"] = {"completed": min(offset+100, len(items)),
                "total": len(items), "seconds": time.perf_counter() - start}
            save(root / "rehearsal-progress.json", report)
            if offset % 1000 == 0:
                print("recent replay:", min(offset+100, len(items)), "/", len(items), flush=True)
        records = list(pipe.recent_memory._records())
        expected = {item.segment_id for item in items}
        actual = {r.payload.get("rems_segment") for r in records}
        report["recent_replay"].update({"native_records": len(records),
            "missing_segments": sorted(expected-actual), "disk_bytes": pipe.recent_memory.disk_bytes()})
        if resume:
            report["resume"]["seconds"] = time.perf_counter() - resumed_at
        report["queries"] = []
        if cases_path:
            from evaluate_dual_memory import score_case
            for case in json.loads(Path(cases_path).read_text(encoding="utf-8")):
                begin = time.perf_counter()
                fragments = pipe.recall(manifest["subject"], case["query"], limit=6,
                    budget_chars=4000, reinforce=False)
                report["queries"].append({"id": case["id"], "query": case["query"],
                    "ms": (time.perf_counter()-begin)*1000,
                    "errors": list(pipe.last_recall_errors), **score_case(fragments, case)})
        save(root / "rehearsal.json", report)
        return {k:v for k,v in report.items() if k != "queries"}
    finally:
        pipe.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", required=True)
    p.add_argument("--cases")
    p.add_argument("--resume", action="store_true", help="Continue this isolated audit after its process has stopped")
    args = p.parse_args()
    print(json.dumps(rehearse(args.audit, args.cases, resume=args.resume), ensure_ascii=False, indent=2))
