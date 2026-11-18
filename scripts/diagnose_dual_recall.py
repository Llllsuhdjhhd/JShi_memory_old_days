"""Profile and inspect actual frozen cutover stores; no LLM or live store access."""
import argparse
import cProfile
import hashlib
import io
import json
from pathlib import Path
import pstats
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rems.config import REMSConfig
from rems.pipeline import REMSPipeline
from rems.services.recent_memory import AGENT


def config_for(root):
    root = Path(root).resolve()
    if not (root / "rehearsal.json").is_file():
        raise ValueError("Requires the isolated completed cutover rehearsal")
    return REMSConfig(_env_file=None,
        llm={"api_key": "unused", "base_url": "http://127.0.0.1:9/v1"},
        storage={"database_url": "sqlite:///"+(root/"long.db").as_posix(),
                 "qdrant_path": str(root/"long-vectors"), "qdrant_url": ":memory:"},
        recent_memory={"enabled": True, "path": str(root/"recent"), "process_isolation": False},
        recall_event_index_level="L1", memory_refinement={"enabled": False},
        recall_query_compress_enabled=False, recall_reinforce_enabled=False)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def diagnose(root, cases_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    pipe = REMSPipeline.from_config(config_for(root))
    report = {"cases": [], "profile": {}}
    try:
        pipe.recall("stone", cases[0]["query"], limit=6, budget_chars=4000, reinforce=False)
        events = pipe.event_repo.list_all()
        for case in cases:
            query = case["query"]
            gold = set(case["expected_sources"])
            targets = [e.event_id for e in events if gold.intersection(e.source_ids)]
            row = {"id": case["id"], "query": query, "target_long_event_ids": targets, "arms": {}}
            operations = {
                "recent": lambda: pipe.recent_memory.recall("stone", query),
                "long": lambda: pipe.recall_pipeline.recall("stone", query, limit=20,
                    budget_chars=4000, reinforce=False),
                "dual": lambda: pipe.recall("stone", query, limit=6, budget_chars=4000, reinforce=False)}
            for arm, operation in operations.items():
                times = []
                for _ in range(3):
                    begin = time.perf_counter()
                    found = operation()
                    times.append((time.perf_counter()-begin)*1000)
                row["arms"][arm] = {"ms": times, "median_ms": statistics.median(times),
                    "gold_ranks": [i for i,f in enumerate(found,1) if gold.intersection(f.source_ids)],
                    "quote_coverage": sum(any(q in f.content for f in found) for q in case["expected_quotes"])/len(case["expected_quotes"]),
                    "fragments": [f.model_dump(mode="json") for f in found]}
            if case["id"] in ("interface", "voice"):
                memory = pipe.recent_memory._get_memory()
                native = memory.search(query, filters={"agent_id":AGENT,
                    "user_id":hashlib.sha256(b"stone").hexdigest()}, top_k=300,
                    threshold=0, explain=True)["results"]
                row["native_300_gold"] = [{"rank":i,"row":r} for i,r in enumerate(native,1)
                    if gold.intersection((r.get("metadata") or {}).get("rems_sources",[]))]
                sink = []
                pipe.recall_pipeline.recall("stone", query, limit=100, budget_chars=30000,
                    reinforce=False, candidate_sink=sink)
                row["long_gold_candidates"] = [c.as_dict() for c in sink if c.event_id in targets]
                for arm in ("recent", "long"):
                    profiler = cProfile.Profile()
                    profiler.runcall(operations[arm])
                    stream = io.StringIO()
                    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(30)
                    report["profile"][case["id"]+"_"+arm] = stream.getvalue()
            report["cases"].append(row)
            save(output/"diagnostic.json", report)
            print(case["id"], {a:(round(r["median_ms"],1),r["gold_ranks"]) for a,r in row["arms"].items()},flush=True)
        return report
    finally:
        pipe.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit",required=True)
    p.add_argument("--cases",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    diagnose(a.audit,a.cases,a.out)
