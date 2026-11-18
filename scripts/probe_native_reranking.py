"""Offline ranking experiment against the isolated, real-data rehearsal."""
import argparse
import json
import time
from pathlib import Path
from diagnose_dual_recall import config_for, save
from rems.pipeline import REMSPipeline
from mem0.reranker.sentence_transformer_reranker import SentenceTransformerReranker


def main(root, cases_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ranker = SentenceTransformerReranker({
        "model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "device": "cpu",
        "batch_size": 16, "show_progress_bar": False})
    pipe = REMSPipeline.from_config(config_for(root))
    report = []
    try:
        for case in json.loads(Path(cases_path).read_text(encoding="utf-8")):
            captured = []
            original = pipe.recall_pipeline._arrange
            def capture(admitted, wanted, **kw):
                ordered = original(admitted, wanted, **kw)
                captured.extend(ordered[:100])
                return ordered
            pipe.recall_pipeline._arrange = capture
            try:
                pipe.recall_pipeline.recall("stone", case["query"], limit=6, reinforce=False)
            finally:
                pipe.recall_pipeline._arrange = original
            docs = [{"id": c.event_id, "memory": c.levels.get("L1") or c.raw,
                     "sources": c.source_ids} for c in captured]
            start = time.perf_counter()
            ranked = ranker.rerank(case["query"], docs)
            row = {"id": case["id"], "ms": (time.perf_counter()-start)*1000,
                   "gold_ranks": [i for i,d in enumerate(ranked,1)
                                  if set(case["expected_sources"]).intersection(d["sources"])],
                   "top": ranked[:6], "candidates": docs}
            report.append(row)
            save(output/"ranking.json", report)
            print(row["id"], round(row["ms"]), row["gold_ranks"],flush=True)
    finally:
        pipe.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit",required=True)
    p.add_argument("--cases",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    main(a.audit,a.cases,a.out)
