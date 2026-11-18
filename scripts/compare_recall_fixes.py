"""Paired real-data A/B: identical stores, questions, limits and warm process."""
import argparse
import hashlib
import json
import statistics
import shutil
import time
from threadpoolctl import ThreadpoolController
from pathlib import Path
from diagnose_dual_recall import config_for, save
from rems.pipeline import REMSPipeline


def compare(a):
    output=Path(a.out)
    output.mkdir(parents=True,exist_ok=False)
    if a.isolate_recent:
        # Closed test store only. Independent copies contain identical native IDs,
        # payloads and vectors; do not open two clients on one Qdrant directory.
        shutil.copytree(Path(a.audit)/"recent", output/"recent-worker")
    cases_path=Path(a.cases)
    cases=json.loads(cases_path.read_text(encoding="utf-8"))
    if a.paraphrases:
        rewritten=("匠石和记忆模块连接时，应当以哪个项目作为接口？",
            "之前选择的光影包叫什么？", "怎么把原来的版本升级到新版？",
            "之前要求将说的话记录到什么文件里？", "被人叫到名字时，匠石该接哪一句话？",
            "家里几个人正在聊天时，语音助手要不要插话？")
        cases += [{**c,"id":c["id"]+"_paraphrase","query":q} for c,q in zip(cases,rewritten)]
    pipe=REMSPipeline.from_config(config_for(a.audit))
    baseline_recent = pipe.recent_memory
    recent_worker = None
    if a.isolate_recent:
        from rems.services.recent_process import Mem0RecentProcess
        worker_config = config_for(a.audit)
        worker_config.recent_memory.path = str((output/"recent-worker").resolve())
        recent_worker = Mem0RecentProcess(worker_config)
    report={"case_sha256":hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "replay_sha256":hashlib.sha256((Path(a.audit)/"recent-replay.json").read_bytes()).hexdigest(),
        "limit":6,"budget_chars":4000,"repeats":a.repeats,"cases":[]}
    report["blas_threads_after"] = a.blas_threads
    report["recent_process_isolation_after"] = a.isolate_recent
    recall=pipe.recall_pipeline
    ranker=recall._reranker
    complete=recall._vector_store.search_complete
    controller = None
    def run(arm,case):
        recall._reranker=ranker if arm=="after" else None
        recall._vector_store.search_complete=complete if arm=="after" else None
        recall._vector_store._local = arm == "after"
        pipe.recent_memory = recent_worker if arm == "after" and recent_worker else baseline_recent
        with controller.limit(limits=a.blas_threads if arm=="after" else None, user_api="blas"):
            start=time.perf_counter()
            found=pipe.recall("stone",case["query"],limit=6,budget_chars=4000,reinforce=False)
            elapsed=(time.perf_counter()-start)*1000
        return elapsed,found
    try:
        # Cold loading is a startup cost, separately reported, not hidden in warm p50.
        start=time.perf_counter()
        ranker._ensure()
        baseline_recent.prepare()
        if recent_worker:
            recent_worker.prepare()
        controller = ThreadpoolController()
        report["cold_prepare_ms"]=(time.perf_counter()-start)*1000
        for arm in ("before","after"):
            run(arm,cases[0])
        for case in cases:
            row={"id":case["id"],"query":case["query"],"arms":{}}
            times={"before":[],"after":[]}
            for repeat in range(a.repeats):
                for arm in (("before","after") if repeat%2==0 else ("after","before")):
                    elapsed,found=run(arm,case)
                    times[arm].append(elapsed)
                    gold=set(case["expected_sources"])
                    row["arms"][arm]={"gold_ranks":[i for i,f in enumerate(found,1) if gold.intersection(f.source_ids)],
                        "errors":list(pipe.last_recall_errors),
                        "quote_coverage":sum(any(q in f.content for f in found) for q in case["expected_quotes"])/len(case["expected_quotes"]),
                        "fragments":[f.model_dump(mode="json") for f in found]}
            for arm in times:
                row["arms"][arm].update(ms=times[arm],median_ms=statistics.median(times[arm]))
            report["cases"].append(row)
            save(output/"comparison.json",report)
            print(case["id"],{k:(round(v["median_ms"]),v["gold_ranks"],v["quote_coverage"]) for k,v in row["arms"].items()},flush=True)
        report["summary"]={arm:{
            "source_hits":sum(bool(row["arms"][arm]["gold_ranks"]) for row in report["cases"]),
            "case_count":len(cases),
            "median_ms":statistics.median(t for row in report["cases"] for t in row["arms"][arm]["ms"])
        } for arm in ("before","after")}
        save(output/"comparison.json",report)
        print(report["summary"],flush=True)
    finally:
        if recent_worker:
            recent_worker.close()
        pipe.recent_memory = baseline_recent
        pipe.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit",required=True)
    p.add_argument("--cases",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--paraphrases",action="store_true")
    p.add_argument("--blas-threads",type=int)
    p.add_argument("--isolate-recent",action="store_true")
    compare(p.parse_args())
