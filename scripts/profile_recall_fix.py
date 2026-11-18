"""Component timings on the same frozen stores, excluding cold initialization."""
import time
import argparse
import json
from pathlib import Path
from diagnose_dual_recall import config_for, save
from rems.pipeline import REMSPipeline

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    pipe=REMSPipeline.from_config(config_for(a.audit))
    report=[]
    try:
        pipe.prepare_recall()
        query='匠石项目和记忆项目之间应该怎样对接？'
        pipe.recall('stone',query,limit=6,budget_chars=4000,reinforce=False)
        timings={}
        def wrap(obj,name,label):
            original=getattr(obj,name)
            def call(*args,**kwargs):
                start=time.perf_counter()
                try: return original(*args,**kwargs)
                finally: timings[label]=timings.get(label,0)+(time.perf_counter()-start)*1000
            setattr(obj,name,call)
        recall=pipe.recall_pipeline
        for name in ('_cascade_candidate_ids','_event_candidates','_fill_budget'):
            wrap(recall,name,name)
        wrap(recall._reranker,'rerank','rerank')
        wrap(recall._vector_store,'search','qdrant')
        wrap(recall._embedding,'embed_query','embedding')
        wrap(pipe.recent_memory,'recall','recent')
        for _ in range(3):
            timings={}
            start=time.perf_counter()
            pipe.recall('stone',query,limit=6,budget_chars=4000,reinforce=False)
            timings['total']=(time.perf_counter()-start)*1000
            report.append(timings)
            print(timings,flush=True)
        save(Path(a.out),report)
    finally:
        pipe.close()
