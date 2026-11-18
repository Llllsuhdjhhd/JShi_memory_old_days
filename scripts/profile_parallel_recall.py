"""Real-data timings with both stores running concurrently, on an isolated copy."""
import argparse
import cProfile
import hashlib
import gc
import io
import json
import os
from pathlib import Path
import pstats
import shutil
import statistics
import threading
import time

from diagnose_dual_recall import config_for, save
from rems.pipeline import REMSPipeline
from rems.services import recent_process

native_worker = recent_process._worker


def profiled_worker(connection, configuration):
    from rems.services.recent_memory import Mem0RecentMemory
    original = Mem0RecentMemory.recall
    serial = 0
    def recall(self, subject, query, **kwargs):
        nonlocal serial
        serial += 1
        profile = cProfile.Profile()
        result = profile.runcall(original, self, subject, query, **kwargs)
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(45)
        Path(os.environ['REMS_PROFILE_OUTPUT'],f'recent-{serial}.txt').write_text(
            query+'\n'+stream.getvalue(),encoding='utf-8')
        return result
    Mem0RecentMemory.recall = recall
    native_worker(connection, configuration)


def measure(a):
    output=Path(a.out).resolve()
    output.mkdir(parents=True,exist_ok=False)
    audit=Path(a.audit).resolve()
    if not (audit/'rehearsal.json').is_file() or '.jshi' in audit.parts:
        raise ValueError('Only a completed isolated rehearsal may be measured')
    # Copy only closed isolated stores. Never open the production Qdrant path.
    for name in ('long.db','rehearsal.json','recent-replay.json'):
        shutil.copy2(audit/name,output/name)
    for name in ('recent','long-vectors'):
        shutil.copytree(audit/name,output/name)
    config=config_for(output)
    config.recent_memory.process_isolation=True
    config.recall_rerank_threads=a.threads
    if a.profile:
        os.environ['REMS_PROFILE_OUTPUT']=str(output)
        recent_process._worker=profiled_worker
    pipe=REMSPipeline.from_config(config)
    cases=json.loads(Path(a.cases).read_text(encoding='utf-8'))
    if a.only:
        cases=[c for c in cases if c['id'] in a.only.split(',')]
    report={'parallel':True,'profile_overhead':a.profile,'cases':[],
            'case_sha256':hashlib.sha256(Path(a.cases).read_bytes()).hexdigest(),
            'replay_sha256':hashlib.sha256((audit/'recent-replay.json').read_bytes()).hexdigest()}
    timings={}
    spans={}
    ranker_inputs=[]
    collections=[]
    gc_starts={}
    def gc_timing(phase,info):
        key=(threading.get_ident(),info['generation'])
        moment=time.perf_counter()
        if phase=='start':
            gc_starts[key]=moment
        elif key in gc_starts:
            start=gc_starts.pop(key)
            collections.append({'generation':info['generation'],'thread':key[0],
                'start_ms':(start-origin)*1000,'duration_ms':(moment-start)*1000,
                'collected':info['collected']})
    gc.callbacks.append(gc_timing)
    origin=0
    def wrap(obj,name,label):
        original=getattr(obj,name)
        def call(*args,**kwargs):
            begin=time.perf_counter()
            try:
                if a.profile and label=='long':
                    profile=cProfile.Profile()
                    result=profile.runcall(original,*args,**kwargs)
                    stream=io.StringIO()
                    pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(50)
                    key=hashlib.sha256(str(args[1]).encode()).hexdigest()[:12]
                    (output/f'long-{key}.txt').write_text(str(args[1])+'\n'+stream.getvalue(),encoding='utf-8')
                    return result
                return original(*args,**kwargs)
            finally:
                end=time.perf_counter()
                timings[label]=timings.get(label,0)+(end-begin)*1000
                if label in ('long','recent'):
                    spans[label]=[(begin-origin)*1000,(end-origin)*1000]
        setattr(obj,name,call)
    try:
        start=time.perf_counter()
        pipe.prepare_recall()
        report['cold_prepare_ms']=(time.perf_counter()-start)*1000
        rec=pipe.recall_pipeline
        native_rank=rec._reranker.rerank
        def capture_rank(query,texts):
            scores=native_rank(query,texts)
            ranker_inputs.append({'query':query,'texts':list(texts),'scores':scores})
            return scores
        rec._reranker.rerank=capture_rank
        wrap(rec,'recall','long')
        wrap(pipe.recent_memory,'recall','recent')
        for name in ('_cascade_candidate_ids','_fill_budget','_event_candidates'):
            wrap(rec,name,name)
        wrap(rec._embedding,'embed_query','embedding')
        wrap(rec._reranker,'rerank','rerank')
        wrap(rec._vector_store,'search_complete','vector_complete')
        wrap(rec._vector_store,'search','vector_search')
        wrap(pipe,'_record_recall_trace','trace')
        pipe.recall('stone',cases[0]['query'],limit=6,budget_chars=4000,reinforce=False)
        for case in cases:
            row={'id':case['id'],'query':case['query'],'runs':[]}
            for repeat in range(a.repeats):
                timings={}
                spans={}
                collections=[]
                origin=time.perf_counter()
                found=pipe.recall('stone',case['query'],limit=6,budget_chars=4000,reinforce=False)
                elapsed=(time.perf_counter()-origin)*1000
                gold=set(case['expected_sources'])
                row['runs'].append({'total_ms':elapsed,'components_ms':dict(timings),'spans_ms':dict(spans),
                    'gc_events':list(collections),
                    'errors':list(pipe.last_recall_errors),
                    'gold_ranks':[i for i,f in enumerate(found,1) if gold.intersection(f.source_ids)],
                    'quote_coverage':sum(any(q in f.content for f in found)
                        for q in case['expected_quotes'])/len(case['expected_quotes']),
                    'fragments':[f.model_dump(mode='json') for f in found]})
            row['median_ms']=statistics.median(r['total_ms'] for r in row['runs'])
            report['cases'].append(row)
            save(output/'timings.json',report)
            save(output/'ranker-inputs.json',ranker_inputs)
            print(case['id'],round(row['median_ms']),
                  {k:round(statistics.median(r['components_ms'].get(k,0) for r in row['runs']))
                   for k in timings},flush=True)
        return report
    finally:
        gc.callbacks.remove(gc_timing)
        pipe.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('audit','cases','out'):
        p.add_argument('--'+arg,required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--only')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--threads',type=int,default=4)
    measure(p.parse_args())
