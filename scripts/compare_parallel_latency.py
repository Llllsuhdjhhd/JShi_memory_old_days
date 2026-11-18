"""Paired CPU-setting benchmark: identical real data and concurrent native stores."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

from diagnose_dual_recall import config_for, save
from rems.pipeline import REMSPipeline
from rems.recall.providers import BgeReranker


def evidence(fragments):
    # Scores can contain age-dependent signals; compare actual model evidence.
    return [{k:getattr(f,k) for k in ('event_id','type','kind','object_id','content',
        'text','source_ids','summary_level','representation_level')} for f in fragments]


def compare(a):
    out=Path(a.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    audit=Path(a.audit).resolve()
    if not (audit/'rehearsal.json').is_file() or '.jshi' in audit.parts:
        raise ValueError('Use a closed isolated rehearsal, never the live stores')
    for name in ('long.db','rehearsal.json','recent-replay.json'):
        shutil.copy2(audit/name,out/name)
    for name in ('recent','long-vectors'):
        shutil.copytree(audit/name,out/name)
    config=config_for(out)
    config.recent_memory.process_isolation=True
    cases_path=Path(a.cases)
    cases=json.loads(cases_path.read_text(encoding='utf-8'))
    if a.paraphrases:
        queries=('匠石和记忆模块连接时，应当以哪个项目作为接口？',
            '之前选择的光影包叫什么？','怎么把原来的版本升级到新版？',
            '之前要求将说的话记录到什么文件里？','被人叫到名字时，匠石该接哪一句话？',
            '家里几个人正在聊天时，语音助手要不要插话？')
        cases += [{**c,'id':c['id']+'_paraphrase','query':q} for c,q in zip(cases[:6],queries)]
    report={'case_sha256':hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        'replay_sha256':hashlib.sha256((audit/'recent-replay.json').read_bytes()).hexdigest(),
        'before_threads':4,'after_threads':a.threads,'parallel_both_arms':True,
        'limit':6,'budget_chars':4000,'repeats':a.repeats,'cases':[]}
    pipe=REMSPipeline.from_config(config)
    timings={}
    spans={}
    origin=0
    ranks={}
    def timed(obj,name,label):
        original=getattr(obj,name)
        def call(*args,**kwargs):
            start=time.perf_counter()
            try:
                return original(*args,**kwargs)
            finally:
                end=time.perf_counter()
                timings[label]=timings.get(label,0)+(end-start)*1000
                spans[label]=[(start-origin)*1000,(end-origin)*1000]
        setattr(obj,name,call)
    try:
        begin=time.perf_counter()
        pipe.prepare_recall()
        before=pipe.recall_pipeline._reranker
        after=BgeReranker(config.recall_rerank_model,
            onnx_file=config.recall_rerank_onnx_file,threads=a.threads)
        after._ensure()
        report['cold_prepare_ms']=(time.perf_counter()-begin)*1000
        for arm,ranker in [('before',before),('after',after)]:
            original=ranker.rerank
            def capture(query,texts,original=original,arm=arm):
                begin=time.perf_counter()
                scores=original(query,texts)
                timings['rerank']=(time.perf_counter()-begin)*1000
                ranks[arm]={'input_sha256':hashlib.sha256(json.dumps([query,list(texts)],ensure_ascii=False).encode()).hexdigest(),
                            'scores':scores}
                return scores
            ranker.rerank=capture
        timed(pipe.recall_pipeline,'recall','long')
        timed(pipe.recent_memory,'recall','recent')
        timed(pipe.recall_pipeline._embedding,'embed_query','embedding')
        timed(pipe.recall_pipeline._vector_store,'search_complete','vector_complete')
        for ranker in (before,after):
            pipe.recall_pipeline._reranker=ranker
            pipe.recall('stone',cases[0]['query'],limit=6,budget_chars=4000,reinforce=False)
        for case in cases:
            row={'id':case['id'],'query':case['query'],'arms':{'before':[],'after':[]}}
            for repeat in range(a.repeats):
                for arm in (('before','after') if repeat%2==0 else ('after','before')):
                    pipe.recall_pipeline._reranker=before if arm=='before' else after
                    timings={}
                    spans={}
                    origin=time.perf_counter()
                    found=pipe.recall('stone',case['query'],limit=6,budget_chars=4000,reinforce=False)
                    elapsed=(time.perf_counter()-origin)*1000
                    row['arms'][arm].append({'total_ms':elapsed,'components_ms':dict(timings),
                        'spans_ms':dict(spans),'ranker':ranks[arm],
                        'errors':list(pipe.last_recall_errors),
                        'gold_ranks':[i for i,f in enumerate(found,1) if set(case['expected_sources']).intersection(f.source_ids)],
                        'quote_coverage':sum(any(q in f.content for f in found)
                            for q in case['expected_quotes'])/len(case['expected_quotes']),
                        'evidence':evidence(found)})
            baseline=row['arms']['before'][0]
            row['identical_evidence']=all(r['evidence']==baseline['evidence'] for values in row['arms'].values() for r in values)
            row['identical_ranker_inputs']=all(r['ranker']['input_sha256']==baseline['ranker']['input_sha256']
                for values in row['arms'].values() for r in values)
            row['max_ranker_score_delta']=max(abs(s-t) for values in row['arms'].values() for r in values
                for s,t in zip(r['ranker']['scores'],baseline['ranker']['scores']))
            row['median_ms']={arm:statistics.median(r['total_ms'] for r in values) for arm,values in row['arms'].items()}
            report['cases'].append(row)
            save(out/'comparison.json',report)
            print(case['id'],{k:round(v) for k,v in row['median_ms'].items()},
                  'same_evidence',row['identical_evidence'],'score_delta',row['max_ranker_score_delta'],flush=True)
        report['quality_gate']=all(row['identical_evidence'] and row['identical_ranker_inputs']
            and row['max_ranker_score_delta'] <= 1e-6
            and all(not r['errors'] for runs in row['arms'].values() for r in runs)
            for row in report['cases'])
        report['summary']={arm:{
            'median_ms':statistics.median(r['total_ms'] for row in report['cases'] for r in row['arms'][arm]),
            'p95_ms':statistics.quantiles([r['total_ms'] for row in report['cases'] for r in row['arms'][arm]],n=100,method='inclusive')[94],
            'source_hits':sum(bool(row['arms'][arm][0]['gold_ranks']) for row in report['cases']),
            'case_count':len(cases)} for arm in ('before','after')}
        save(out/'comparison.json',report)
        print('quality_gate',report['quality_gate'],report['summary'],flush=True)
    finally:
        pipe.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('audit','cases','out'):
        p.add_argument('--'+arg,required=True)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--paraphrases',action='store_true')
    compare(p.parse_args())
