"""Compare current CPU recall with CUDA using identical real data and parallel stores."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

from compare_parallel_latency import evidence
from diagnose_dual_recall import config_for, save
from rems.pipeline import REMSPipeline
from rems.recall.providers import BgeReranker, SentenceTransformerEmbedding
from rems.services.recent_process import Mem0RecentProcess


class ExperimentalGuardedRanker(BgeReranker):
    """Reproduce the rejected CPU-reference experiment; not a production backend."""
    def __init__(self,*args,reference=None,margin=0.0,**kwargs):
        super().__init__(*args,**kwargs)
        self.reference,self.margin=reference,margin
        self.guard_calls=0

    def rerank(self,query,candidates):
        scores=super().rerank(query,candidates)
        ordered=sorted(scores,reverse=True)
        if self.reference is not None and any(ordered[i]-ordered[i+1] < self.margin
                for i in range(min(3,len(ordered)-1))):
            self.guard_calls+=1
            return self.reference.rerank(query,candidates)
        return scores


def compare(a):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; refusing a silently CPU-only GPU benchmark')
    output=Path(a.out).resolve()
    output.mkdir(parents=True,exist_ok=False)
    audit=Path(a.audit).resolve()
    if not (audit/'rehearsal.json').is_file() or '.jshi' in audit.parts:
        raise ValueError('Use a completed isolated rehearsal, never a production vector store')
    for name in ('long.db','rehearsal.json','recent-replay.json'):
        shutil.copy2(audit/name,output/name)
    for name in ('long-vectors','recent'):
        shutil.copytree(audit/name,output/name)
    shutil.copytree(audit/'recent',output/'recent-gpu')
    config=config_for(output)
    config.recall_rerank_threads=8
    if a.cpu_fp32:
        config.recall_rerank_onnx_file=''
        torch.set_num_threads(8)
    config.embedding.device='cpu'
    config.recent_memory.process_isolation=True
    cases=json.loads(Path(a.cases).read_text(encoding='utf-8'))
    if a.paraphrases:
        queries=('匠石和记忆模块连接时，应当以哪个项目作为接口？',
            '之前选择的光影包叫什么？','怎么把原来的版本升级到新版？',
            '之前要求将说的话记录到什么文件里？','被人叫到名字时，匠石该接哪一句话？',
            '家里几个人正在聊天时，语音助手要不要插话？')
        cases += [{**c,'id':c['id']+'_paraphrase','query':q} for c,q in zip(cases[:6],queries)]
    report={'cases':[],'parallel_both_arms':True,'repeats':a.repeats,'limit':6,'budget_chars':4000,
        'case_sha256':hashlib.sha256(Path(a.cases).read_bytes()).hexdigest(),
        'replay_sha256':hashlib.sha256((audit/'recent-replay.json').read_bytes()).hexdigest(),
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__}
    pipe=REMSPipeline.from_config(config)
    gpu_config=config.model_copy(deep=True)
    gpu_config.recent_memory.path=str(output/'recent-gpu')
    gpu_config.embedding.device='cuda'
    gpu_recent=Mem0RecentProcess(gpu_config)
    cpu_recent=pipe.recent_memory
    cpu_embedding=pipe.recall_pipeline._embedding
    cpu_ranker=pipe.recall_pipeline._reranker
    gpu_embedding=SentenceTransformerEmbedding(config.embedding.model_name,
        config.embedding.query_instruction,device='cuda')
    gpu_ranker=ExperimentalGuardedRanker(config.recall_rerank_model,device='cuda',onnx_file=None,
        reference=cpu_ranker if a.guard_margin>0 else None,margin=a.guard_margin)
    report['cpu_guard_margin']=a.guard_margin
    report['cpu_precision']='fp32' if a.cpu_fp32 else 'int8_onnx'
    logits={}
    for arm,ranker in [('cpu',cpu_ranker),('gpu',gpu_ranker)]:
        original=ranker.rerank
        def capture(query,texts,original=original,arm=arm):
            scores=original(query,texts)
            logits[arm]={'scores':scores,'input_sha256':hashlib.sha256(
                json.dumps([query,list(texts)],ensure_ascii=False).encode()).hexdigest()}
            return scores
        ranker.rerank=capture
    timings={}
    spans={}
    origin=0
    def wrap(obj,name,label):
        original=getattr(obj,name)
        def call(*args,**kwargs):
            begin=time.perf_counter()
            try:
                return original(*args,**kwargs)
            finally:
                end=time.perf_counter()
                # GPU rerank may call its CPU reference; report the inclusive
                # outer duration once instead of double-counting nested calls.
                timings[label]=((end-begin)*1000 if label=='rerank' else
                    timings.get(label,0)+(end-begin)*1000)
                spans[label]=[(begin-origin)*1000,(end-origin)*1000]
        setattr(obj,name,call)
    def select(arm):
        pipe.recent_memory=cpu_recent if arm=='cpu' else gpu_recent
        pipe.recall_pipeline._embedding=cpu_embedding if arm=='cpu' else gpu_embedding
        pipe.recall_pipeline._reranker=cpu_ranker if arm=='cpu' else gpu_ranker
    try:
        begin=time.perf_counter()
        pipe.prepare_recall()
        gpu_embedding.embed_query('记忆检索准备')
        gpu_ranker._ensure()
        gpu_info=gpu_recent.prepare()
        report['devices']={'cpu_embedding':str(cpu_embedding._model.device),
            'cpu_recent':cpu_recent.runtime_info,'gpu_embedding':str(gpu_embedding._model.device),
            'gpu_ranker':str(gpu_ranker._model.device),'gpu_recent':gpu_info,
            'gpu_ranker_dtype':str(next(gpu_ranker._model.model.parameters()).dtype)}
        if not all(str(v).startswith('cuda') for v in (gpu_embedding._model.device,
                gpu_ranker._model.device,gpu_info['embedding_device'])):
            raise RuntimeError('One of the intended GPU models did not use CUDA')
        report['cold_prepare_ms']=(time.perf_counter()-begin)*1000
        wrap(pipe.recall_pipeline,'recall','long')
        for store in (cpu_recent,gpu_recent):
            wrap(store,'recall','recent')
        for embedding in (cpu_embedding,gpu_embedding):
            wrap(embedding,'embed_query','embedding')
        for ranker in (cpu_ranker,gpu_ranker):
            wrap(ranker,'rerank','rerank')
        for arm in ('cpu','gpu'):
            select(arm)
            pipe.recall('stone',cases[0]['query'],limit=6,budget_chars=4000,reinforce=False)
        for case in cases:
            row={'id':case['id'],'query':case['query'],'arms':{'cpu':[],'gpu':[]}}
            for repeat in range(a.repeats):
                for arm in (('cpu','gpu') if repeat%2==0 else ('gpu','cpu')):
                    select(arm)
                    timings={}
                    spans={}
                    origin=time.perf_counter()
                    guard_before=gpu_ranker.guard_calls
                    found=pipe.recall('stone',case['query'],limit=6,budget_chars=4000,reinforce=False)
                    elapsed=(time.perf_counter()-origin)*1000
                    gold=set(case['expected_sources'])
                    row['arms'][arm].append({'total_ms':elapsed,'components_ms':dict(timings),
                        'spans_ms':dict(spans),'errors':list(pipe.last_recall_errors),
                        'cpu_guard_used':gpu_ranker.guard_calls>guard_before,
                        'ranker':logits[arm],
                        'gold_ranks':[i for i,f in enumerate(found,1) if gold.intersection(f.source_ids)],
                        'quote_coverage':sum(any(q in f.content for f in found)
                            for q in case['expected_quotes'])/len(case['expected_quotes']),
                        'evidence':evidence(found)})
            baseline=row['arms']['cpu'][0]
            row['non_regression']=all(not r['errors'] and bool(r['gold_ranks'])>=bool(baseline['gold_ranks'])
                and r['quote_coverage']>=baseline['quote_coverage'] for r in row['arms']['gpu'])
            row['identical_evidence']=all(r['evidence']==baseline['evidence'] for r in row['arms']['gpu'])
            row['identical_ranker_inputs']=all(r['ranker']['input_sha256']==baseline['ranker']['input_sha256']
                for r in row['arms']['gpu'])
            row['max_score_delta']=max(abs(s-t) for r in row['arms']['gpu']
                for s,t in zip(r['ranker']['scores'],baseline['ranker']['scores']))
            row['median_ms']={arm:statistics.median(r['total_ms'] for r in runs) for arm,runs in row['arms'].items()}
            report['cases'].append(row)
            save(output/'comparison.json',report)
            print(case['id'],{k:round(v) for k,v in row['median_ms'].items()},
                  'quality',row['non_regression'],'same_evidence',row['identical_evidence'],flush=True)
        report['quality_gate']=all(row['non_regression'] and all(not r['errors'] for r in row['arms']['cpu'])
            for row in report['cases'])
        report['summary']={arm:{'median_ms':statistics.median(r['total_ms'] for c in report['cases'] for r in c['arms'][arm]),
            'p95_ms':statistics.quantiles([r['total_ms'] for c in report['cases'] for r in c['arms'][arm]],n=100,method='inclusive')[94],
            'source_hits':sum(bool(c['arms'][arm][0]['gold_ranks']) for c in report['cases']),
            'case_count':len(cases)} for arm in ('cpu','gpu')}
        report['parent_gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        save(output/'comparison.json',report)
        print('quality_gate',report['quality_gate'],report['summary'],flush=True)
    finally:
        gpu_recent.close()
        pipe.recent_memory=cpu_recent
        pipe.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('audit','cases','out'):
        p.add_argument('--'+arg,required=True)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--paraphrases',action='store_true')
    p.add_argument('--guard-margin',type=float,default=0.0)
    p.add_argument('--cpu-fp32',action='store_true')
    compare(p.parse_args())
