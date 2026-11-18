"""Compare padding-only batching changes on captured real reranker inputs."""
import argparse
import json
from pathlib import Path
import statistics
import time

from diagnose_dual_recall import save
from rems.config import REMSConfig
from rems.recall.providers import BgeReranker


def main(a):
    config=REMSConfig(_env_file=None)
    ranker=BgeReranker(config.recall_rerank_model,onnx_file=config.recall_rerank_onnx_file)
    ranker._ensure()
    if a.threads:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        options=ort.SessionOptions()
        options.intra_op_num_threads=a.threads
        options.inter_op_num_threads=1
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        options.add_session_config_entry('session.inter_op.allow_spinning','0')
        ranker._model=ort.InferenceSession(hf_hub_download(config.recall_rerank_model,
            config.recall_rerank_onnx_file,local_files_only=True),sess_options=options,
            providers=['CPUExecutionProvider'])
    cases={row['query']:row for row in json.loads(Path(a.inputs).read_text(encoding='utf-8'))}
    results=[]
    for query,row in cases.items():
        encoded=ranker._tokenizer([query]*len(row['texts']),row['texts'],padding=True,
            truncation=True,max_length=512,return_tensors='np')
        names={x.name for x in ranker._model.get_inputs()}
        lengths=encoded['attention_mask'].sum(axis=1).tolist()
        result={'query':query,'lengths':lengths,'arms':{}}
        for size in [int(x) for x in a.sizes.split(',')]:
            indexes=sorted(range(len(lengths)),key=lambda i:lengths[i]) if size<6 else list(range(len(lengths)))
            scores=[None]*len(lengths)
            times=[]
            for repeat in range(4):
                start=time.perf_counter()
                for begin in range(0,len(indexes),size):
                    batch=indexes[begin:begin+size]
                    width=max(lengths[i] for i in batch)
                    values=ranker._model.run(None,{k:v[batch,:width] for k,v in encoded.items() if k in names})[0]
                    for i,value in zip(batch,values):
                        scores[i]=float(value[0])
                if repeat:
                    times.append((time.perf_counter()-start)*1000)
            result['arms'][str(size)]={'median_ms':statistics.median(times),'ms':times,
                'scores':scores,'order':sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)}
        results.append(result)
        save(Path(a.out),results)
        print(query,lengths,{k:(round(v['median_ms']),v['order']) for k,v in result['arms'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--threads',type=int)
    p.add_argument('--sizes',default='6,1,2,3')
    main(p.parse_args())
