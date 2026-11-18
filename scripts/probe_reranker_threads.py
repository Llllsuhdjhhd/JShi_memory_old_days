"""Measure native Mem0 reranker CPU cost on captured real candidates."""
import json
import time
import torch
import math
import re
from collections import Counter
from pathlib import Path
from mem0.reranker.sentence_transformer_reranker import SentenceTransformerReranker

if __name__ == "__main__":
    ranker = SentenceTransformerReranker({
        "model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "device": "cpu",
        "batch_size": 16, "show_progress_bar": False})
    rows=json.loads(Path("runs/recall-fix-20261117/rerank-probe/ranking.json").read_text(encoding="utf-8"))
    cases=json.loads(Path("runs/dual-actual-20261116/cases.json").read_text(encoding="utf-8"))
    for n in (8,):
        torch.set_num_threads(n)
        for row,case in zip(rows,cases):
            if row["id"] not in ("interface","voice"):
                continue
            def terms(t):
                t=re.sub(r'\W','',t.casefold())
                return Counter(t[i:i+k] for k in (2,3) for i in range(len(t)-k+1))
            docs=row['candidates']; ts=[terms(d['memory']) for d in docs]
            df=Counter(w for t in ts for w in t); avg=sum(sum(t.values()) for t in ts)/len(ts); q=terms(case['query'])
            scores=[sum(math.log(1+(len(ts)-df[w]+.5)/(df[w]+.5))*t[w]*2.2/(t[w]+1.2*(.25+.75*sum(t.values())/avg)) for w in q if t[w]) for t in ts]
            lexical=sorted(range(len(docs)),key=lambda i:scores[i],reverse=True)
            for size in (6,12):
                ids=list(dict.fromkeys([i for pair in zip(range(size//2),lexical[:size//2]) for i in pair]))
                short=[docs[i] for i in ids]
                start=time.perf_counter()
                ranked=ranker.rerank(case["query"],short)
                print(n,size,row["id"],len(short),round((time.perf_counter()-start)*1000),
                    [i for i,d in enumerate(ranked,1) if set(case["expected_sources"]).intersection(d["sources"])],flush=True)
