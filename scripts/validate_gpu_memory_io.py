"""GPU-native Mem0 write/retry, then CPU reopen, using real replay records only."""
import argparse
import json
from pathlib import Path

from diagnose_dual_recall import config_for, save
from rems.port import MemoryBatch, MemoryExperience
from rems.services.recent_process import Mem0RecentProcess
from rems.services.recent_memory import Mem0RecentMemory


def main(a):
    output=Path(a.out).resolve()
    output.mkdir(parents=True,exist_ok=False)
    config=config_for(a.audit)
    config.storage.database_url='sqlite:///'+(output/'unused-long.db').as_posix()
    config.storage.qdrant_path=str(output/'unused-long-vectors')
    config.recent_memory.path=str(output/'recent')
    config.embedding.device='cuda'
    rows=json.loads((Path(a.audit)/'recent-replay.json').read_text(encoding='utf-8'))
    items=tuple(MemoryExperience.model_validate(row) for row in rows[::max(1,len(rows)//25)][:25])
    batch=MemoryBatch(subject_id='stone',experiences=items)
    worker=Mem0RecentProcess(config)
    try:
        runtime=worker.prepare()
        assert runtime['embedding_device'].startswith('cuda')
        worker.ingest_batch(batch)
        worker.ingest_batch(batch)
        found=worker.recall('stone',items[0].text)
        gpu_sources={source for f in found for source in f.source_ids}
        assert set(items[0].source_ids).issubset(gpu_sources)
    finally:
        worker.close()
    config.embedding.device='cpu'
    store=Mem0RecentMemory(config,background=False)
    try:
        runtime_cpu=store.prepare()
        records=list(store._records())
        by_segment={r.payload['rems_segment']:r.payload for r in records}
        assert len(records)==len(items)==25
        for item in items:
            payload=by_segment[item.segment_id]
            assert payload['data']==item.text
            assert payload['rems_sources']==list(item.source_ids)
        recovered=store.recall('stone',items[0].text)
        assert any(f.text==items[0].text for f in recovered)
        result={'passed':True,'real_records':len(items),'records_after_retry':len(records),
                'gpu_runtime':runtime,'cpu_runtime':runtime_cpu,'raw_and_sources_preserved':True}
        save(output/'result.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',required=True)
    p.add_argument('--out',required=True)
    main(p.parse_args())
