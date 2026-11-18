"""Stage/refresh independent dual stores from actual JShi data, without switching it.

Only SQLite online backups read live stores. Live Qdrant is never opened.
After stopping JShi, run --refresh with a fresh --snapshot for the final delta.
"""
import argparse
import hashlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

from audit_jshi_cutover import audit, save
from rems.config import REMSConfig
from rems.pipeline import REMSPipeline
from rems.port import MemoryBatch, MemoryExperience


class NoLLM:
    def complete(self, *args, **kwargs):
        raise AssertionError("Staging must not generate or rewrite historical memories")
    complete_json = complete


def event_digest(path):
    digest=hashlib.sha256()
    with sqlite3.connect(Path(path).as_uri()+'?mode=ro',uri=True) as db:
        for row in db.execute('SELECT * FROM events ORDER BY event_id'):
            digest.update(repr(row).encode('utf-8'))
    return digest.hexdigest()


def stage(a):
    jshi, source, output, snapshot = map(lambda p:Path(p).resolve(),
        (a.jshi_root,a.audit,a.out,a.snapshot))
    if output.parent != jshi/'.jshi' or not output.name.startswith('rems-dual-'):
        raise ValueError("Stage must be a new .jshi/rems-dual-* directory")
    if not (source/'rehearsal.json').is_file():
        raise ValueError("Requires the closed, completed isolated rehearsal")
    if output.exists() and not (a.refresh and (output/'cutover-stage.json').is_file()):
        raise ValueError("Refusing to replace an unrecognized stage")
    audit(jshi,snapshot)
    same_events = a.refresh and event_digest(snapshot/'long.db') == event_digest(output/'rems.db')
    output.mkdir(parents=True,exist_ok=True)
    if not a.refresh:
        shutil.copytree(source/'long-vectors',output/'qdrant')
        shutil.copytree(source/'recent',output/'recent_memory')
    shutil.copy2(snapshot/'long.db',output/'rems.db')
    for path in (jshi/'.jshi'/'rems').glob('jshi_person_views*'):
        if path.is_dir():
            shutil.copytree(path,output/path.name,dirs_exist_ok=True)
        elif path.is_file():
            shutil.copy2(path,output/path.name)
    # Parse real JShi settings, but never publish secrets in the report.
    sys.path.insert(0,str(jshi/'src'))
    from jshi.app.cli import parse_env_text
    for key,value in parse_env_text((jshi/'.env').read_text(encoding='utf-8-sig')).items():
        os.environ.setdefault(key,value)
    config=REMSConfig(_env_file=str(jshi/'.env'))
    config.storage.database_url='sqlite:///'+(output/'rems.db').as_posix()
    config.storage.qdrant_path=str(output/'qdrant')
    config.recent_memory.path=str(output/'recent_memory')
    config.recent_memory.enabled=True
    config.recent_memory.process_isolation=False  # Only the closed-stage import/inspection.
    config.memory_refinement.enabled=False
    config.event_min_chars,config.event_len_factor=2000,1.5
    config.recall_event_index_level='L1'
    config.recall_reinforce_enabled=False
    config.recall_trace_enabled=False
    config.recall_query_compress_enabled=False
    items=[MemoryExperience.model_validate(x) for x in json.loads((snapshot/'recent-replay.json').read_text(encoding='utf-8'))]
    report={'snapshot':str(snapshot),'stage':str(output),'llm_calls':0,
        'effective_context_window':config.context_window,'event_min_chars':config.event_min_chars,
        'event_len_factor':config.event_len_factor,'recent_process_isolation':True}
    pipe=REMSPipeline.from_config(config,llm=NoLLM())
    try:
        start=time.perf_counter()
        if same_events:
            scanned,written=pipe.event_repo.count_by_subject('stone'),0
        else:
            scanned,written=pipe.recall_pipeline.backfill_index('stone')
        report['long_index']={'scanned':scanned,'written':written,'seconds':time.perf_counter()-start}
        rows=list(pipe.recent_memory._records())
        existing={row.payload.get('rems_segment'):row.payload.get('data') for row in rows}
        changed=[x.segment_id for x in items if x.segment_id in existing and existing[x.segment_id]!=x.text]
        if changed:
            raise ValueError(f"Historical segments changed content: {len(changed)}; explicit correction required")
        missing=[x for x in items if x.segment_id not in existing]
        for offset in range(0,len(missing),100):
            pipe.recent_memory.ingest_batch(MemoryBatch(subject_id='stone',experiences=tuple(missing[offset:offset+100])))
        actual={row.payload.get('rems_segment') for row in pipe.recent_memory._records()}
        assert all(x.segment_id in actual for x in items)
        report['recent']={'expected':len(items),'records':len(actual),'imported_delta':len(missing),
                          'bytes':pipe.recent_memory.disk_bytes()}
        print({'long':report['long_index'],'recent':report['recent']},flush=True)
    finally:
        pipe.close()
    config.recent_memory.process_isolation=True
    from jshi.memory.rems3 import Rems3MemoryBackend
    import rems.port
    pipe=REMSPipeline.from_config(config,llm=NoLLM())
    backend=Rems3MemoryBackend(pipeline=pipe,data_dir=output,
        engine_types=(rems.port.MemoryBatch,rems.port.MemoryExperience))
    try:
        backend.prepare_local_recall()
        report['recent_worker_pid']=pipe.recent_memory._process.pid
        # Retry real historical records through the process boundary. No new
        # content, no long-store replay and no change to JShi's consumption cursor.
        sample=tuple(items[::max(1,len(items)//25)][:25])
        batch=MemoryBatch(subject_id='stone',experiences=sample)
        pipe.recent_memory.ingest_batch(batch)
        pipe.recent_memory.ingest_batch(batch)
        report['queries']=[]
        for case in json.loads(Path(a.cases).read_text(encoding='utf-8')):
            begin=time.perf_counter()
            found=backend.recall('stone',case['query'],limit=6)
            row={'id':case['id'],'ms':(time.perf_counter()-begin)*1000,
                'source_hit':any(set(case['expected_sources']).intersection(f.source_ids) for f in found),
                'quote_coverage':sum(any(q in f.content for f in found) for q in case['expected_quotes'])/len(case['expected_quotes']),
                'errors':list(pipe.last_recall_errors)}
            report['queries'].append(row)
            print(row,flush=True)
        report['ready']=all(r['source_hit'] and r['quote_coverage']==1 and not r['errors'] for r in report['queries'])
        save(output/'cutover-stage.json',report)
        save(snapshot/'stage-result.json',report)
        if not report['ready']:
            raise AssertionError("Stage failed actual JShi adapter acceptance")
    finally:
        backend.close()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('jshi-root','audit','out','snapshot','cases'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--refresh',action='store_true')
    stage(p.parse_args())
