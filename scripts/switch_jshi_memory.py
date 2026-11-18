"""Switch a validated local JShi instance, with final snapshots and rollback."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.request import urlopen

import psutil
import portalocker


def status(port):
    with urlopen(f'http://127.0.0.1:{port}/status',timeout=3) as response:
        return json.load(response)


def unlocked(path):
    lock=path/'.lock'
    if lock.exists():
        with lock.open('r+b') as stream:
            portalocker.lock(stream,portalocker.LOCK_EX|portalocker.LOCK_NB)
            portalocker.unlock(stream)


def launch(executable,arguments,cwd,environment,output,name):
    startup=subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow=0
    with (output/(name+'.stdout.log')).open('ab') as out, (output/(name+'.stderr.log')).open('ab') as err:
        return subprocess.Popen([executable,*arguments],cwd=cwd,env=environment,
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,startupinfo=startup,
            creationflags=subprocess.CREATE_NEW_CONSOLE|subprocess.CREATE_NEW_PROCESS_GROUP)


def stop(pid,allowed):
    subprocess.run([sys.executable,str(Path(__file__).with_name('jshi_console_signal.py')),
        '--pid',str(pid),'--allowed',*[str(x) for x in allowed],'--signal'],check=True)
    try:
        psutil.Process(pid).wait(timeout=90)
    except psutil.NoSuchProcess:
        pass


def wait_ready(process,port,expected=None):
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        if process.poll() is not None:
            raise RuntimeError(f'JShi exited during startup with code {process.returncode}')
        try:
            value=status(port)
            if expected:
                info=value.get('memory') or {}
                if Path(info.get('engine_source') or '').resolve() != expected/'src'/'rems'/'__init__.py':
                    raise ValueError('Unexpected memory engine source')
                if not info.get('recent_process') or not info.get('recent_worker_pid'):
                    raise ValueError('Native Mem0 process is not ready')
            return value
        except (OSError,ValueError):
            time.sleep(1)
    raise TimeoutError('JShi did not become ready within five minutes')


def switch(a):
    root,stage,output=Path(a.jshi_root).resolve(),Path(a.stage).resolve(),Path(a.out).resolve()
    project=Path(__file__).resolve().parents[1]
    if stage.parent!=root/'.jshi' or not stage.name.startswith('rems-dual-'):
        raise ValueError('Unexpected stage directory')
    report=json.loads((stage/'cutover-stage.json').read_text(encoding='utf-8'))
    if not report.get('ready'):
        raise ValueError('Staged JShi adapter acceptance did not pass')
    if output.exists():
        raise ValueError('Use a fresh rollback directory')
    old=psutil.Process(a.pid)
    command=old.cmdline()
    if 'jshi.app.cli' not in command or Path(old.cwd()).resolve()!=root:
        raise ValueError('PID is not the expected JShi instance')
    live=status(a.port)
    if any(live.get(k) for k in ('busy','text_scene_writing','text_jev_writing','cleanup_stage')):
        raise RuntimeError('JShi is still handling a conversation or write; no switch performed')
    unlocked(stage/'qdrant')
    unlocked(stage/'recent_memory'/'qdrant')
    output.mkdir(parents=True)
    environment=old.environ()
    executable,arguments=command[0],command[1:]
    env_path=root/'.env'
    env_original=env_path.read_bytes()
    (output/'original.env').write_bytes(env_original)
    python_file=root/'.jshi'/'python-path.txt'
    python_original=python_file.read_bytes() if python_file.exists() else None
    if python_original is not None:
        (output/'original-python-path.txt').write_bytes(python_original)
    # Private local recovery metadata; never echo environment values.
    (output/'original-launch.json').write_text(json.dumps({'executable':executable,
        'arguments':arguments,'cwd':str(root),'pid':a.pid},indent=2),encoding='utf-8')
    changed=False
    new=None
    try:
        print('Stopping the verified idle JShi console',flush=True)
        stop(a.pid,a.allowed)
        unlocked(root/'.jshi'/'rems'/'qdrant')
        print('Taking final quiescent snapshots and verifying the staged stores',flush=True)
        from stage_jshi_memory import stage as refresh
        refreshed=refresh(SimpleNamespace(jshi_root=str(root),audit=a.audit,out=str(stage),
            snapshot=str(output/'final-snapshot'),cases=a.cases,refresh=True))
        if env_path.read_bytes()!=env_original:
            raise RuntimeError('JShi .env changed during staging; refusing to overwrite it')
        settings={
            'JSHI_MEMORY_BACKEND':'rems3','JSHI_REMS_SOURCE_ROOT':project.as_posix(),
            'JSHI_REMS_DATA_DIR':stage.as_posix(),
            'REMS_RECENT_MEMORY__PATH':(stage/'recent_memory').as_posix(),
            'REMS_RECENT_MEMORY__ENABLED':'true','REMS_RECENT_MEMORY__PROCESS_ISOLATION':'true',
            'REMS_RECENT_MEMORY__RETENTION_DAYS':'365','REMS_RECENT_MEMORY__MAX_BYTES':'10000000000',
            'REMS_RECENT_MEMORY__CLEANUP_TRIGGER_BYTES':'8000000000',
            'REMS_RECENT_MEMORY__CLEANUP_TARGET_BYTES':'7000000000',
            'REMS_MEMORY_REFINEMENT__ENABLED':'false','REMS_EVENT_MIN_CHARS':'2000',
            'REMS_EVENT_LEN_FACTOR':'1.5','REMS_RECALL_EVENT_INDEX_LEVEL':'L1',
            'REMS_RECALL_RERANK_ENABLED':'true'}
        lines=env_original.decode('utf-8-sig').splitlines()
        lines=[line for line in lines if line.strip().partition('=')[0].strip() not in settings]
        new_python=project/'.venv'/'Scripts'/'python.exe'
        if not new_python.is_file():
            raise RuntimeError('Validated Python environment is missing')
        env_path.write_text('\n'.join([*lines,'',*[f'{k}={v}' for k,v in settings.items()]])+'\n',encoding='utf-8')
        changed=True
        python_file.write_text(str(new_python)+'\n',encoding='utf-8')
        new_environment={**environment,**settings}
        print('Starting JShi with the validated independent memory stores',flush=True)
        new=launch(str(new_python),arguments,root,new_environment,output,'dual-jshi')
        live=wait_ready(new,a.port,project)
        memory=live['memory']
        if Path(memory['recent_database']).resolve()!=stage/'recent_memory':
            raise RuntimeError('Recent store path is not the validated stage')
        if memory['long_database']!='sqlite:///'+(stage/'rems.db').as_posix():
            raise RuntimeError('Long-term store path is not the validated stage')
        listeners=[c.pid for c in psutil.net_connections(kind='tcp')
                   if c.status==psutil.CONN_LISTEN and c.laddr.port==a.port]
        final={'switched':True,'old_pid':a.pid,'launcher_pid':new.pid,
            'server_pid':listeners[0] if len(listeners)==1 else None,'memory':memory,
            'old_store_preserved':str(root/'.jshi'/'rems'),
            'rollback_directory':str(output),'validation':refreshed}
        (output/'cutover-result.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'switched':True,'new_pid':new.pid,'memory':memory},ensure_ascii=False),flush=True)
    except BaseException:
        if new is not None and new.poll() is None:
            family=[p.pid for p in psutil.Process(new.pid).children(recursive=True)]
            stop(new.pid,family)
        if changed:
            env_path.write_bytes(env_original)
            if python_original is not None:
                python_file.write_bytes(python_original)
            elif python_file.exists():
                python_file.unlink()  # Only the launcher file this operation created.
        if not psutil.pid_exists(a.pid):
            restored=launch(executable,arguments,root,environment,output,'rollback-jshi')
            wait_ready(restored,a.port)
            print(json.dumps({'rolled_back':True,'pid':restored.pid}),flush=True)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('jshi-root','stage','audit','cases','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--pid',type=int,required=True)
    p.add_argument('--allowed',type=int,nargs='*',default=[])
    p.add_argument('--port',type=int,default=8765)
    switch(p.parse_args())
