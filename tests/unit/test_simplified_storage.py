from pathlib import Path
import sys
from rems.storage.database import Database
from rems.storage.repository import EventRepository
from rems.models.event import Event
from rems.utils.zh_normalize import normalize_for_substring_match

def test_working_store_and_originals(tmp_path):
    db=Database('sqlite:///'+(tmp_path/'memory.db').as_posix());db.create_tables()
    repo=EventRepository(db)
    e=Event(event_id='事件-聲紋',subject_id='s',content_raw='聲紋已確認',summaries={'L1':'請繼續'})
    repo.save(e)
    got=repo.get(e.event_id)
    assert got.content_raw=='声纹已确认'
    assert got.summaries=={'L1':'请继续'}
    assert got.event_id=='事件-聲紋'
    from sqlalchemy import text
    with db.engine.connect() as c:
        assert c.execute(text('select count(*) from simplified_originals')).scalar()==2
    assert normalize_for_substring_match('聲紋ABC')=='聲紋abc'



def test_model_result_converts_prose_and_archives_original(tmp_path):
    import json
    from types import SimpleNamespace
    from rems.llm.provider import LLMProvider
    provider=LLMProvider.__new__(LLMProvider)
    provider.config=SimpleNamespace(storage=SimpleNamespace(database_url='sqlite:///'+(tmp_path/'memory.db').as_posix()))
    raw='{"text":"請繼續","event_id":"事件-聲紋"}'
    assert provider._canonical_json(raw)=={'text':'请继续','event_id':'事件-聲紋'}
    saved=json.loads((tmp_path/'simplified-model-originals.jsonl').read_text(encoding='utf-8'))
    assert saved['raw']==raw
