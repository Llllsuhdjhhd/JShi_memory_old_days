"""Canonical working prose plus originals; SQL identity columns are never rewritten."""
from datetime import datetime, timezone
import hashlib,json
from sqlalchemy import Table,Column,Text,event,inspect,select
from .simplified_text import simplify_data

# Explicit owned columns, not a recursive conversion of arbitrary DB payloads.
PROSE_COLUMNS={
 'events':('content_raw','summaries','role_list','decoration','insight','location','recall_metadata'),
 'roles':('name','aliases'), 'shadow':('content','buffer_items'),
 'unclosed_events':('content_fragments','logical_gaps','buffer_items'),
 'white_painting_entries':('role_summary','l1_mention','l2_interaction','l3_decision'),
 'object_memory_entries':('name','summary'), 'object_portraits':('name','levels'),
 'portrait_summaries':('text',), 'object_traits':('body',), 'object_states':('body',),
 'object_dispositions':('body',), 'person_experience_portraits':('content','brief','shape'),
 'person_experience_evidence':('text',), 'person_experience_revisions':('reason','revised_claims','snapshot'),
}

def install(base):
 originals=Table('simplified_originals',base.metadata,
    Column('id',Text,primary_key=True),Column('table_name',Text),Column('record_key',Text),
    Column('field_name',Text),Column('original_json',Text),Column('created_at',Text))
 def normalize(mapper,connection,target):
    state=inspect(target);table=target.__table__.name
    identity=[getattr(target,c.key,None) for c in mapper.primary_key]
    if any(value is None for value in identity):
        identity.extend(getattr(target,k,None) for k in ('subject_id','event_id','object_id','role_id'))
    key=json.dumps(identity,ensure_ascii=False,default=str)
    for name in PROSE_COLUMNS.get(table,()):
        if name not in state.attrs or not state.attrs[name].history.has_changes():continue
        old=getattr(target,name);new=simplify_data(old,prose=True)
        if old==new:continue
        encoded=json.dumps(old,ensure_ascii=False,default=str)
        digest=hashlib.sha256((table+'\0'+key+'\0'+name+'\0'+encoded).encode()).hexdigest()
        if not connection.execute(select(originals.c.id).where(originals.c.id==digest)).first():
            connection.execute(originals.insert().values(id=digest,table_name=table,record_key=key,
                field_name=name,original_json=encoded,created_at=datetime.now(timezone.utc).isoformat()))
        setattr(target,name,new)
 event.listen(base,'before_insert',normalize,propagate=True)
 event.listen(base,'before_update',normalize,propagate=True)

def canonicalize_event(event):
 """Index the same canonical prose as EventRepository persists."""
 for name in ('content_raw','summaries','decoration','insight','location'):
    old=getattr(event,name,None)
    if old is not None:setattr(event,name,simplify_data(old,prose=True))
 return event
