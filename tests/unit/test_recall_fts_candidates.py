"""Derived lexical index and candidate loading contracts."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

from rems.models.event import Event, EventRoleEntry
from rems.recall import HashEmbedding, RecallPipeline
from rems.storage.database import EventRecord
from rems.storage.repository import EventRepository


def _save(events, subject, text, summary=None, at=None):
    ev = Event(
        subject_id=subject,
        content_raw=text,
        summaries={"L1": summary or text},
        create_time=at or datetime(2026, 1, 1),
    )
    events.save(ev)
    return ev


def test_fts_tracks_insert_update_and_delete(db):
    events = EventRepository(db)
    ev = _save(events, "s", "今天下午聊到旧收音机", "旧收音机声音有点闷")
    assert events.search_fts("s", "旧收音机", 10) == [ev.event_id]
    assert events.search_fts("s", "扬声器", 10) == []

    ev.summaries = {"L1": "扬声器进灰了"}
    events.save(ev)
    assert events.search_fts("s", "扬声器", 10) == [ev.event_id]
    assert events.search_fts("s", "旧收音机", 10) == [ev.event_id]  # raw text remains

    ev.content_raw = "只留扬声器"
    ev.summaries = {"L1": "只留扬声器"}
    events.save(ev)
    assert events.search_fts("s", "旧收音机", 10) == []

    with db.session() as session:
        session.delete(session.get(EventRecord, ev.event_id))
        session.commit()
    assert events.search_fts("s", "扬声器", 10) == []


def test_two_character_query_uses_like_fallback(db):
    events = EventRepository(db)
    ev = _save(events, "s", "他说今天下午来", "下午来")
    assert events.search_fts("s", "下午", 10) == [ev.event_id]
    assert events.search_fts("s", "晚上", 10) == []


def test_existing_rows_are_backfilled_once(db):
    events = EventRepository(db)
    first = _save(events, "s", "旧收音机在客厅", at=datetime(2026, 1, 1))
    second = _save(events, "s", "扬声器进灰了", at=datetime(2026, 1, 2))
    with db.engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM events_fts")

    db.create_tables()
    assert events.search_fts("s", "旧收音机", 10) == [first.event_id]
    assert events.search_fts("s", "扬声器", 10) == [second.event_id]


def test_candidate_loaders_are_subject_and_id_scoped(db):
    events = EventRepository(db)
    first = _save(events, "s", "first", at=datetime(2026, 1, 1))
    second = _save(events, "s", "second", at=datetime(2026, 1, 2))
    _save(events, "other", "other", at=datetime(2026, 1, 3))

    assert events.count_by_subject("s") == 2
    assert events.recent_ids_by_subject("s", 1) == [second.event_id]
    loaded = events.list_by_ids([second.event_id, first.event_id], subject_id="s")
    assert [e.event_id for e in loaded] == [first.event_id, second.event_id]
    assert events.list_by_ids([first.event_id], subject_id="other") == []


def test_object_lane_finds_role_linked_events(db):
    events = EventRepository(db)
    linked = _save(events, "s", "和这件事无关的文字", at=datetime(2026, 1, 5))
    linked.role_list = [EventRoleEntry(role_id="p")]
    events.save(linked)
    _save(events, "s", "别的对象", at=datetime(2026, 1, 6))

    assert events.ids_for_object("s", "p", 10) == [linked.event_id]
    assert events.ids_for_object("s", "absent", 10) == []


def test_cascade_loads_only_the_candidate_union(db):
    events = EventRepository(db)
    for index in range(6):
        _save(events, "s", f"普通记录 {index}", at=datetime(2026, 1, 1 + index))
    target = _save(events, "s", "周末去看旧收音机", at=datetime(2026, 1, 8))

    pipe = RecallPipeline(
        HashEmbedding(), None, events,
        candidate_mode="cascade", cascade_min_events=3,
        candidate_limit=20, candidate_lexical_k=5,
        candidate_recent_k=1, candidate_semantic_backfill=0,
    )
    events.list_all = Mock(side_effect=AssertionError("cascade must not full scan"))

    found = pipe.recall("s", "旧收音机", channels=("lexical",), reinforce=False)
    assert [item.event_id for item in found] == [target.event_id]


def test_cascade_falls_back_when_the_union_is_too_small(db):
    events = EventRepository(db)
    target = _save(events, "s", "周末去看旧收音机", at=datetime(2026, 1, 8))
    for index in range(5):
        _save(events, "s", f"普通记录 {index}", at=datetime(2026, 1, 1 + index))

    pipe = RecallPipeline(
        HashEmbedding(), None, events,
        candidate_mode="cascade", cascade_min_events=3,
        candidate_limit=20, candidate_lexical_k=2,
        candidate_recent_k=1, candidate_min_events=50,
    )
    found = pipe.recall("s", "旧收音机", channels=("lexical",), reinforce=False)
    assert target.event_id in [item.event_id for item in found]


def test_cascade_excludes_hidden_and_tombstoned_matches(db):
    events = EventRepository(db)
    for index in range(6):
        _save(events, "s", f"普通记录 {index}", at=datetime(2026, 1, 1 + index))
    hidden = _save(events, "s", "旧收音机被吸收", at=datetime(2026, 1, 8))
    hidden.refinement_hidden_by = ["parent"]
    events.save(hidden)
    removed = _save(events, "s", "旧收音机已删除", at=datetime(2026, 1, 9))
    removed.is_tombstoned = True
    events.save(removed)

    pipe = RecallPipeline(
        HashEmbedding(), None, events,
        candidate_mode="cascade", cascade_min_events=3,
        candidate_limit=30, candidate_lexical_k=10,
    )
    found = pipe.recall("s", "旧收音机", channels=("lexical",), reinforce=False)
    ids = [item.event_id for item in found]
    assert hidden.event_id not in ids
    assert removed.event_id not in ids


