"""Recall candidate loading must stay subject-scoped and index-backed."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

from rems.models.event import Event
from rems.recall import HashEmbedding, RecallPipeline
from rems.storage.repository import EventRepository


def _event(subject: str, text: str, at: datetime) -> Event:
    return Event(subject_id=subject, content_raw=text, summaries={"L1": text},
                 create_time=at)


def test_list_by_subject_matches_the_old_python_filter(db):
    events = EventRepository(db)
    events.save(_event("s2", "other", datetime(2026, 1, 1)))
    events.save(_event("s1", "second", datetime(2026, 1, 3)))
    events.save(_event("s1", "first", datetime(2026, 1, 2)))
    removed = _event("s1", "tombstoned", datetime(2026, 1, 4))
    removed.is_tombstoned = True
    events.save(removed)

    expected = [
        e.event_id for e in events.list_all(exclude_tombstoned=True)
        if (e.subject_id or "") == "s1" and not e.is_tombstoned
    ]
    assert [e.event_id for e in events.list_by_subject("s1")] == expected


def test_recall_indexes_exist_on_an_existing_database(db):
    with db.engine.connect() as conn:
        event_indexes = {row[1] for row in conn.exec_driver_sql("PRAGMA index_list('events')")}
        white_indexes = {
            row[1] for row in conn.exec_driver_sql("PRAGMA index_list('white_painting_entries')")
        }
        object_indexes = {
            row[1] for row in conn.exec_driver_sql("PRAGMA index_list('object_memory_entries')")
        }
    assert "ix_events_subject_tombstone_time" in event_indexes
    assert "ix_white_painting_entries_event_id" in white_indexes
    assert "ix_object_memory_entries_lookup" in object_indexes


def test_candidate_events_uses_the_subject_query(db):
    events = EventRepository(db)
    events.save(_event("s1", "kept", datetime(2026, 1, 1)))
    events.save(_event("s2", "other subject", datetime(2026, 1, 2)))
    events.list_all = Mock(side_effect=AssertionError("subject query expected"))

    pipe = RecallPipeline(HashEmbedding(), None, events)
    assert [e.content_raw for e in pipe._candidate_events("s1")] == ["kept"]
