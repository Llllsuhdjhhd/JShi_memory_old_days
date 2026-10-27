"""Latency-path contracts: one legacy timeline read, no giant MatchAny by default."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

from rems.models.event import Event
from rems.models.object_entry import ObjectMemoryEntry
from rems.recall import HashEmbedding, RecallPipeline
from rems.storage.repository import EventRepository, ObjectTimelineRepository


class RecordingStore:
    """Vector store double that records filters and can force a truncated result."""

    def __init__(self, hits_per_search: int):
        self.hits_per_search = hits_per_search
        self.calls: list[dict] = []

    def search(self, vector, *, top_k, payload_filter=None, any_of=None, ranges=None):
        self.calls.append({
            "top_k": top_k,
            "payload_filter": payload_filter,
            "any_of": any_of,
            "ranges": ranges,
        })
        count = min(self.hits_per_search, top_k)
        return [
            {
                "event_id": f"e{i}",
                "score": 1.0,
                "payload": {"event_id": f"e{i}", "type": "event"},
            }
            for i in range(count)
        ]


def _pipeline(store) -> RecallPipeline:
    return RecallPipeline(HashEmbedding(), store, Mock(), None)


def test_event_id_filter_skipped_when_search_is_not_truncated():
    store = RecordingStore(hits_per_search=2)
    pipe = _pipeline(store)
    pipe._semantic_signals(
        "s", "q", (), None,
        query_vector=[0.0] * 64, event_ids=("e1", "e2"), top_k=10,
    )
    assert len(store.calls) == 1
    assert store.calls[0]["any_of"] is None


def test_event_id_filter_restored_when_result_is_truncated():
    store = RecordingStore(hits_per_search=10)
    pipe = _pipeline(store)
    pipe._semantic_signals(
        "s", "q", (), None,
        query_vector=[0.0] * 64, event_ids=("e1", "e2"), top_k=10,
    )
    assert len(store.calls) == 2
    assert store.calls[1]["any_of"] == {"event_id": ["e1", "e2"]}


def test_object_filter_survives_without_event_id_filter():
    store = RecordingStore(hits_per_search=2)
    pipe = _pipeline(store)
    pipe._semantic_signals(
        "s", "q", ("obj-1",), None,
        query_vector=[0.0] * 64, event_ids=("e1",), top_k=10,
    )
    assert store.calls[0]["any_of"] == {"object_ids": ["obj-1"]}


def test_preloaded_timeline_route_needs_no_repository_read():
    timeline = Mock()
    pipe = RecallPipeline(HashEmbedding(), None, Mock(), timeline)
    ev = Event(event_id="e1", subject_id="s", content_raw="x")
    assert pipe._matches_objects("s", ev, ("p",), route_event_ids=frozenset({"e1"})) is True
    assert pipe._matches_objects("s", ev, ("p",), route_event_ids=frozenset()) is False
    timeline.list_by_object.assert_not_called()


def test_recall_reads_legacy_timeline_once_for_many_events(db):
    events, timeline = EventRepository(db), ObjectTimelineRepository(db)
    for text in ("bicycle repair", "weather", "dinner"):
        events.save(Event(subject_id="s", content_raw=text, summaries={"L1": text}))
    first = events.list_all()[0]
    timeline.append(ObjectMemoryEntry(
        subject_id="s", object_id="p", event_id=first.event_id,
        summary="bicycle repair", create_time=datetime.now(),
    ))
    pipe = RecallPipeline(HashEmbedding(), None, events, timeline)
    timeline.list_by_object = Mock(wraps=timeline.list_by_object)

    found = pipe.recall(
        "s", "bicycle", object_ids=("p",), channels=("lexical",), reinforce=False,
    )

    assert timeline.list_by_object.call_count == 1
    assert [item.event_id for item in found] == [first.event_id]


def test_longest_common_substring_is_exact():
    import random
    from rems.recall.pipeline import _longest_common_substring

    def brute(a: str, b: str) -> int:
        best = 0
        for i in range(len(a)):
            for j in range(len(b)):
                k = 0
                while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                    k += 1
                best = max(best, k)
        # lexical_score has always ignored single-character overlap.
        return best if best >= 2 else 0

    rng = random.Random(20261027)
    for _ in range(200):
        a = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 12)))
        b = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 12)))
        assert _longest_common_substring(a, b) == brute(a, b), (a, b)
