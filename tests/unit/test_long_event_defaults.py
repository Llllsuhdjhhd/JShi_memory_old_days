"""Exercise the restored production policy without short-text fixture overrides."""
from datetime import datetime

import pytest

from rems.config import REMSConfig
from rems.pipeline import REMSPipeline
from rems.port import MemoryBatch, MemoryExperience
from tests.conftest import FakeLLM


@pytest.fixture
def long_case(tmp_path, monkeypatch):
    import os
    for name in list(os.environ):
        if name.startswith("REMS_"):
            monkeypatch.delenv(name)
    config = REMSConfig(_env_file=None,
        storage={"database_url": "sqlite:///" + (tmp_path / "events.db").as_posix(),
                 "qdrant_url": ":memory:"},
        embedding={"provider": "hash"},
        recent_memory={"enabled": False},
        recall_query_compress_enabled=False, ingest_llm_session_enabled=False)
    llm = FakeLLM(config)
    pipe = REMSPipeline.from_config(config, llm=llm)
    yield pipe, llm
    pipe.recall_pipeline._vector_store._client.close()
    pipe.db.engine.dispose()


def experience(text, source):
    return MemoryExperience(subject_id="stone", text=text, source_ids=(source,),
        segment_id=source, occurred_at=datetime(2026, 11, 16, 10))


def ingest(pipe, *items):
    result = pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=items))
    assert not result.errors
    return result


def enrichment(llm):
    llm.push_response({"summaries": {"L1": "修车完成，更换轴承后运行平稳。"}, "objects": [],
                       "emotion": {"joy": 0.6}})


def test_short_event_waits_then_seals_as_one_enriched_long_event(long_case):
    pipe, llm = long_case
    first = experience("修车" + "甲" * 1097 + "。", "first")
    second = experience("乙" * 1099 + "。", "second")
    llm.push_response({"events": [{"indices": [1]}], "residual": []})
    initial = ingest(pipe, first)
    assert not initial.sealed_event_ids and initial.unclosed_count == 1
    assert [call["task_type"] for call in llm.calls] == ["boundary_detection"]

    # JShi can resend a partly acknowledged batch. Pending text must not double.
    llm.push_response({"events": [{"indices": [1, 2]}], "residual": []})
    enrichment(llm)
    result = ingest(pipe, first, second)
    assert len(result.sealed_event_ids) == 1 and result.unclosed_count == 0
    event = pipe.event_repo.get(result.sealed_event_ids[0])
    assert event.content_raw == first.text + "\n" + second.text
    assert 2000 <= len(event.content_raw) < 3000
    assert event.memory_stage is None and event.summaries["L1"]
    assert len(event.source_fragment_ids) == 2
    assert set(event.source_ids) == {"first", "second"}
    assert result.stored_marks["first"] == result.stored_marks["second"] == [event.event_id]
    assert "ev_len" in llm.calls[0]["messages"][1]["content"]
    assert "2000" in llm.calls[0]["messages"][1]["content"]

    found = pipe.recall_pipeline.recall("stone", "修车", budget_chars=150,
        channels=("lexical",), reinforce=False)
    assert found and found[0].event_id == event.event_id
    assert found[0].summary_level == "L1" and "更换轴承" in found[0].content

    # Persisted ownership also protects forced replay after consuming the buffer.
    assert pipe.metabolism_service.process_input(first.text, memory=first, force_save=True) == []
    assert len(pipe.event_repo.list_all()) == 1
    calls_before = len(llm.calls)
    assert pipe.refine_memory("stone", now=datetime(2030, 1, 1))["batches"] == []
    assert len(llm.calls) == calls_before


@pytest.mark.parametrize("length, should_seal", [(300, False), (1999, False), (2000, True), (2999, True)])
def test_normal_sealing_boundaries(long_case, length, should_seal):
    pipe, llm = long_case
    text = "甲" * (length - 1) + "。"
    llm.push_response({"events": [{"indices": [1]}], "residual": []})
    if should_seal:
        enrichment(llm)
    result = ingest(pipe, experience(text, "boundary"))
    assert bool(result.sealed_event_ids) is should_seal
    assert result.unclosed_count == (0 if should_seal else 1)


def test_force_save_bounds_long_sentence_without_losing_source(long_case):
    pipe, llm = long_case
    text = "甲" * 6500
    for _ in range(3):
        enrichment(llm)
    memory = experience(text, "overlong")
    events = pipe.metabolism_service.process_input(text, memory=memory, force_save=True)
    assert len(events) == 3
    assert "".join(event.content_raw for event in events) == text
    assert all(len(event.content_raw) < 3000 and event.summaries for event in events)
    assert all(event.seal_reason == "truncated" for event in events)


def test_long_event_time_evidence_and_recall_on_last_source_day(long_case):
    pipe, llm = long_case
    first = experience("修车" + "甲" * 1097 + "。", "first")
    first.occurred_at = datetime(2026, 11, 15, 10)
    second = experience("乙" * 1099 + "。", "second")
    llm.push_response({"events": [{"indices": [1, 2]}], "residual": []})
    enrichment(llm)
    result = ingest(pipe, first, second)
    assert len(result.sealed_event_ids) == 1
    event = pipe.event_repo.get(result.sealed_event_ids[0])
    assert event.event_time_start == first.occurred_at
    assert event.event_time_end == second.occurred_at
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "[1] 来源时间：2026-11-15 10:00" in prompt
    assert "[2] 来源时间：2026-11-16 10:00" in prompt
    assert event.content_raw == first.text + "\n" + second.text
    found = pipe.recall_pipeline.recall("stone", "修车", budget_chars=150,
        time_range=(datetime(2026, 11, 16), datetime(2026, 11, 17)),
        channels=("lexical",), reinforce=False)
    assert found and found[0].event_id == event.event_id
    assert not pipe.recall_pipeline.recall("stone", "修车",
        time_range=(datetime(2026, 11, 17), datetime(2026, 11, 18)),
        channels=("lexical",), reinforce=False)


def test_time_boundary_does_not_force_short_long_mode_events(long_case):
    pipe, llm = long_case
    first = experience("甲" * 299 + "。", "first")
    first.occurred_at = datetime(2026, 11, 15, 10)
    second = experience("乙" * 299 + "。", "second")
    llm.push_response({"events": [{"indices": [1]}, {"indices": [2]}], "residual": []})
    result = ingest(pipe, first, second)
    assert not result.sealed_event_ids and result.unclosed_count == 2
    assert [call["task_type"] for call in llm.calls] == ["boundary_detection"]


def test_unknown_time_is_not_invented_in_long_event_prompt(long_case):
    pipe, llm = long_case
    memory = experience("甲" * 1999 + "。", "unknown")
    memory.occurred_at = None
    llm.push_response({"events": [{"indices": [1]}], "residual": []})
    enrichment(llm)
    result = ingest(pipe, memory)
    assert "[1] 来源时间：时间未知" in llm.calls[0]["messages"][1]["content"]
    event = pipe.event_repo.get(result.sealed_event_ids[0])
    assert event.source_dates == []
    assert event.event_time_start is None and event.event_time_end is None
