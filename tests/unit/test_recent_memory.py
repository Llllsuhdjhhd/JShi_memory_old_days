import os
os.environ["MEM0_TELEMETRY"] = "false"

from datetime import datetime, timezone
import threading

import pytest

from rems.port import MemoryBatch, MemoryExperience, RecalledFragment
from rems.recall.fusion import merge_memories
from rems.recall.providers import HashEmbedding
from rems.services.recent_memory import Mem0RecentMemory, RecentCapacityError


def test_dual_memory_production_defaults(monkeypatch):
    from rems.config import REMSConfig
    for name in list(os.environ):
        if name.startswith("REMS_"):
            monkeypatch.delenv(name)
    config = REMSConfig(_env_file=None)
    assert config.recent_memory.enabled and config.recall_event_index_level == "L1"
    assert config.recent_memory.retention_days == 365
    assert (config.recent_memory.cleanup_target_bytes, config.recent_memory.cleanup_trigger_bytes,
            config.recent_memory.max_bytes) == (7_000_000_000, 8_000_000_000, 10_000_000_000)
    assert not config.memory_refinement.enabled and config.event_min_chars == 2000


def test_retention_root_cannot_overlap_long_term_files(config, tmp_path):
    config.storage.database_url = 'sqlite:///'+(tmp_path/'long'/'rems.db').as_posix()
    config.storage.qdrant_path = str(tmp_path/'long'/'vectors')
    for root in (tmp_path/'long', tmp_path/'long'/'vectors', tmp_path/'long'/'vectors'/'recent'):
        config.recent_memory.path = str(root)
        with pytest.raises(ValueError):
            Mem0RecentMemory(config)


@pytest.fixture
def native_recent(config, tmp_path, monkeypatch):
    # Only the embedding model is replaced. Real Mem0 add/search/delete,
    # Qdrant, SQLite history and disk reclamation all execute offline.
    from mem0.utils.factory import EmbedderFactory
    class OfflineEmbedding:
        def embed(self, text, memory_action=None):
            return HashEmbedding().embed_query(text)
        def embed_batch(self, texts, memory_action=None):
            return [self.embed(text) for text in texts]
    monkeypatch.setattr(EmbedderFactory, "create", lambda *a, **k: OfflineEmbedding())
    config.recent_memory.path = str(tmp_path / "recent")
    config.recent_memory.embedding_dims = 64
    config.recent_memory.search_threshold = 0
    now = datetime(2026, 11, 16, tzinfo=timezone.utc).timestamp()
    recent = Mem0RecentMemory(config, clock=lambda: now, background=False)
    yield recent
    recent.close()


def item(text="维修费是358元。", *, source="one", subject="stone", date=None, who="lee"):
    return MemoryExperience(subject_id=subject, text=text, source_ids=(source,),
        segment_id=source, interlocutor=who, objects={"Lee": who} if who else {},
        occurred_at=date or datetime(2026, 11, 15, tzinfo=timezone.utc))


def write(recent, *items):
    recent.ingest_batch(MemoryBatch(subject_id=items[0].subject_id, experiences=items))


def test_native_mem0_raw_retry_and_reopen(native_recent):
    recent = native_recent
    write(recent, item())
    write(recent, item())
    assert len(list(recent._records())) == 1
    found = recent.recall("stone", "维修费")
    assert len(found) == 1 and found[0].text == "维修费是358元。"
    assert found[0].source_ids == ["one"] and found[0].interlocutor == "lee"
    config, clock = recent.config, recent._clock
    recent.close()
    reopened = Mem0RecentMemory(config, clock=clock, background=False)
    try:
        write(reopened, item())
        assert len(list(reopened._records())) == 1
        assert reopened.recall("stone", "358")[0].text == "维修费是358元。"
    finally:
        reopened.close()


def test_device_switch_preserves_native_embedding_identity_and_records(native_recent, monkeypatch):
    from mem0.utils.factory import EmbedderFactory
    from unittest.mock import Mock
    recent=native_recent
    write(recent,item())
    marker=(recent.root/'rems-mem0.json').read_bytes()
    config,clock=recent.config,recent._clock
    recent.close()
    config.embedding.device='cuda'
    factory=Mock(wraps=EmbedderFactory.create)
    monkeypatch.setattr(EmbedderFactory,'create',factory)
    reopened=Mem0RecentMemory(config,clock=clock,background=False)
    try:
        assert reopened.recall('stone','358')[0].text=='维修费是358元。'
        assert (reopened.root/'rems-mem0.json').read_bytes()==marker
        assert factory.call_args.args[1]['model_kwargs']['device']=='cuda'
        write(reopened,item())
        assert len(list(reopened._records()))==1
    finally:
        reopened.close()


def test_native_mem0_filters_subject_object_and_time(native_recent):
    recent = native_recent
    write(recent, item())
    write(recent, item(subject="other", source="other"))
    assert recent.recall("stone", "维修", object_id="absent") == ()
    assert len(recent.recall("stone", "维修", object_id="lee")) == 1
    assert recent.recall("stone", "维修", time_range=(datetime(2026, 1, 1), datetime(2026, 1, 2))) == ()
    assert recent.recall("missing", "维修") == ()


def test_optional_reranking_uses_native_mem0_factory(native_recent, monkeypatch):
    from mem0.utils.factory import RerankerFactory
    from unittest.mock import Mock
    ranker = Mock()
    ranker.rerank.side_effect = lambda query, rows, limit: list(reversed(rows))
    factory = Mock(return_value=ranker)
    monkeypatch.setattr(RerankerFactory, "create", factory)
    native_recent.policy.reranker = {"provider": "sentence_transformer",
        "config": {"model": "local-test"}}
    native_recent.policy.search_limit = 1
    native_recent.policy.rerank_candidate_limit = 10
    write(native_recent, item(), item("明天修理自行车", source="two"))
    result = native_recent.recall("stone", "维修")
    factory.assert_called_once()
    ranker.rerank.assert_called_once()
    assert len(ranker.rerank.call_args.args[1]) == 2
    assert len(result) == 1


def test_native_origin_is_preserved_in_delivered_evidence(native_recent):
    write(native_recent, item().model_copy(update={"origin": "internal"}))
    found = native_recent.recall("stone", "维修")[0]
    assert "内部记录" in found.content and found.signals["source_origin"] == 1
    assert next(native_recent._records()).payload["rems_origin"] == "internal"


def test_replay_then_live_timezone_conversion_does_not_duplicate(native_recent):
    original = item()
    write(native_recent, original)
    # JShi's real adapter normalizes UTC records into local naive wall clock.
    live = original.model_copy(update={"occurred_at": original.occurred_at.astimezone().replace(tzinfo=None)})
    write(native_recent, live)
    assert len(list(native_recent._records())) == 1
    with pytest.raises(ValueError, match="different raw text"):
        write(native_recent, live.model_copy(update={"text": "被意外改写的原文"}))


def test_native_batch_retry_looks_up_segments_together(native_recent, monkeypatch):
    items = [item(source=f"batch-{i}") for i in range(8)]
    write(native_recent, *items)
    memory = native_recent._get_memory()
    original, calls = memory.get_all, []
    def observed(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(memory, "get_all", observed)
    write(native_recent, *items)
    assert len(list(native_recent._records())) == 8 and len(calls) == 1
    assert set(calls[0]["filters"]["rems_segment"]["in"]) == {x.segment_id for x in items}


def test_native_expiry_excluded_before_maintenance_and_physically_deleted(native_recent):
    recent = native_recent
    write(recent, item())
    old = recent._clock()
    recent._clock = lambda: old + 366 * 86400
    assert recent.recall("stone", "维修") == ()
    report = recent.maintain(force=True)
    assert report["deleted"] == 1
    assert list(recent._records()) == []
    # Native history reset removes raw copies left by Mem0.delete.
    assert recent._memory.db.connection.execute("SELECT count(*) FROM history").fetchone()[0] == 0


def test_native_capacity_deletes_oldest_batches(native_recent, monkeypatch):
    recent = native_recent
    for i in range(4):
        write(recent, item(source=str(i), date=datetime(2026, 11, 10+i, tzinfo=timezone.utc)))
    recent.policy.cleanup_batch_size = 1
    recent.policy.cleanup_target_bytes = 200
    recent.policy.cleanup_trigger_bytes = 350
    recent.policy.max_bytes = 500
    # Inject a tiny deterministic meter to exercise watermarks without 8 GB data.
    monkeypatch.setattr(recent, "disk_bytes", lambda: len(list(recent._records())) * 100)
    report = recent.maintain(force=True)
    assert report["disk_bytes"] == 200 and report["deleted"] == 2
    assert {r.payload["rems_segment"] for r in recent._records()} == {"2", "3"}


def test_native_real_disk_reclaimed(native_recent):
    recent = native_recent
    for i in range(3):
        write(recent, item(text=("原始记录" * 20000) + str(i), source=str(i)))
    before = recent.disk_bytes()
    old = recent._clock()
    recent._clock = lambda: old + 366 * 86400
    recent.maintain(force=True)
    assert recent.disk_bytes() < before
    assert list(recent._records()) == []


def test_capacity_failure_does_not_acknowledge_or_write(native_recent, monkeypatch):
    recent = native_recent
    recent._get_memory()
    monkeypatch.setattr(recent, "disk_bytes", lambda: recent.policy.max_bytes)
    with pytest.raises(RecentCapacityError):
        write(recent, item())
    assert list(recent._records()) == []


def fragment(identity, content, source="s", raw=None, level="L1"):
    return RecalledFragment(event_id=identity, text=raw or content, content=content,
        source_ids=[source], summary_level=level)


def test_fusion_respects_budget_limit_and_keeps_complementary_summary():
    long = [fragment("e", "长期概括"), fragment("e2", "另一事件", "s2")]
    recent = [fragment("mem0:r", "原话细节", level="raw")]
    out = merge_memories(long, recent, budget_chars=8, limit=2)
    assert [f.event_id for f in out] == ["mem0:r", "e"]
    assert sum(len(f.content) for f in out) == 8
    assert merge_memories(long, recent, budget_chars=100, limit=0) == ()


def test_fusion_deduplicates_same_source_raw_without_clipping():
    raw = "具体原话"
    out = merge_memories([fragment("e", "摘要", raw=raw)],
        [fragment("mem0:r", raw, level="raw")], budget_chars=100)
    assert len(out) == 1 and out[0].event_id == "mem0:r"
    assert merge_memories([], [fragment("r", "很长的完整材料")], budget_chars=2) == ()


def test_legacy_block_budget_never_truncates_recent_raw(config):
    from rems.services.recall_budget import RecallBudgetManager
    manager = RecallBudgetManager(config)
    recent = fragment("mem0:r", "原文" * 150, level="raw").model_copy(update={"kind": "recent"})
    assert manager.allocate([recent], budget=200) == []
    selected = manager.allocate([recent], budget=300)
    assert selected[0]["used"] == 300 and selected[0]["level"] == "raw"
    assert manager.allocate([recent], budget=0) == []


def test_recent_dated_duplicates_keep_sources_and_respect_speaker_and_day():
    def greeting(identity, hour, *, who="stone", day=16):
        return RecalledFragment(event_id=identity, text="你好。我是匠石。",
            content=f"〔{day}日 {hour}点〕你好。我是匠石。", kind="recent",
            summary_level="raw", source_ids=[identity], interlocutor=who,
            occurred_at=datetime(2026, 11, day, hour, tzinfo=timezone.utc))
    out = merge_memories([], [greeting("a", 1), greeting("b", 2), greeting("c", 3, who="other"),
                             greeting("d", 1, day=15)], budget_chars=1000)
    assert len(out) == 3 and out[0].source_ids == ["a", "b"]
    assert out[0].signals["merged_occurrences"] == 2
    unknown = merge_memories([], [greeting("u1", 1, who=None), greeting("u2", 2, who=None)],
                             budget_chars=1000)
    assert len(unknown) == 1 and unknown[0].source_ids == ["u1", "u2"]


def test_parallel_pipeline_ingest_recall_and_retry(config, fake_llm, native_recent, monkeypatch):
    from rems.pipeline import REMSPipeline
    config.recent_memory.enabled = False
    pipe = REMSPipeline.from_config(config, llm=fake_llm)
    pipe.recent_memory = native_recent
    batch = MemoryBatch(subject_id="stone", experiences=(item(),))
    barrier = threading.Barrier(2, timeout=3)
    original_long, original_recent = pipe._ingest_long_term, native_recent.ingest_batch
    def long_write(batch):
        barrier.wait()
        return original_long(batch)
    def recent_write(batch):
        barrier.wait()
        return original_recent(batch)
    monkeypatch.setattr(pipe, "_ingest_long_term", long_write)
    monkeypatch.setattr(native_recent, "ingest_batch", recent_write)
    fake_llm.push_response({"events": [{"indices": [1]}], "residual": []})
    fake_llm.push_response({"summaries": {"L1": "维修费358元。"}, "objects": []})
    result = pipe.ingest_batch(batch)
    assert not result.errors and result.stored_marks["one"]
    query_barrier = threading.Barrier(2, timeout=3)
    def long_read(*args, **kwargs):
        query_barrier.wait()
        return (fragment("long", "长期概括", source="different"),)
    def recent_read(*args, **kwargs):
        query_barrier.wait()
        return (fragment("mem0:recent", "近期细节", level="raw"),)
    monkeypatch.setattr(pipe.recall_pipeline, "recall", long_read)
    monkeypatch.setattr(native_recent, "recall", recent_read)
    assert len(pipe.recall("stone", "维修", budget_chars=100, reinforce=False)) == 2
    pipe.close()


def test_l1_index_can_retrieve_raw_only_fact_on_explicit_lookup(config, fake_llm):
    from rems.pipeline import REMSPipeline
    from rems.models.event import Event
    config.recall_event_index_level = "L1"
    pipe = REMSPipeline.from_config(config, llm=fake_llm)
    event = Event(subject_id="stone", content_raw="维修费准确为358元。", summaries={"L1": "修车完成。"})
    pipe.event_repo.save(event)
    pipe.recall_pipeline.index_event(event)
    found = pipe.recall_pipeline.recall("stone", "358", channels=("lexical",), expand_raw=True)
    assert found and "358" in found[0].content and found[0].summary_level == "raw"
    pipe.close()


@pytest.mark.parametrize("stage", [None, 0])
def test_old_unsummarized_events_keep_semantic_index_on_l1_cutover(config, fake_llm, stage):
    from rems.pipeline import REMSPipeline
    from rems.models.event import Event
    config.recall_event_index_level = "L1"
    pipe = REMSPipeline.from_config(config, llm=fake_llm)
    try:
        event = Event(subject_id="stone", content_raw="维修费准确为358元。", memory_stage=stage)
        pipe.event_repo.save(event)
        assert pipe.recall_pipeline.index_event(event) == 1
        stored = pipe.recall_pipeline._vector_store.get(event.event_id)
        assert stored["representation_level"] == "L0"
        found = pipe.recall_pipeline.recall("stone", event.content_raw,
            channels=("semantic",), reinforce=False)
        assert found and found[0].event_id == event.event_id
        event.summaries = {"L1": "修车费用358元。"}
        pipe.event_repo.save(event)
        pipe.recall_pipeline.index_event(event)
        assert pipe.recall_pipeline._vector_store.get(event.event_id)["representation_level"] == "L1"
    finally:
        pipe.close()


def test_recent_failure_withholds_receipts_and_retry_recovers_without_duplicate(
        config, fake_llm, native_recent, monkeypatch):
    from rems.pipeline import REMSPipeline
    pipe = REMSPipeline.from_config(config, llm=fake_llm)
    pipe.recent_memory = native_recent
    original = native_recent.ingest_batch
    def fail(batch):
        raise RuntimeError("recent unavailable")
    monkeypatch.setattr(native_recent, "ingest_batch", fail)
    fake_llm.push_response({"events": [{"indices": [1]}], "residual": []})
    fake_llm.push_response({"summaries": {"L1": "维修358元。"}, "objects": []})
    batch = MemoryBatch(subject_id="stone", experiences=(item(),))
    from rems.pipeline import MemoryIngestError
    with pytest.raises(MemoryIngestError) as error:
        pipe.ingest_batch(batch)
    failed = error.value.result
    assert failed.errors and not failed.stored_marks
    assert len(pipe.event_repo.list_all()) == 1
    monkeypatch.setattr(native_recent, "ingest_batch", original)
    recovered = pipe.ingest_batch(batch)
    assert not recovered.errors and recovered.stored_marks["one"]
    assert len(pipe.event_repo.list_all()) == len(list(native_recent._records())) == 1
    pipe.close()


def test_long_failure_does_not_prevent_recent_write(config, fake_llm, native_recent, monkeypatch):
    from rems.pipeline import REMSPipeline
    pipe = REMSPipeline.from_config(config, llm=fake_llm)
    pipe.recent_memory = native_recent
    def fail(batch):
        raise RuntimeError("long unavailable")
    monkeypatch.setattr(pipe, "_ingest_long_term", fail)
    from rems.pipeline import MemoryIngestError
    with pytest.raises(MemoryIngestError) as error:
        pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=(item(),)))
    result = error.value.result
    assert result.errors and not result.stored_marks
    assert len(native_recent.recall("stone", "维修")) == 1
    pipe.close()


def test_retry_repairs_long_index_and_empty_placeholders_are_noops(config, fake_llm, native_recent, monkeypatch):
    from rems.pipeline import REMSPipeline, MemoryIngestError
    pipe = REMSPipeline.from_config(config, llm=fake_llm)
    pipe.recent_memory = native_recent
    original = pipe.recall_pipeline.index_event
    def fail(event):
        raise RuntimeError("index unavailable")
    monkeypatch.setattr(pipe.recall_pipeline, "index_event", fail)
    fake_llm.push_response({"events": [{"indices": [1]}], "residual": []})
    fake_llm.push_response({"summaries": {"L1": "维修358元"}, "objects": []})
    batch = MemoryBatch(subject_id="stone", experiences=(item(), item(text="", source="empty")))
    try:
        with pytest.raises(MemoryIngestError):
            pipe.ingest_batch(batch)
        assert len(pipe.event_repo.list_all()) == 1
        monkeypatch.setattr(pipe.recall_pipeline, "index_event", original)
        result = pipe.ingest_batch(batch)
        identity = result.stored_marks["one"][0]
        assert not result.errors and pipe.recall_pipeline._vector_store.get(identity)
        assert len(pipe.event_repo.list_all()) == 1
        assert not pipe.ingest_batch(MemoryBatch(subject_id="stone", experiences=(item(text=""),))).errors
    finally:
        pipe.close()


def test_recall_degrades_to_surviving_store_and_reports_double_failure(
        config, fake_llm, native_recent, monkeypatch):
    from rems.pipeline import REMSPipeline
    pipe = REMSPipeline.from_config(config, llm=fake_llm)
    pipe.recent_memory = native_recent
    write(native_recent, item())
    def fail(*args, **kwargs):
        raise RuntimeError("unavailable")
    monkeypatch.setattr(pipe.recall_pipeline, "recall", fail)
    found = pipe.recall("stone", "维修", reinforce=False)
    assert len(found) == 1 and found[0].event_id.startswith("mem0:")
    assert len(pipe.last_recall_errors) == 1
    monkeypatch.setattr(native_recent, "recall", fail)
    with pytest.raises(RuntimeError, match="Both memory stores failed"):
        pipe.recall("stone", "维修", reinforce=False)
    pipe.close()


def test_capacity_stays_active_after_compaction_crosses_trigger(native_recent, monkeypatch):
    recent = native_recent
    for i in range(4):
        write(recent, item(source=str(i)))
    recent.policy.cleanup_batch_size = 1
    recent.policy.cleanup_target_bytes = 200
    recent.policy.cleanup_trigger_bytes = 350
    original = recent._compact
    reclaimed = [False]
    def compact(**kwargs):
        original(**kwargs)
        reclaimed[0] = True
    monkeypatch.setattr(recent, "_compact", compact)
    monkeypatch.setattr(recent, "disk_bytes",
        lambda: len(list(recent._records())) * 100 - (70 if reclaimed[0] else 0))
    report = recent.maintain(force=True)
    assert report["capacity_cleanup"] and report["disk_bytes"] <= 200
    assert report["deleted"] == 2
