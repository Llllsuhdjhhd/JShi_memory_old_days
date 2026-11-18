from types import SimpleNamespace
from unittest.mock import Mock

from rems.recall.ranking import rerank_candidates
from rems.recall.providers import QdrantRecallVectorStore
from rems.recall.pipeline import RecallPipeline


def test_reranker_thread_setting_reaches_pipeline_without_changing_limits(config):
    from rems.pipeline import REMSPipeline
    config.embedding.provider = 'local'
    config.recall_rerank_threads = 8
    pipe = REMSPipeline.from_config(config)
    try:
        assert pipe.recall_pipeline._reranker._threads == 8
        assert pipe.recall_pipeline._rerank_top_n == config.recall_rerank_top_n == 6
        assert pipe.recall_pipeline._rerank_pool == config.recall_rerank_pool == 100
    finally:
        pipe.close()


def test_zero_reranker_threads_cannot_silently_enable_unbounded_runtime(monkeypatch):
    import pytest
    from pydantic import ValidationError
    from rems.config import REMSConfig
    monkeypatch.setenv('REMS_RECALL_RERANK_THREADS','0')
    with pytest.raises(ValidationError):
        REMSConfig(_env_file=None)


def test_gpu_configuration_selects_both_models_and_rejects_cpu_onnx(config):
    import pytest
    from rems.pipeline import REMSPipeline
    from rems.recall.providers import BgeReranker
    with pytest.raises(ValueError, match='CPU ONNX'):
        BgeReranker(onnx_file='cpu-quantized.onnx', device='cuda')
    config.embedding.provider='local'
    config.embedding.device='cuda'
    config.recall_rerank_device='cuda'
    config.recall_rerank_onnx_file=''
    pipe=REMSPipeline.from_config(config)
    try:
        assert pipe.recall_pipeline._embedding._device=='cuda'
        assert pipe.recall_pipeline._reranker._device=='cuda'
        assert pipe.recall_pipeline._rerank_top_n==6
    finally:
        pipe.close()


def test_specific_evidence_enters_small_rerank_lane():
    ordered = [SimpleNamespace(levels={"L1": "大家正在讨论"}, raw="old") for _ in range(60)]
    target = SimpleNamespace(levels={"L1": "报告保存在蓝色文件夹"}, raw="wrong raw")
    ordered.append(target)
    ranker = Mock()
    ranker.rerank.side_effect = lambda q, texts: [float("蓝色" in t) for t in texts]
    result = rerank_candidates("报告放在哪个文件夹", ordered, ranker, top_n=6)
    assert result[0] is target
    assert len(result) == len(ordered)
    assert len(ranker.rerank.call_args.args[1]) <= 6
    assert "wrong raw" not in ranker.rerank.call_args.args[1]


def test_small_index_pool_preserves_constraints_and_avoids_requery():
    store = QdrantRecallVectorStore()
    try:
        for eid, subject, model, version, vector in (
            ("a", "s", "m", 1, [1., 0.]), ("b", "s", "m", 1, [0.8, 0.2]),
            ("foreign", "other", "m", 1, [1., 0.]),
            ("old_model", "s", "old", 1, [1., 0.]),
            ("old_version", "s", "m", 0, [1., 0.]),
        ):
            store.upsert(eid, vector, {"event_id": eid, "type": "event",
                "subject_id": subject, "embedding_model": model, "embedding_version": version})
        pipe = RecallPipeline(None, store, Mock(), None, embedding_model="m", embedding_version=1)
        store.search = Mock(wraps=store.search)
        pool = []
        first = pipe._semantic_signals("s", "query", (), None, query_vector=[1.,0.],
                                      top_k=1, complete_pool_sink=pool)
        second = pipe._semantic_signals("s", "query", (), None, query_vector=[1.,0.],
                    event_ids=("b",), top_k=1, prefetched_hits=pool[0])
        assert set(first) == {("event", "a", "")}
        assert set(second) == {("event", "b", "")}
        assert store.search.call_count == 1
        assert store.search_complete([1.,0.], max_points=2) is None
    finally:
        store._client.close()


def test_uniform_filter_proof_is_invalidated_on_write():
    store=QdrantRecallVectorStore()
    try:
        store.upsert("a", [1.,0.], {"event_id":"a", "subject_id":"alice"})
        assert store.search([1.,0.],top_k=10,payload_filter={"subject_id":"alice"})
        assert store._effective_filter({"subject_id":"alice"}) == {}
        store.upsert("b", [1.,0.], {"event_id":"b", "subject_id":"bob"})
        rows=store.search([1.,0.],top_k=10,payload_filter={"subject_id":"alice"})
        assert [r["event_id"] for r in rows] == ["a"]
        assert store._effective_filter({"subject_id":"alice"}) == {"subject_id":"alice"}
        assert store.search([1.,0.],top_k=10,payload_filter={"subject_id":"nobody"}) == []
    finally:
        store._client.close()
