"""回忆可插拔 provider（docs/project-overview.md）：Embedding / Reranker / VectorStore 接口与本地实现。"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Protocol, Sequence

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """文本嵌入接口（本地模型 / API / hash 均可插拔）。"""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class RerankerProvider(Protocol):
    """精排接口（cross-encoder / 无精排时用 NullReranker）。"""

    def rerank(self, query: str, candidates: Sequence[str]) -> list[float]: ...


def versioned_collection_name(base: str, version: int) -> str:
    """嵌入版本变了就换集合名，旧集合留在原地。"""
    suffix = f"_emb{int(version)}"
    if base.endswith(suffix):
        return base
    return f"{base}{suffix}"


class RecallVectorStore(Protocol):
    """回忆向量库接口（Qdrant 本地 / 远程 / 内存均可插拔）。"""

    def upsert(self, event_id: str, vector: list[float], payload: dict) -> None: ...

    def get(self, logical_id: str) -> dict | None: ...

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        payload_filter: dict | None = None,
        any_of: dict[str, list] | None = None,
        ranges: dict[str, tuple[float, float]] | None = None,
    ) -> list[dict]: ...

    def delete(self, event_id: str) -> None: ...

    def list_event_logical_ids(self, event_id: str) -> list[str]: ...


class HashEmbedding:
    """确定性 hash 嵌入：离线测试 / 无模型环境用。"""

    def __init__(self, dim: int = 64):
        self._dim = dim

    def _vec(self, text: str) -> list[float]:
        out = [0.0] * self._dim
        for i, ch in enumerate(text):
            h = int(hashlib.md5(ch.encode("utf-8")).hexdigest(), 16)
            out[h % self._dim] += 1.0
        norm = sum(v * v for v in out) ** 0.5 or 1.0
        return [v / norm for v in out]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class SentenceTransformerEmbedding:
    """本地 sentence-transformers 嵌入（bge-base-zh / BGE-M3 等）。"""

    def __init__(self, model_name: str, query_instruction: str = "", *, device: str = "cpu"):
        self._model_name = model_name
        self._query_instruction = query_instruction
        self._device = device
        self._model = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            self._ensure()
            return self._model.encode(list(texts), normalize_embeddings=True).tolist()

    def split_document(self, text: str) -> list[str]:
        """按实际编码器 token 上限分块；保留原文切片和相邻 32 token 重叠。"""
        with self._lock:
            self._ensure()
            tokenizer = self._model.tokenizer
            capacity = int(self._model.max_seq_length) - tokenizer.num_special_tokens_to_add(pair=False)
            if capacity < 2:
                raise ValueError("Encoder document capacity is too small")
            offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                                return_offsets_mapping=True)["offset_mapping"]
            if not offsets:
                return [text]
            chunks = []
            start = 0
            while start < len(offsets):
                end = min(start + capacity, len(offsets))
                left = 0 if start == 0 else offsets[start][0]
                right = len(text) if end == len(offsets) else offsets[end][0]
                chunk = text[left:right]
                # Re-tokenizing a substring can change a boundary wordpiece.
                while len(tokenizer(chunk, add_special_tokens=True)["input_ids"]) > self._model.max_seq_length:
                    end -= 1
                    if end <= start:
                        raise ValueError("Unable to form an encoder-sized raw chunk")
                    right = offsets[end][0]
                    chunk = text[left:right]
                chunks.append(chunk)
                if end == len(offsets):
                    break
                start = max(start + 1, end - min(32, capacity // 4))
            return chunks

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            self._ensure()
            return self._model.encode(
                [self._query_instruction + text], normalize_embeddings=True
            )[0].tolist()


class NullReranker:
    """无精排：保持候选顺序（纯 RRF 场景）。"""

    def rerank(self, query: str, candidates: Sequence[str]) -> list[float]:
        return [1.0] * len(candidates)


class BgeReranker:
    """Local cross-encoder; model choice is configurable, weights stay local."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", *,
                 onnx_file: str | None = None, threads: int = 4, device: str = "cpu"):
        if not 1 <= threads <= 64:
            raise ValueError("Reranker threads must be between 1 and 64")
        if onnx_file and device != "cpu":
            raise ValueError("CPU ONNX weights require device=cpu; clear onnx_file for CUDA PyTorch")
        self._model_name = model_name
        self._onnx_file = onnx_file
        self._threads = threads
        self._device = device
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is None:
            if self._onnx_file:
                from huggingface_hub import hf_hub_download
                from transformers import AutoTokenizer
                import onnxruntime as ort
                path = hf_hub_download(self._model_name, self._onnx_file, local_files_only=True)
                options = ort.SessionOptions()
                options.intra_op_num_threads = self._threads
                options.inter_op_num_threads = 1
                options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                options.add_session_config_entry("session.inter_op.allow_spinning", "0")
                self._model = ort.InferenceSession(path, sess_options=options,
                    providers=["CPUExecutionProvider"])
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, local_files_only=True)
                return
            from sentence_transformers import CrossEncoder
            import torch

            self._model = CrossEncoder(self._model_name, local_files_only=True,
                device=self._device, max_length=512, activation_fn=torch.nn.Identity())

    def rerank(self, query: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        with self._lock:
            self._ensure()
            if self._onnx_file:
                # Official exported cross-encoder; only tokenization and native
                # ONNX inference here. Raw logits preserve the ranking order.
                encoded = self._tokenizer([query]*len(candidates), list(candidates),
                    padding=True, truncation=True, max_length=512, return_tensors="np")
                names = {item.name for item in self._model.get_inputs()}
                scores = self._model.run(None, {k:v for k,v in encoded.items() if k in names})[0]
                return [float(row[0]) for row in scores]
            pairs = [(query, c) for c in candidates]
            scores = self._model.predict(pairs, batch_size=8, show_progress_bar=False)
            return [float(s) for s in scores]


class QdrantRecallVectorStore:
    """Qdrant 本地 / 内存向量库（qdrant-client 本地模式，无需 docker）。"""

    def __init__(
        self,
        collection: str = "events",
        *,
        url: str | None = None,
        path: str | None = None,
        api_key: str | None = None,
    ):
        from qdrant_client import QdrantClient

        if path:
            self._client = QdrantClient(path=path)
        elif url and url != ":memory:":
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(":memory:")
        self._collection = collection
        self._dim: int | None = None
        self._local = bool(path) or not url or url == ":memory:"
        self._uniform_payload = None
        # 同实例并发 upsert/search：本地 Qdrant 无细粒度内部锁，窄锁保护。
        self._lock = threading.RLock()

    def _ensure_collection(self, dim: int) -> None:
        if self._dim == dim:
            return
        from qdrant_client import models

        exists = False
        try:
            exists = self._client.collection_exists(self._collection)
        except Exception:  # noqa: BLE001
            exists = False
        if exists:
            info = self._client.get_collection(self._collection)
            size = int(info.config.params.vectors.size)
            if size != dim:
                raise RuntimeError(
                    f"集合 {self._collection} 的向量维度是 {size}，当前嵌入维度是 {dim}。"
                    "请提高 recall_embedding_version 后使用新集合，旧集合保留。"
                )
            self._dim = dim
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        self._dim = dim

    def upsert(self, event_id: str, vector: list[float], payload: dict) -> None:
        with self._lock:
            self._uniform_payload = None
            from qdrant_client import models

            self._ensure_collection(len(vector))
            self._client.upsert(
                collection_name=self._collection,
                points=[
                    models.PointStruct(
                        id=hashlib.md5(event_id.encode("utf-8")).hexdigest(),
                        vector=vector,
                        payload={"event_id": event_id, **payload},
                    )
                ],
            )

    def get(self, logical_id: str) -> dict | None:
        with self._lock:
            try:
                if not self._client.collection_exists(self._collection):
                    return None
            except Exception:  # noqa: BLE001
                return None
            found = self._client.retrieve(
                collection_name=self._collection,
                ids=[self._point_id(logical_id)],
                with_payload=True,
                with_vectors=False,
            )
            if not found or not found[0].payload:
                return None
            return dict(found[0].payload)

    def list_event_logical_ids(self, event_id: str) -> list[str]:
        """Return event and fact point keys currently stored for one event."""
        with self._lock:
            from qdrant_client import models

            if not self._client.collection_exists(self._collection):
                return []
            condition = models.Filter(must=[models.FieldCondition(
                key="event_id", match=models.MatchValue(value=event_id),
            )])
            found: list[str] = []
            offset = None
            while True:
                points, offset = self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=condition,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    if payload.get("type") == "event":
                        index = int(payload.get("chunk_index") or 0)
                        found.append(event_id if index == 0 else f"{event_id}#raw#{index}")
                    elif payload.get("type") == "object_event_fact" and payload.get("object_id"):
                        found.append(f"{event_id}#fact#{payload['object_id']}")
                    elif payload.get("type") == "event_detail" and payload.get("source_event_id"):
                        found.append(f"{event_id}#detail#{payload['source_event_id']}")
                if offset is None:
                    break
            return found

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        payload_filter: dict | None = None,
        any_of: dict[str, list] | None = None,
        ranges: dict[str, tuple[float, float]] | None = None,
    ) -> list[dict]:
        with self._lock:
            from qdrant_client import models

            self._ensure_collection(len(vector))
            payload_filter = self._effective_filter(payload_filter)
            conditions = []
            for key, value in (payload_filter or {}).items():
                conditions.append(models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                ))
            for key, values in (any_of or {}).items():
                if not values:
                    continue
                conditions.append(models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=list(values)),
                ))
            for key, bounds in (ranges or {}).items():
                start, end = bounds
                conditions.append(models.FieldCondition(
                    key=key,
                    range=models.Range(gte=start, lte=end),
                ))
            qfilter = models.Filter(must=conditions) if conditions else None
            hits: list[dict] = []
            if hasattr(self._client, "query_points"):
                resp = self._client.query_points(
                    collection_name=self._collection,
                    query=vector,
                    query_filter=qfilter,
                    limit=top_k,
                    with_payload=True,
                )
                for h in resp.points:
                    if h.payload:
                        hits.append({
                            "event_id": h.payload.get("event_id"),
                            "score": float(h.score),
                            "payload": h.payload,
                        })
            else:
                raw = self._client.search(
                    collection_name=self._collection,
                    query_vector=vector,
                    query_filter=qfilter,
                    limit=top_k,
                    with_payload=True,
                )
                hits = [
                    {"event_id": h.payload.get("event_id"), "score": float(h.score), "payload": h.payload}
                    for h in raw
                    if h.payload
                ]
            return hits

    def _effective_filter(self, payload_filter):
        """Elide proven constant filters only on this exclusively owned local DB.

        Qdrant local evaluates metadata predicates in Python. Uniform subject /
        model / version constraints are tautologies. Verify once from native
        payloads, invalidate on every mutation, never assume this for a server.
        """
        if not self._local or not payload_filter:
            return payload_filter
        if self._uniform_payload is None:
            keys = ("subject_id", "embedding_model", "embedding_version")
            uniform = None
            offset = None
            while True:
                rows, offset = self._client.scroll(self._collection, limit=512, offset=offset,
                    with_payload=list(keys), with_vectors=False)
                for row in rows:
                    values = {k: row.payload[k] for k in keys if k in (row.payload or {})}
                    uniform = values if uniform is None else {
                        k:v for k,v in uniform.items() if k in values and v == values[k]}
                if offset is None:
                    break
            self._uniform_payload = uniform or {}
        return {k:v for k,v in payload_filter.items()
                if k not in self._uniform_payload or self._uniform_payload[k] != v}

    def search_complete(self, vector, *, payload_filter=None, any_of=None,
                        ranges=None, max_points=4096):
        """One exact native search for a small index; None means use top-K.

        The lock covers count and query, so a concurrent upsert cannot make a
        truncated pool look complete. No vectors or results survive this call.
        """
        with self._lock:
            self._ensure_collection(len(vector))
            count = self._client.get_collection(self._collection).points_count
            if count is None or count > max_points:
                return None
            return self.search(vector, top_k=max(1, count),
                payload_filter=payload_filter, any_of=any_of, ranges=ranges)

    def delete(self, event_id: str) -> None:
        with self._lock:
            self._uniform_payload = None
            try:
                if not self._client.collection_exists(self._collection):
                    return
            except Exception:  # noqa: BLE001
                return
            self._client.delete(
                collection_name=self._collection,
                points_selector=[self._point_id(event_id)],
            )

    @staticmethod
    def _point_id(logical_id: str) -> str:
        return hashlib.md5(logical_id.encode("utf-8")).hexdigest()
