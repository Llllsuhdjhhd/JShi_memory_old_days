"""Bound expensive cross-encoding with complementary semantic/lexical lanes."""
from collections import Counter
import math
import re

from ..utils.simplified_text import simplify_text


def tokens(text):
    # Chinese needs character n-grams; Latin words should not match substrings.
    parts = re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", simplify_text(text).casefold())
    return Counter(token for part in parts for token in (
        [part] if part.isascii() else
        [part[i:i+n] for n in (2, 3) for i in range(len(part)-n+1)]))


def lexical_ranks(query, texts):
    """BM25 within this call's bounded pool; no persistent corpus cache."""
    docs = [tokens(text) for text in texts]
    if not docs:
        return []
    df = Counter(term for doc in docs for term in doc)
    lengths = [sum(doc.values()) for doc in docs]
    average = max(1, sum(lengths)/len(docs))
    terms = tokens(query)
    scores = [sum(
        math.log(1+(len(docs)-df[t]+0.5)/(df[t]+0.5)) * doc[t]*2.2 /
        (doc[t]+1.2*(0.25+0.75*length/average))
        for t in terms if doc[t]) for doc, length in zip(docs, lengths)]
    return sorted((i for i, score in enumerate(scores) if score > 0),
                  key=lambda i: scores[i], reverse=True)


def rerank_candidates(query, ordered, reranker, *, pool_size=100, top_n=6):
    """Reorder a small union, retaining every other eligible candidate as fallback."""
    if not ordered or reranker is None or top_n <= 0:
        return ordered
    pool = ordered[:pool_size]
    # L1 remains primary. Original text is only used when no L1 exists.
    texts = [c.levels.get("L1") or c.raw for c in pool]
    lexical = lexical_ranks(query, texts)
    quota = max(1, top_n//2)
    indexes = list(dict.fromkeys([*range(min(quota, len(pool))), *lexical[:quota]]))
    # Reuse spare slots when the two lanes overlap.
    indexes = list(dict.fromkeys([*indexes, *range(len(pool))]))[:top_n]
    scores = reranker.rerank(query, [texts[i] for i in indexes])
    if len(scores) != len(indexes) or not all(math.isfinite(s) for s in scores):
        raise ValueError("Reranker returned invalid scores")
    ranked = sorted(zip(indexes, scores), key=lambda item: item[1], reverse=True)
    selected = set(indexes)
    return [pool[i] for i, _score in ranked] + [
        c for i, c in enumerate(ordered) if i not in selected]
