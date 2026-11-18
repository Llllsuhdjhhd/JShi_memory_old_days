"""Merge independent ranked lists without comparing incompatible store scores."""
from __future__ import annotations

from itertools import zip_longest

from ..utils.simplified_text import simplify_text


def _coalesce_recent(items):
    """Repeated same-interlocutor, same-origin, same-day text uses one slot.

    Different dates/speakers remain separate. Keep all source identities so a
    deduplicated delivery remains traceable to each original occurrence.
    """
    result, positions = [], {}
    for item in items:
        if item.kind != "recent" or item.occurred_at is None:
            result.append(item)
            continue
        key = ("".join(simplify_text(item.text).split()), item.interlocutor,
               item.occurred_at.astimezone().date(), item.signals.get("source_origin", -1))
        if key not in positions:
            positions[key] = len(result)
            result.append(item)
            continue
        index = positions[key]
        previous = result[index]
        attributions = {a.segment_id: a for a in previous.interlocutor_attributions}
        attributions.update((a.segment_id, a) for a in item.interlocutor_attributions)
        result[index] = previous.model_copy(update={
            "source_ids": list(dict.fromkeys([*previous.source_ids, *item.source_ids])),
            "interlocutor_attributions": list(attributions.values()),
            "signals": {**previous.signals, "merged_occurrences": previous.signals.get("merged_occurrences", 1) + 1},
        })
    return result


def merge_memories(long_term, recent, *, budget_chars, limit=None):
    if budget_chars <= 0 or (limit is not None and limit <= 0):
        return ()
    selected, remaining = [], budget_chars
    seen_ids, seen_texts = set(), set()
    # Reciprocal rank interleaving gives both stores access to the shared budget.
    # Recent wins equal rank so exact current evidence precedes its abstraction.
    for rank, pair in enumerate(zip_longest(_coalesce_recent(recent), long_term), 1):
        for item in pair:
            if item is None:
                continue
            text = item.content or item.text
            normalized = "".join(simplify_text(text).split())
            identity = (item.event_id, item.type, item.object_id)
            if identity in seen_ids or normalized in seen_texts:
                continue
            # Equal original evidence from both stores is one hit. A summary
            # with additional facts is complementary and must not be lost merely
            # because a broad source ID overlaps.
            raw = "".join(simplify_text(item.text).split())
            duplicate = any(
                raw == "".join(simplify_text(old.text).split())
                and set(item.source_ids).intersection(old.source_ids)
                and (item.summary_level == "raw" or old.summary_level == "raw")
                for old in selected)
            if duplicate or len(text) > remaining:
                continue
            selected.append(item.model_copy(update={
                "score": 1 / (60 + rank),
                "signals": {**item.signals, "store_score": item.score, "fusion_rank": float(rank)},
            }))
            seen_ids.add(identity)
            seen_texts.add(normalized)
            remaining -= len(text)
            if limit is not None and len(selected) >= limit:
                return tuple(selected)
    return tuple(selected)
