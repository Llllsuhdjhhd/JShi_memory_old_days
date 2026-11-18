"""Audit actual JShi data on consistent SQLite backups; never open live Qdrant.

Creates isolated snapshots and a recent replay manifest. It neither switches
JShi nor resets its memory cursor nor sends content to a model.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rems.eval.jshi_snapshot import experience_mapper
from rems.storage.database import Database, EventRecord
from rems.storage.repository import EventRepository, MetabolismRepository


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def backup(source, target):
    with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as src:
        src.execute("PRAGMA query_only=ON")
        with sqlite3.connect(target) as dst:
            src.backup(dst, pages=256)


def audit(jshi_root, output, *, subject="stone", now=None):
    root, output = Path(jshi_root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=365)
    output.mkdir(parents=True)
    backup(root / ".jshi" / "subject.sqlite3", output / "subject.sqlite3")
    backup(root / ".jshi" / "rems" / "rems.db", output / "long.db")
    report = {"captured_at": now.isoformat(), "subject": subject,
        "live_qdrant_opened": False, "live_databases_modified": False,
        "note": "Two independent SQLite backups; not an atomic cross-database snapshot"}
    convert = experience_mapper(root)
    replay, differences, failures = [], [], []
    counts, output_kinds, delivery = Counter(), Counter(), Counter()
    segment_ids, all_sources = set(), set()
    with sqlite3.connect(output / "subject.sqlite3") as db:
        for (payload,) in db.execute("SELECT payload FROM experience_segments WHERE subject_id=? ORDER BY sequence", (subject,)):
            row = json.loads(payload)
            counts["segments"] += 1
            segment_ids.add(row["segment_id"])
            all_sources.update(row.get("source_ids") or [])
            output_kinds[row["output_kind"]] += 1
            tag = (row.get("state_delta") or {}).get("delivery") or "unmarked"
            delivery[tag] += 1
            counts["raw_text_segments"] += bool((row.get("text_raw") or "").strip())
            try:
                experience = convert(row)
            except Exception as exc:
                failures.append({"segment_id": row["segment_id"], "error": str(exc)})
                continue
            if experience is None:
                counts["not_memory_text"] += 1
                continue
            counts["mapped_experiences"] += 1
            counts["mapped_chars"] += len(experience.text)
            counts["empty_sources"] += not bool(experience.source_ids)
            moment = experience.occurred_at
            if moment.tzinfo is None:
                moment = moment.astimezone()
            if moment < cutoff:
                counts["expired"] += 1
                continue
            replay.append(experience.model_dump(mode="json"))
            delta = []
            if experience.interlocutor != row.get("actor_object_id"):
                counts["old_evaluator_missing_interlocutor"] += 1
                delta.append("interlocutor")
            old_origin = "external" if row.get("actor_kind") == "external" else "internal"
            if experience.origin != old_origin:
                counts["old_evaluator_wrong_origin"] += 1
                delta.append("origin")
            if experience.text != (row.get("text_raw") or ""):
                counts["old_evaluator_missing_speaker_or_action"] += 1
                delta.append("text")
            if delta and len(differences) < 12:
                differences.append({"segment_id": row["segment_id"], "output_kind": row["output_kind"],
                    "differences": delta, "old_interlocutor": row.get("actor_object_id"),
                    "actual_interlocutor": experience.interlocutor,
                    "actual_text": experience.text[:220]})
        ledger = db.execute("SELECT payload FROM experience_ledger WHERE subject_id=?", (subject,)).fetchone()
        report["ledger_state"] = json.loads(ledger[0]) if ledger else None
    report["ledger"] = {**counts, "eligible_recent": len(replay),
        "output_kinds": output_kinds, "delivery": delivery, "conversion_failures": failures,
        "mapping_examples": differences}
    save(output / "recent-replay.json", replay)

    db = Database("sqlite:///" + (output / "long.db").as_posix())
    stats, stages, errors = Counter(), Counter(), []
    try:
        db.create_tables()  # Compatibility/schema upgrades run on the copy only.
        event_ids, event_sources, owned_fragments = set(), set(), set()
        with db.session() as session:
            for record in session.query(EventRecord).yield_per(100):
                try:
                    event = EventRepository._to_model(record)
                except Exception as exc:
                    errors.append({"event_id": record.event_id, "error": str(exc)})
                    continue
                event_ids.add(event.event_id)
                stats["events_readable"] += 1
                stats["raw_nonempty"] += bool(event.content_raw.strip())
                stats["has_l1"] += bool(event.summaries.get("L1"))
                stats["no_l1"] += not bool(event.summaries.get("L1"))
                stats["normal_long_length"] += 2000 <= len(event.content_raw) < 3000
                stats["shorter_than_2000"] += len(event.content_raw) < 2000
                stats["hidden"] += bool(event.refinement_hidden_by)
                stats["tombstoned"] += bool(event.is_tombstoned)
                stats["without_source_ids"] += not bool(event.source_ids)
                stages[str(event.memory_stage)] += 1
                event_sources.update(event.source_ids)
                owned_fragments.update(event.source_fragment_ids)
        pending = MetabolismRepository(db).get_unclosed_events()
        with sqlite3.connect(output / "long.db") as connection:
            marks = [(r[0], json.loads(r[1] or "[]")) for r in connection.execute(
                "SELECT input_id,event_ids FROM stored_marks WHERE subject_id=?", (subject,))]
        dangling = [(identity, eid) for identity, ids in marks for eid in ids if eid not in event_ids]
        marked_segments = {identity for identity, _ in marks}
        report["long_term"] = {**stats, "stages": stages, "conversion_failures": errors,
            "pending_events_readable": len(pending), "stored_marks": len(marks),
            "marks_with_missing_event": len(dangling), "missing_event_examples": dangling[:10],
            "marked_segments_still_in_ledger": sum(identity in segment_ids for identity, _ in marks),
            "event_sources_still_in_ledger": len(event_sources & all_sources),
            "event_sources_not_in_ledger": len(event_sources - all_sources),
            "owned_fragments_still_in_ledger": len(owned_fragments & segment_ids),
            "recent_replay_without_long_receipt": sum(item["segment_id"] not in marked_segments for item in replay)}
    finally:
        db.engine.dispose()
    # Ledger state can contain context text; keep detailed evidence private.
    save(output / "audit.json", report)
    return {"output": str(output), "ledger": {k:v for k,v in report["ledger"].items() if k != "mapping_examples"},
            "long_term": report["long_term"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jshi-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--subject", default="stone")
    args = parser.parse_args()
    print(json.dumps(audit(args.jshi_root, args.out, subject=args.subject), ensure_ascii=False, indent=2))
