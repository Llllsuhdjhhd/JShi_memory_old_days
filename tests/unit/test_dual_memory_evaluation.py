import json
import sqlite3

import pytest

from scripts.evaluate_dual_memory import prepare, run, score_case
from rems.port import RecalledFragment


def test_prepare_reads_latest_actual_text_without_modifying_source(tmp_path):
    source = tmp_path / "subject.sqlite3"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE experience_segments(subject_id, sequence, payload)")
        for index, text in enumerate(("旧记录", None, "新记录")):
            db.execute("INSERT INTO experience_segments VALUES(?,?,?)", ("stone", index,
                json.dumps({"text_raw": text, "segment_id": str(index),
                    "subject_id": "stone", "sequence": index, "output_kind": "external_input",
                    "actor_kind": "external", "actor_object_id": "lee",
                    "occurred_at": "2026-11-16T12:00:00+08:00"})))
    before = source.read_bytes()
    output = tmp_path / "snapshot"
    result = prepare(source, output, subject="stone", limit=1, max_chars=100)
    assert result["segments"] == 1
    items = json.loads((output / "experiences.json").read_text(encoding="utf-8"))
    assert items[0]["text"] == "新记录" and items[0]["interlocutor"] == "lee"
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match="new output"):
        prepare(source, output, subject="stone", limit=1, max_chars=100)


def test_quality_scores_delivered_content_instead_of_hidden_raw_text():
    case = {"expected_sources": ["one", "two"], "expected_quotes": ["358元", "原话"]}
    fragment = RecalledFragment(event_id="e", source_ids=["one"],
        text="358元 原话", content="只交付358元")
    score = score_case([fragment], case)
    assert score["source_recall"] == .5 and score["quote_coverage"] == .5
    assert score_case([], case)["quote_coverage"] == 0


def test_evaluation_rejects_invalid_budgets_before_opening_databases(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        run(tmp_path / "missing", tmp_path / "missing", tmp_path / "out",
            env_file=None, max_calls=0, batch_chars=3000, budget_chars=4000,
            limit=6, repeats=3)
    assert not (tmp_path / "out").exists()


def test_replay_uses_actual_jshi_speaker_delivery_and_interlocutor_mapping():
    from rems.eval.jshi_snapshot import experience_mapper
    convert = experience_mapper()
    row = {"subject_id": "stone", "sequence": 1, "segment_id": "real-format",
           "output_kind": "subject_reply", "actor_kind": "subject", "actor_object_id": None,
           "text_raw": "我已经完成了", "objects": {"甲": "lee"},
           "source_ids": ["source"], "occurred_at": "2026-11-17T12:00:00+08:00"}
    experience = convert(row)
    assert experience.text == "匠石：我已经完成了"
    assert experience.interlocutor == "lee" and experience.origin == "external"
    assert convert({**row, "state_delta": {"delivery": "planned_audio"}}) is None
    action = convert({**row, "text_raw": None, "output_kind": "subject_state",
                      "state_delta": {"embodied": "点头"}})
    assert action.text == "匠石：（动作：点头）"
