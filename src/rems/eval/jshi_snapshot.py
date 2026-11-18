"""Replay ledger records through JShi's actual memory-input mapping."""
from pathlib import Path
import sys

from ..port import MemoryExperience


def experience_mapper(jshi_root=None):
    root = Path(jshi_root) if jshi_root else Path(__file__).resolve().parents[4] / "JShi"
    source = root.resolve() / "src"
    if not (source / "jshi" / "memorycontrol" / "inprocess.py").is_file():
        raise ValueError("JShi checkout is required for faithful ledger replay")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from jshi.experienceledger.sqlite import _segment_from_dict
    from jshi.memorycontrol.inprocess import _experience_text, _interlocutor_for, _origin_for

    def convert(payload, *, subject_name="匠石"):
        segment = _segment_from_dict(payload)
        text = _experience_text(segment, subject_name)
        if not text.strip():
            return None
        return MemoryExperience(subject_id=segment.subject_id, text=text,
            objects=dict(segment.objects or {}),
            interlocutor=_interlocutor_for(segment, segment.subject_id),
            source_ids=tuple(segment.source_ids), segment_id=segment.segment_id,
            occurred_at=segment.occurred_at, origin=_origin_for(segment.output_kind))

    return convert
