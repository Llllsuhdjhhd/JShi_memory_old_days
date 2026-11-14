"""Time evidence shared by boundary prompts and deterministic guards."""


def comparable(value):
    # REMS ingest uses local naive wall time. Normalize aware values to that clock.
    return value.astimezone().replace(tzinfo=None) if value.tzinfo else value


def label(value):
    return value.isoformat(sep=" ", timespec="minutes") if value else "时间未知"


def cohorts(items, gap_minutes, span_minutes):
    result, group, first, previous = [], 0, None, None
    for index, item in enumerate(items):
        current = comparable(item.occurred_at) if item.occurred_at else None
        if index and ((current is None) != (previous is None) or
                      (current is not None and previous is not None and
                       (current < previous or (current-previous).total_seconds() > gap_minutes*60 or
                        (current-first).total_seconds() > span_minutes*60))):
            group += 1
            first = None
        if current is not None and first is None:
            first = current
        result.append(group)
        previous = current
    return result
