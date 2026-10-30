"""When does the production loop retrain? One small rule, kept next to the state it reads.

  python -m credit_drift.retraining.policy --week 5     # prints the trigger, or nothing

Retrain #1 on the first major data-drift alert; later retrains when a performance-drop alert is about a week the current
champion has not seen. Labels arrive a week late, so the run of week N can use labels through week N-1. The state file
is the retraining history: it also stops a failed gate from retraining again on the same labels.
"""
import argparse
import json
from pathlib import Path

from credit_drift.monitoring.weekly import ROOT, read_json

STATE = ROOT / "retraining.json"


def load_state(path: Path = STATE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_state(events: list[dict], path: Path = STATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events, indent=1), encoding="utf-8")


def champion_labels_through(events: list[dict]) -> int:
    """The last labelled week the serving champion was trained on (0: the Reference only)."""
    return max((e["through_week"] for e in events if e.get("promoted")), default=0)


def trigger_for(week: int, alerts: list[dict], events: list[dict]) -> str | None:
    through = week - 1
    if through < 3 or any(e["through_week"] >= through for e in events):
        return None  # too few labelled weeks, or this labelled data was already tried
    rules = {a["rule"]: a for a in alerts}
    if "performance_drop" in rules and rules["performance_drop"]["about_week"] > champion_labels_through(events):
        return "performance_drop"
    if "data_drift_major" in rules and not events:
        return "data_drift_major"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()
    alerts = (read_json(ROOT / f"week_{args.week:02d}" / "alerts.json") or {}).get("alerts", [])
    if trigger := trigger_for(args.week, alerts, load_state()):
        print(trigger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
