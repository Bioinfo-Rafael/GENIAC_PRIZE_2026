"""Fixed wall-clock profiles. Resuming never renews the deadline."""
import json
from pathlib import Path
import sys


def make_budget(start, profile):
    if profile not in ('60m', '15m'):
        raise ValueError('OSW_PROFILE must be 60m or 15m')
    short = profile == '15m'
    seconds = 900 if short else 3600
    return dict(started_epoch=start, deadline_epoch=start+seconds,
                train_deadline_epoch=start+(360 if short else 2040),
                max_seconds=seconds, profile=profile,
                maximum_questions=16 if short else 128,
                prepare_seconds=120 if short else 240,
                fetch_seconds=150 if short else 300,
                train_finish_seconds=390 if short else 2100,
                merge_finish_seconds=510 if short else 2550,
                submit_finish_seconds=870 if short else 3420,
                skip_evaluation=short,
                finish_submission=short,
                policy=('15-minute target; bounded training and merge; once submission starts allow completion beyond target' if short else 'Absolute wall clock, never reset on resume; preserve submission time'))


if __name__ == '__main__':
    root, start, profile = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    path = root / 'budget.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('profile', '60m') != profile:
            raise ValueError('Cannot change an existing run profile; use a new run ID')
    else:
        path.write_text(json.dumps(make_budget(start, profile), indent=2))
