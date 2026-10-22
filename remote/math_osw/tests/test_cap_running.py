import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('cap_running', Path(__file__).parents[1] / 'cap_running.py')
cap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap)


class CapTests(unittest.TestCase):
    def test_only_committed_nonzero_update_is_usable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            work = root / 'chain/stage2/training/rft'
            work.mkdir(parents=True)
            (work / 'latest.json').write_text(json.dumps(dict(checkpoint='checkpoint-1', step=1)))
            (work / 'train.jsonl').write_text('\n'.join(json.dumps(r) for r in [
                dict(step=1, nonzero_lr_updates=0), dict(step=2, nonzero_lr_updates=1)]))
            (work / 'checkpoint-1').mkdir()
            (work / 'checkpoint-1/state.pt').touch()
            self.assertFalse(cap.updated(root))
            (work / 'latest.json').write_text(json.dumps(dict(checkpoint='checkpoint-1', step=2)))
            self.assertTrue(cap.updated(root))

    def run_cap(self, elapsed, updated=False, ready=False):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'budget.json').write_text(json.dumps(dict(started_epoch=1000, deadline_epoch=4600)))
            (root / 'suite_report.json').write_text(json.dumps(dict(submission_ready=ready)))
            with patch.object(sys, 'argv', ['cap', '--root', d]), \
                 patch.object(cap, 'processes', side_effect=[[(10, 'runner'), (20, 'train')], []]), \
                 patch.object(cap, 'updated', return_value=updated), \
                 patch.object(cap.time, 'time', return_value=1000+elapsed), \
                 patch.object(cap.time, 'sleep'), patch.object(cap, 'send') as send, \
                 patch.object(cap.os, 'getpgid', return_value=20), \
                 patch.object(cap.os, 'killpg') as killpg:
                cap.main()
                return send.call_args_list, killpg.call_args_list

    def test_early_finish_signals_training_only(self):
        calls, groups = self.run_cap(180, updated=True)
        self.assertEqual([c.args for c in calls], [(20, signal.SIGTERM)])
        self.assertFalse(groups)

    def test_no_deadline_reset_and_targeted_group_shutdown(self):
        calls, groups = self.run_cap(1190)
        self.assertEqual([c.args for c in calls], [(10, signal.SIGTERM), (20, signal.SIGTERM)])
        self.assertEqual([c.args for c in groups], [(20, signal.SIGKILL)])

    def test_seven_minute_training_cutoff(self):
        calls, _ = self.run_cap(420)
        self.assertEqual([c.args for c in calls], [(20, signal.SIGTERM)])


if __name__ == '__main__':
    unittest.main()
