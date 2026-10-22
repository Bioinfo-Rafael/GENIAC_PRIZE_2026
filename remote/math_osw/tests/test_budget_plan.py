import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from budget_plan import make_budget


class BudgetTests(unittest.TestCase):
    def test_short_profile_leaves_inference_time(self):
        b=make_budget(1000,'15m')
        self.assertEqual(b['deadline_epoch'],1900)
        self.assertEqual(b['train_deadline_epoch'],1360)
        self.assertEqual(b['maximum_questions'],16)
        self.assertTrue(b['skip_evaluation'])
        self.assertEqual(b['submit_finish_seconds']-b['merge_finish_seconds'],360)
        self.assertLess(b['train_finish_seconds'],b['merge_finish_seconds'])
        self.assertLess(b['submit_finish_seconds'],b['max_seconds'])

    def test_legacy_and_invalid_profiles(self):
        self.assertEqual(make_budget(1000,'60m')['deadline_epoch'],4600)
        with self.assertRaises(ValueError): make_budget(1000,'bad')
