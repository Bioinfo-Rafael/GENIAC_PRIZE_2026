"""Code-only tests. No competition/approved dataset or model downloads/execution.
The structural fixtures are not mathematical tasks or training/evaluation data.
Run: python -m unittest discover -s colab_t4/experiments -v
Optional math-verify and nbformat enable parser/notebook validation tests.
"""
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import dev_eval as d


class AnswerTests(unittest.TestCase):
    def test_last_box_nested_and_incomplete(self):
        self.assertEqual(d.last_boxed(r'\boxed{2} then \boxed{\frac{1}{2}}'), r'\frac{1}{2}')
        self.assertIsNone(d.last_boxed(r'\boxed{2} then \boxed{\frac{1}{2}'))
        self.assertEqual(d.last_boxed(r'\boxed{\{1,2\}}'), r'\{1,2\}')

    def test_only_final_expression_counts(self):
        self.assertIsNone(d.final_candidate('intermediate 42 and 17')[0])
        self.assertEqual(d.final_candidate('intermediate 42\nAnswer: 17')[0], '17')
        self.assertEqual(d.final_candidate(r'\boxed{42} then \boxed{17}')[0], '17')
        self.assertIsNone(d.final_candidate('Answer: 42\n\\boxed{17')[0])

    def test_exact_numeric_parser(self):
        self.assertEqual(d.numeric_answer('1,234.50'), d.numeric_answer('1234.5'))
        self.assertEqual(d.numeric_answer(r'\frac{1}{2}'), d.numeric_answer('0.5'))
        self.assertEqual(d.numeric_answer('$-3/2$'), d.numeric_answer('-1.5'))
        for s in ('1,2', '42 or 17', '1/0', 'there were 42', '42 dollars', ''):
            self.assertIsNone(d.numeric_answer(s), s)

    def test_gsm_grade_no_intermediate_match(self):
        row = {'dataset': 'gsm8k'}
        self.assertFalse(d.grade(row, r'42 \boxed{17}', d.numeric_answer('42'))['correct'])
        self.assertTrue(d.grade(row, r'\boxed{0.5}', d.numeric_answer('1/2'))['correct'])
        result = d.grade(row, 'Answer: 1', None)
        self.assertTrue(result['gold_unparseable'])
        self.assertFalse(result['extraction_failed'])
        self.assertFalse(result['correct'])

    def test_c1_preserves_short_box_and_limits_long(self):
        text = r'\boxed{1}'
        self.assertEqual(d.c1_target(text, '1'), (text, text))
        result, head = d.c1_target('x' * 900, '1')
        self.assertEqual(len(head), 800)
        self.assertEqual(d.last_boxed(result), '1')

    @unittest.skipUnless(importlib.util.find_spec('math_verify'), 'optional math-verify not installed')
    def test_real_math_verify(self):
        gold = d.parse_math(r'\frac{1}{2}')
        self.assertTrue(gold)
        self.assertTrue(d.grade({'dataset': 'prm800k'}, r'\boxed{0.5}', gold)['correct'])
        self.assertFalse(d.grade({'dataset': 'prm800k'}, r'\boxed{0.3}', gold)['correct'])
        self.assertFalse(d.grade({'dataset': 'prm800k'}, r'\boxed{0.5} \boxed{0.3}', gold)['correct'])


class SplitTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(d.problem_hash('Ａ  B\nC'), d.problem_hash('A B C'))

    def test_split_is_deterministic_disjoint_and_excludes_history(self):
        # Opaque schema placeholders only, no actual mathematical problems.
        rows = [{'dataset': source, 'source_id': str(i), 'problem': f'{source}-placeholder-{i}',
                 'answer': 'fixture', 'subject': 'fixture'}
                for source, n in [('gsm8k', 240), ('prm800k', 650)] for i in range(n)]
        excluded = {d.problem_hash(rows[0]['problem'])}
        rows.append(dict(rows[10], dataset='prm800k', source_id='duplicate'))
        split = d.select_split(rows, excluded)
        self.assertEqual(split, d.select_split(list(reversed(rows)), excluded))
        dev, final = ({r['id'] for r in split[s]} for s in ('dev', 'final'))
        self.assertEqual(len(dev), 400)
        self.assertEqual(len(final), 400)
        self.assertFalse(dev & final or (dev | final) & excluded)
        self.assertEqual(len(split['quick_ids']), 128)
        self.assertTrue(set(split['quick_ids']) <= dev)
        self.assertEqual(sum(r['dataset'] == 'prm800k' for r in split['dev']), 300)

    def test_insufficient_pool_stops(self):
        with self.assertRaises(RuntimeError):
            d.select_split([], set())

    def test_missing_training_history_stops(self):
        with patch.object(d, 'require_colab'), tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                d.training_inventory([{'name': 'fixture', 'path': tmp + '/missing', 'expected_rows': 1}])

    def test_local_runtime_stops(self):
        with self.assertRaises(RuntimeError):
            d.require_colab()


class ResumeTests(unittest.TestCase):
    def test_different_signature_and_wrong_order_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'state.json'
            state = d.resume_state(p, {'model': 'a'}, ['a', 'b'])
            state['rows'] = [{'id': 'a'}]
            d.atomic_json(p, state)
            self.assertEqual(len(d.resume_state(p, {'model': 'a'}, ['a', 'b'])['rows']), 1)
            with self.assertRaises(RuntimeError):
                d.resume_state(p, {'model': 'b'}, ['a', 'b'])
            with self.assertRaises(RuntimeError):
                d.resume_state(p, {'model': 'a'}, ['b', 'a'])

    def test_chunk_resume_and_summary_without_answers(self):
        rows = [{'id': 'a', 'dataset': 'gsm8k', 'problem': 'opaque-schema-fixture'},
                {'id': 'b', 'dataset': 'prm800k', 'problem': 'opaque-schema-fixture'}]
        calls = []
        class StubLLM:
            def chat(self, batch, params, use_tqdm):
                calls.append(len(batch))
                if len(calls) == 2:
                    raise RuntimeError('simulated interruption')
                return [types.SimpleNamespace(prompt_token_ids=[1], outputs=[
                    types.SimpleNamespace(text='opaque-response', token_ids=[2, 3], finish_reason='length')])
                        for _ in batch]
        fake_grade = {'correct': False, 'extraction_failed': True, 'answer_method': 'missing',
                      'grader_error': False, 'gold_unparseable': False}
        with tempfile.TemporaryDirectory() as tmp, patch.object(d, 'require_colab'), \
             patch.dict('sys.modules', {'vllm': types.SimpleNamespace(SamplingParams=lambda **kw: kw)}), \
             patch.object(d, 'grade', return_value=fake_grade):
            with self.assertRaises(RuntimeError):
                d.evaluate_rows(StubLLM(), rows, {'a': 1, 'b': 1}, {'s': 1}, tmp, 10, 1)
            self.assertEqual(len(d.read_json(Path(tmp) / 'predictions_private.json')['rows']), 1)
            summary = d.evaluate_rows(StubLLM(), rows, {'a': 1, 'b': 1}, {'s': 1}, tmp, 10, 1)
            self.assertEqual(summary['overall']['n'], 2)
            self.assertEqual(len(calls), 3)
            self.assertNotIn('opaque-response', json.dumps(summary))
            self.assertNotIn('opaque-schema-fixture', json.dumps(summary))
            self.assertEqual(summary['overall']['length_stop_rate'], 1)
            d.evaluate_rows(StubLLM(), rows, {'a': 1, 'b': 1}, {'s': 1}, tmp, 10, 1)
            self.assertEqual(len(calls), 3)  # completed run never generates again

    def test_wilson_bounds(self):
        lo, hi = d.wilson(0, 32)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, 0)
        lo, hi = d.wilson(32, 32)
        self.assertLess(lo, 1)
        self.assertAlmostEqual(hi, 1)


class NotebookTests(unittest.TestCase):
    def test_plain_cells_compile_and_embedded_helper_matches(self):
        here = Path(__file__).parent
        nb = json.loads((here / 'evaluate_dev.ipynb').read_text())
        source = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                compile(''.join(c['source']), c['id'], 'exec')
                self.assertFalse(c['outputs'])
                self.assertIsNone(c['execution_count'])
        self.assertIn((here / 'dev_eval.py').read_text().strip(), source)
        self.assertNotIn('compe_screening_eval', source)
        self.assertNotIn('base64', source)
        if importlib.util.find_spec('nbformat'):
            import nbformat
            nbformat.validate(nbformat.from_dict(nb))

    def test_template_exactly_matches_distribution(self):
        here = Path(__file__).parent
        nb = json.loads((here / 'evaluate_dev.ipynb').read_text())
        cell = next(c for c in nb['cells'] if ''.join(c['source']).startswith('# [inference-template]'))
        scope = {'hashlib': __import__('hashlib')}
        exec(''.join(cell['source']), scope)
        config = json.loads((here.parent / 'inference_config.json').read_text())
        self.assertEqual(scope['CHAT_TEMPLATE'], config['chat_template']['jinja'])


if __name__ == '__main__':
    unittest.main()
