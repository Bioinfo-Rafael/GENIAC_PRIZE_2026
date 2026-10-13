"""Code/schema tests only; fixtures are not mathematical training examples."""
import json
import unittest
from pathlib import Path
import stage2_recipe as r


class TokenizerStub:
    eos_token_id = 9
    def apply_chat_template(self, messages, tokenize):
        return [1]*int(messages[0]['content'].split(':')[-1])


class RecipeTests(unittest.TestCase):
    def record(self, name, length, source='prm800k'):
        return dict(problem=name+':'+str(length), solution='opaque fixture', answer='opaque', dataset=source)

    def test_holdout_duplicates_and_length_filter_before_sampling(self):
        blocked = self.record('blocked', 3)
        kept = self.record('kept', 4)
        long = self.record('long', 9)
        gsm = self.record('gsm', 4, 'gsm8k')
        records = [blocked, kept, dict(kept), long, gsm]
        raw, tokens, stats = r.build_recipe(records, TokenizerStub(),
            [r.question_hash(blocked['problem'])], {'prm800k':1,'gsm8k':1}, max_length=5)
        self.assertEqual({x['problem'] for x in raw}, {kept['problem'],gsm['problem']})
        self.assertTrue(all(x['input_ids'][-1] == 9 and len(x['input_ids']) == 5 for x in tokens))
        self.assertEqual(stats['prm800k_over_length_excluded'],1)
        self.assertEqual(stats['prm800k_duplicate_excluded'],1)
        self.assertEqual(stats['prm800k_heldout_excluded'],1)
        self.assertEqual((raw,tokens,stats),r.build_recipe(list(reversed(records)), TokenizerStub(),
            [r.question_hash(blocked['problem'])], {'prm800k':1,'gsm8k':1},max_length=5))

    def test_insufficient_pool_does_not_silently_use_smaller_sample(self):
        with self.assertRaises(RuntimeError):
            r.build_recipe([self.record('x',3)],TokenizerStub(),[],{'prm800k':2},max_length=5)

    def test_no_solution_truncation(self):
        s='opaque'*200
        result=r.full_solution(s,'placeholder')
        self.assertTrue(result.startswith(s))
        self.assertEqual(r.boxed_answer(result),'placeholder')
        self.assertEqual(r.full_solution(r'opaque \boxed{placeholder}','placeholder'),r'opaque \boxed{placeholder}')

    def test_notebook_and_frozen_reference(self):
        here=Path(__file__).parent
        ref=json.loads((here/'run02_reference.json').read_text())
        self.assertEqual(len(ref['hashes']),800)
        self.assertEqual(len(set(ref['hashes'])),800)
        nb=json.loads((here/'train_stage2.ipynb').read_text())
        codes=[]
        for c in nb['cells']:
            if c['cell_type']=='code':
                source=''.join(c['source'])
                compile(source,c['id'],'exec')
                codes.append(source)
                self.assertFalse(c['outputs'])
        self.assertIn((here/'stage2_recipe.py').read_text().strip(),'\n'.join(codes))
        import importlib.util
        if importlib.util.find_spec('nbformat'):
            import nbformat
            nbformat.validate(nbformat.from_dict(nb))


if __name__=='__main__':
    unittest.main()
