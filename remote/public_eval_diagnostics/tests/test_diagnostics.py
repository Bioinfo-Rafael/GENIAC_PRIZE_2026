import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from data import DuplicateIndex, canonical, identity, stratified
from extraction import METHODS, select, summary
from run import analyze, digest, grade, load_saved, prepare_gold, read, rows_from, sha, version, write


def row(problem,subject='Algebra',level=1):
    return dict(problem=problem,id=identity(problem),dataset='prm800k',answer='2',subject=subject,level=level)


class ExtractionTests(unittest.TestCase):
    def test_multiple_nested_and_malformed_boxes(self):
        text=r'\boxed{\frac{1}{2}} then \fbox{3}'
        self.assertEqual(select(text,'first_box'),r'\frac{1}{2}')
        self.assertEqual(select(text,'last_box'),'3')
        text=r'\boxed{2} then \boxed{3'
        self.assertEqual(select(text,'first_box'),'2')
        self.assertIsNone(select(text,'last_box'))  # no fallback to earlier answer
        self.assertIsNone(select('reasoning 2 then 3','first_box'))

    def test_explicit_is_separate_and_conservative(self):
        text='Final answer is 2\nMore text \\boxed{3}'
        self.assertEqual(select(text,'explicit_final'),'2')
        self.assertEqual(select(text,'last_box'),'3')
        self.assertEqual(select('#### 4\nFinal answer: \\boxed{5}','explicit_final'),'5')
        self.assertIsNone(select('The answer is 2','explicit_final'))
        self.assertIsNone(select(r'\boxed{2}','explicit_final'))

    def test_prm_equivalence_uses_existing_math_verifier(self):
        r=row('Synthetic fraction question');r['answer']=r'\frac{1}{2}'
        gold,failures=prepare_gold([r])
        self.assertEqual(failures,{})
        text=r'\boxed{0.5} then \boxed{2}'
        first=select(text,'first_box');last=select(text,'last_box')
        self.assertTrue(grade(r,'\\boxed{'+first+'}',gold[r['id']])['correct'])
        self.assertFalse(grade(r,'\\boxed{'+last+'}',gold[r['id']])['correct'])

    def test_real_grader_failure_denominator_and_resume(self):
        dev=[dict(row('public synthetic '+str(i)),dataset='gsm8k') for i in range(3)]
        texts=[r'\boxed{2} then \boxed{3}',r'\boxed{2} then \boxed{', 'Final answer is 2']
        values={r['id']:dict(text=text,token_ids=[1]*10,finish_reason='stop',grade={'correct':i==2})
                for i,(r,text) in enumerate(zip(dev,texts))}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sources={'prm_train':[],'prm_test':[]}
            result=analyze(dev,{n:values for n in ('A','D','R12')},sources,root)
            a=result['models']['A']['groups']['all']
            self.assertEqual(a['first_box']['correct'],2)
            self.assertEqual(a['first_box']['n'],3)
            self.assertEqual(a['first_box']['accuracy'],2/3)
            self.assertEqual(a['last_box']['extraction_failure'],2)
            self.assertEqual(a['explicit_final']['correct'],1)
            self.assertEqual(a['transitions']['first_correct_last_wrong'],2)
            self.assertEqual(a['transitions']['first_correct_last_wrong_both_extractable'],1)
            self.assertEqual(analyze(dev,{n:values for n in ('A','D','R12')},sources,root),result)
            self.assertNotIn('oracle',a)


class PublicSetTests(unittest.TestCase):
    def test_normalization_and_near_duplicates(self):
        original=row('A long algebra problem with many separate words that asks a student to calculate the sum of twenty integers in a fixed arithmetic sequence today.')
        index=DuplicateIndex([original])
        self.assertEqual(index.reason(row(original['problem']+'  ')),'exact')
        self.assertEqual(index.reason(row(original['problem'].upper())),'case_whitespace')
        self.assertEqual(index.reason(row(original['problem']+' Please.')),'near_5gram_jaccard_0.85')
        self.assertIsNone(index.reason(row('Unrelated short problem: compute 27 * 45.')))

    def test_stratification_determinism_and_capacity(self):
        rows=[row(str(i),level=1 if i<3 else 2) for i in range(10)]
        a,q=stratified(rows,7,42);b,_=stratified(list(reversed(rows)),7,42)
        self.assertEqual(a,b);self.assertEqual(q,{'Algebra|1':2,'Algebra|2':5})
        self.assertEqual(len({r['id'] for r in a}),7)
        with self.assertRaises(ValueError):stratified(rows,11,42)

    def test_existing_nested_split_and_unknown_manifest(self):
        split={'dev':[row('dev')],'final':[row('final')]}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'split.json';write(p,dict(split=split,split_sha256=digest(split)))
            self.assertEqual(len(rows_from(p)),2)
            write(p,dict(split=split,split_sha256='bad'))
            with self.assertRaises(ValueError):rows_from(p)
            write(p,{'unrelated':[]})
            with self.assertRaises(ValueError):rows_from(p)

    def test_contract_and_response_alignment(self):
        dev=[dict(row('synthetic dev '+str(i)),dataset='gsm8k') for i in range(400)]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write(root/'data/dev.json',dev)
            c=dict(model_sha='a',dev_sha=digest(dev),grader_sha=sha(HERE.parent/'math_training/grading.py'),
                   packages={'math-verify':version('math-verify')},config={'seed':42})
            values={r['id']:[dict(text='2',token_ids=[1],finish_reason='stop',grade={'correct':True})] for r in dev}
            for n in ('A','D','R12'):
                write(root/f'evaluation/{n}/contract.json',dict(c,model_sha=n))
                write(root/f'evaluation/{n}/generations/b0_eval.json',values)
            self.assertEqual(len(load_saved(root)[0]),400)
            c['config']['seed']=43;write(root/'evaluation/D/contract.json',c)
            with self.assertRaises(ValueError):load_saved(root)


if __name__=='__main__':unittest.main()
