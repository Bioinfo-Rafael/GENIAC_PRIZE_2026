import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common import arithmetic,validate_generated,centered,choose,read,write,sha
from pipeline import Experiments,PhaseTimeout


class Contracts(unittest.TestCase):
    def test_exact_arithmetic_and_reject_unsafe(self):
        self.assertEqual(arithmetic('0.1+0.2'),arithmetic('0.3'))
        for text in ['__import__("os")','2**999999','a+1','1/0','True','[1]']:
            with self.assertRaises((ValueError,SyntaxError,ZeroDivisionError)):arithmetic(text)

    def test_synthesis_rejects_wrong_and_extra_reasoning(self):
        item=dict(lhs='6/2',rhs='3',form='json')
        self.assertIsNone(validate_generated(item,'{"expression":"6/2","value":"3"}'))
        self.assertEqual(validate_generated(item,'{"expression":"6/2","value":"4"}'),'wrong_value')
        self.assertIsNotNone(validate_generated(item,'{"expression":"3+0","value":"3"}'))
        item['form']='boxed'
        self.assertIsNone(validate_generated(item,r'\boxed{3}'))
        self.assertIsNotNone(validate_generated(item,r'\boxed{3} plus incorrect proof'))
        item['rhs']='4'
        self.assertEqual(validate_generated(item,r'\boxed{3}'),'source_equation_wrong')

    def test_centering_no_std(self):
        self.assertEqual(centered([1,0,0,0]),[.75,-.25,-.25,-.25])
        self.assertEqual(centered([1]*8),[0]*8)

    def test_selection_ignores_length_and_ties_prefer_A(self):
        trials={n:dict(status='complete',metrics=dict(correct=200,repetition_rate=i)) for i,n in enumerate(['A','E2','E3-16'])}
        self.assertEqual(choose(trials),'A')
        trials['E3-16']['metrics']['correct']=201
        self.assertEqual(choose(trials),'E3-16')
        trials['E3-16']['status']='running'
        self.assertEqual(choose(trials),'A')

    def test_fixed_length_gradient(self):
        import torch
        from online_worker import policy_objective
        values=torch.tensor([-3.,-3.],requires_grad=True)
        counts=torch.tensor([1.,10.])
        policy_objective(values,counts,[.5,.5],2,fixed_length=20).backward()
        self.assertTrue(torch.equal(values.grad,torch.tensor([-.0125,-.0125])))
        legacy=torch.tensor([-3.,-3.],requires_grad=True)
        policy_objective(legacy,counts,[.5,.5],2).backward()
        self.assertNotEqual(legacy.grad[0],legacy.grad[1])

    def test_final_time_reserved_and_no_command_after_deadline(self):
        s=object.__new__(Experiments);s.deadline=11000;s.bucket='rl';s.summary={'spent':{'rl':10}}
        with patch('pipeline.time.time',return_value=8500):
            self.assertEqual(s.limit(),8600)
            s.bucket='final';self.assertEqual(s.limit(),10900)
            s.deadline=8490
            with self.assertRaises(PhaseTimeout):s.command(['false'],'never_started')

    def test_freeze_selection_and_confirm_only_one(self):
        # Exercise finish with mocked model execution, including failure to confirm.
        with tempfile.TemporaryDirectory() as d:
            s=object.__new__(Experiments);s.root=Path(d);s.summary={'evaluations':{'dev':{
                'A':dict(status='complete',metrics={'correct':100}),
                'E2':dict(status='complete',metrics={'correct':110},adapter='/E2'),
                'E3-16':dict(status='complete',metrics={'correct':109},adapter='/E3')}},'failures':[]}
            s.save=lambda:None;s.attempt=lambda n,fn:fn()
            calls=[]
            def evaluate(name,adapter,split,count):
                calls.append((name,split))
                if split=='independent':
                    self.assertTrue((s.root/'selection.json').exists())
                    return dict(comparison={'paired_bootstrap_95':[-.01,.05]})
            s.evaluate=evaluate
            fallback=s.root/'fallback_submission';fallback.mkdir()
            write(fallback/'submission.json',{})
            write(fallback/'complete.json',{'sha256':sha(fallback/'submission.json')})
            for n in ('manifest.json','hashes.json'):write(fallback/n,{})
            with patch('submit.check',return_value={}):s.finish()
            self.assertEqual(read(s.root/'selection.json')['candidate'],'E2')
            self.assertEqual(read(s.root/'adoption.json')['model'],'A')
            self.assertNotIn(('E3-16','independent'),calls)
            self.assertTrue((s.root/'submission/submission.json').exists())

    def test_timeout_preserves_log_and_records_failure(self):
        import time
        with tempfile.TemporaryDirectory() as d:
            s=object.__new__(Experiments);s.root=Path(d);s.bucket='rl'
            s.deadline=time.time()+120;s.summary={'spent':{}};s.active=None;s.save=lambda:None
            # command reserves 30 seconds for termination; allow this child 0.1 seconds.
            s.limit=lambda:time.time()+30.1
            with patch('pipeline.subprocess.check_output',return_value='123\n'):
                with self.assertRaises(PhaseTimeout):
                    s.command([sys.executable,'-c','import time; time.sleep(10)'],'timeout_probe')
            record=json.loads((s.root/'phase_times.jsonl').read_text().strip())
            self.assertIn('Deadline reached',record['error'])
            self.assertGreater(record['seconds'],0)
            self.assertEqual(record['peak_device_vram_mib'],123)
            self.assertIsNone(s.active)

    def test_reuse_refuses_wrong_model(self):
        from submit import reuse
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'old';source.mkdir();model=root/'model';model.mkdir()
            write(model/'artifacts.json',{})
            write(source/'submission.json',{});write(source/'complete.json',{'sha256':sha(source/'submission.json')})
            write(source/'manifest.json',dict(adapter_sha256='wrong',model_artifacts_sha256=sha(model/'artifacts.json')))
            self.assertFalse(reuse(source,root/'new',model,'run03'))
            self.assertFalse((root/'new').exists())

if __name__=='__main__':unittest.main()
