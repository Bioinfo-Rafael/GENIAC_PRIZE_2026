import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from helpers import (weight,prefix_ids,validate_order,can_start_batch,candidate_name,
                     count_plan,write,read,sha)


def row(i):
    ids=[1,3]+[4]*(2+i%3)+[0]
    return dict(id=str(i),dataset='prm800k',chosen_tokens=dict(
        input_ids=ids,labels=[-100,-100]+ids[2:]))


class ContractTests(unittest.TestCase):
    def test_weights_and_prefix(self):
        self.assertEqual([weight(v) for v in ([False,False],[False,True],[True,True])],[1,.5,0])
        with self.assertRaises(ValueError): weight([0,1])
        self.assertEqual(prefix_ids(row(0)['chosen_tokens']),[1,3])
        bad=row(0)['chosen_tokens']; bad['labels'][3]=-100
        with self.assertRaises(ValueError): prefix_ids(bad)

    def test_ancestor_overlap_and_unique_order(self):
        full=[row(i) for i in range(3128)]
        ids=[r['id'] for r in full]
        self.assertEqual(validate_order(full,ids,[dict(id='heldout')],128),full[3000:])
        with self.assertRaises(ValueError): validate_order(full,ids,[dict(id='3010')],128)
        with self.assertRaises(ValueError): validate_order(full,ids[::-1],[],128)
        with self.assertRaises(ValueError): validate_order(full,ids,[],129)
        full[-1]=full[0]
        with self.assertRaises(ValueError): validate_order(full,[r['id'] for r in full],[],128)

    def test_time_reserve(self):
        self.assertTrue(can_start_batch(0,200,[100]))
        self.assertFalse(can_start_batch(50,200,[100]))
        self.assertFalse(can_start_batch(190,200,[]))


class CPUTrainingTests(unittest.TestCase):
    """Exercise the actual trainer with torch backward/checkpoints and real grading.

    Only GPU transport, policy generation and pretrained weights are substituted.
    This is not a CUDA/Unsloth/vLLM compatibility test.
    """
    def exercise(self,mode='mixed',interrupt_after=None,constant=False,questions=48):
        import torch
        import worker
        from osw_train import train_context
        directory=tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        root=Path(directory.name)
        rows=[row(i) for i in range(questions)]
        plan=count_plan(rows,[(candidate_name(n),n) for n in range(16,questions+1,16)])
        write(root/'chain/stage2/pairs.json',rows)
        write(root/'chain/stage2/schedule.json',plan)
        write(root/'budget.json',dict(train_deadline_epoch=time.time()+3600))
        write(root/'request.json',dict(test=True))
        write(root/'chain/recipe.json',dict(learning_rate=1e-6))
        write(root/'data/gold.json',{r['id']:dict(id=r['id'],dataset='prm800k',answer='1') for r in rows})
        for n in (16,32,48): write(root/candidate_name(n)/'experiment.json',dict(realized=dict(questions=3000+n)))

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.value=torch.nn.Parameter(torch.tensor(.2)); self.calls=0
            def gradient_checkpointing_disable(self): pass
            def gradient_checkpointing_enable(self): pass
            def generate(self,input_ids,**kwargs):
                self.calls+=1
                if interrupt_after and self.calls>interrupt_after: worker.STOP=True
                outcomes=[17,17] if mode=='all_correct' else [17,18]
                return torch.stack([torch.cat((input_ids[0],torch.tensor([v,0]))) for v in outcomes])
            def save_pretrained(self,path,**kwargs):
                path.mkdir(parents=True,exist_ok=True)
                from safetensors.torch import save_file
                save_file(self.state_dict(),path/'adapter_model.safetensors')
                write(path/'adapter_config.json',{'r':8,'lora_alpha':16})

        class Tokenizer:
            eos_token_id=0
            pad_token_id=0
            def decode(self,ids,**kwargs): return r'\boxed{1}' if ids[0]==17 else r'\boxed{2}'
            def save_pretrained(self,path): write(path/'tokenizer_config.json',dict(eos_token_id=0))

        model=Tiny(); initial=model.value.detach().clone(); seen=[]
        def logps(m,t,rs,dtype):
            seen.extend(rs)
            n=sum(v!=-100 for v in rs[0]['labels'][1:])
            return -((m.value-1.)**2).reshape(1)*n,torch.tensor([n])
        real_tensor=torch.tensor
        def cpu_tensor(*args,**kwargs):
            if kwargs.get('device')=='cuda': kwargs['device']='cpu'
            return real_tensor(*args,**kwargs)
        ctx=dict(config=dict(learning_rate=1e-6),inputs=str(root/'tokenizer'),signature='test')
        if constant: ctx['config']['lr_policy']=dict(kind='constant',total_steps=1)
        worker.STOP=False; self.addCleanup(setattr,worker,'STOP',False)
        with patch.object(worker,'load_train_model',return_value=model), \
             patch.object(worker,'tokenizer_for',return_value=Tokenizer()), \
             patch.object(worker,'logps',side_effect=logps), \
             patch.object(torch,'tensor',side_effect=cpu_tensor), \
             patch.object(torch.cuda,'synchronize'), \
             patch.object(torch.cuda,'max_memory_allocated',return_value=0), \
             patch.object(torch.cuda,'get_rng_state_all',return_value=[]):
            if mode=='all_correct':
                with self.assertRaisesRegex(RuntimeError,'No nonzero-LR update'): train_context(root,ctx)
            else: train_context(root,ctx)
        return root,model,initial,seen

    def test_single_batch_constant_lr_really_changes_weights(self):
        import torch
        root,model,initial,seen=self.exercise(constant=True,questions=16)
        self.assertGreater(model.value.item(),initial.item())
        self.assertEqual(len(seen),16)
        self.assertEqual(read(root/'selected.json')['consumed_questions'],16)
        state=torch.load(root/'chain/stage2/training/rft/checkpoint-00001/state.pt',weights_only=False)
        self.assertEqual(state['nonzero_lr_updates'],1)
        record=json.loads((root/'chain/stage2/training/rft/train.jsonl').read_text())
        self.assertEqual(record['learning_rate'],1e-6)

    def test_constant_lr_zero_reward_weight_does_not_emit_parent(self):
        root,model,initial,seen=self.exercise(mode='all_correct',constant=True,questions=16)
        self.assertEqual(model.value.item(),initial.item())
        self.assertFalse((root/'selected.json').exists())

    def test_weighted_teacher_update_and_checkpoint(self):
        import torch
        root,model,initial,seen=self.exercise()
        self.assertGreater(model.value.item(),initial.item())
        self.assertEqual(len(seen),48)
        self.assertTrue(all(set(r['input_ids'])<={0,1,3,4} for r in seen))
        selected=read(root/'selected.json')
        self.assertEqual(selected['consumed_questions'],48)
        state=torch.load(root/'chain/stage2/training/rft/checkpoint-00003/state.pt',weights_only=False)
        self.assertEqual(state['step'],3)
        self.assertEqual(state['nonzero_lr_updates'],2)
        self.assertEqual(state['scheduler']['last_epoch'],3)
        self.assertTrue(state['optimizer']['state'])
        self.assertEqual(read(root/candidate_name(48)/'training_complete.json')['stop_reason'],'pilot_maximum')

    def test_zero_weight_skips_optimizer_and_weight_decay(self):
        import torch
        root,model,initial,seen=self.exercise(mode='all_correct')
        self.assertTrue(torch.equal(model.value.detach(),initial))
        self.assertEqual(seen,[])
        state=torch.load(root/'chain/stage2/training/rft/checkpoint-00003/state.pt',weights_only=False)
        self.assertEqual(state['updates'],0)
        self.assertEqual(state['optimizer']['state'],{})
        self.assertEqual(state['scheduler']['last_epoch'],3)
        self.assertFalse((root/'selected.json').exists())

    def test_interrupted_rollouts_discard_whole_uncommitted_batch(self):
        root,model,initial,seen=self.exercise(interrupt_after=32)
        self.assertEqual(len(seen),32)
        self.assertEqual(read(root/'selected.json')['consumed_questions'],32)
        self.assertFalse((root/'chain/stage2/training/rft/checkpoint-00003').exists())
        self.assertFalse(read(root/'chain/stage2/training/rft/discarded_batch.json')['teacher_update_applied'])
        self.assertEqual(read(root/candidate_name(32)/'training_complete.json')['stop_reason'],'submission_time_reserve')


class CoordinatorTests(unittest.TestCase):
    def runner(self):
        import importlib.util
        from types import SimpleNamespace
        spec=importlib.util.spec_from_file_location('osw_runner_test',Path(__file__).resolve().parents[1]/'run.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        directory=tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        root=Path(directory.name)
        write(root/'budget.json',dict(started_epoch=time.time(),deadline_epoch=time.time()+3600))
        import signal
        handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
        runner=module.Runner(SimpleNamespace(root=root))
        for sig,handler in handlers.items(): self.addCleanup(signal.signal,sig,handler)
        return module,root,runner

    def test_child_timeout_does_not_mark_submission_complete(self):
        module,root,runner=self.runner()
        (root/'phases.py').write_text('import time\ntime.sleep(30)\n')
        start=time.time()
        with patch.object(module,'HERE',root):
            with self.assertRaisesRegex(RuntimeError,'deadline'):
                runner.phase('submit',sys.executable,time.time()+.1)
        self.assertLess(time.time()-start,7)
        self.assertIsNone(runner.child)
        self.assertFalse(read(root/'suite_report.json')['submission_ready'])
        self.assertFalse((root/'receipts/submit.json').exists())

    def test_receipt_tampering_is_detected(self):
        module,root,runner=self.runner()
        write(root/'data/complete.json',dict(valid=True))
        write(root/'receipts/prepare.json',dict(files={'data/complete.json':sha(root/'data/complete.json')}))
        runner.phase('prepare',sys.executable,time.time()+20)
        write(root/'data/complete.json',dict(valid=False))
        with self.assertRaisesRegex(ValueError,'Artifact changed'):
            runner.phase('prepare',sys.executable,time.time()+20)

    def test_submission_can_finish_after_deadline(self):
        module,root,runner=self.runner()
        runner.budget['finish_submission']=True
        name=candidate_name(16)
        write(root/'selected.json',dict(name=name))
        for f in ('submission.json','format_check.json','hashes.json'):
            write(root/name/'submission'/f,dict(test=True))
        (root/'phases.py').write_text('import time\ntime.sleep(.3)\n')
        with patch.object(module,'HERE',root):
            runner.phase('submit',sys.executable,time.time()+.1)
        self.assertTrue((root/'receipts/submit.json').exists())

    def test_submission_extension_does_not_extend_training(self):
        module,root,runner=self.runner()
        runner.budget['finish_submission']=True
        (root/'phases.py').write_text('import time\ntime.sleep(30)\n')
        with patch.object(module,'HERE',root):
            with self.assertRaisesRegex(RuntimeError,'deadline'):
                runner.phase('train',sys.executable,time.time()+.1)
        self.assertFalse((root/'receipts/train.json').exists())

    def test_zip_exists_before_optional_evaluation(self):
        import zipfile
        module,root,runner=self.runner()
        name=candidate_name(32)
        write(root/'selected.json',dict(name=name))
        for f in ('submission.json','format_check.json','hashes.json','manifest.json'):
            write(root/name/'submission'/f,dict(test=True))
        write(root/name/'training_complete.json',dict(consumed_questions=32))
        runner.package()
        report=read(root/'suite_report.json')
        self.assertTrue(report['submission_ready'])
        with zipfile.ZipFile(report['zip']) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(z.read(name+'.json'),(root/name/'submission/submission.json').read_bytes())


if __name__=='__main__': unittest.main()
