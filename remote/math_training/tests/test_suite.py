"""Real CPU preparation/selection with simulated GPU subprocesses."""
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import suite
import suite_utils
from core import digest, write, read, sha, question_id, pairs_from_outputs


def output(index=0,correct=True):
    return {'index':index,'text':str(index),'token_ids':[1 if correct else 2],
        'finish_reason':'stop','grade':{'correct':correct,'extraction_failed':False,
        'grader_error':False,'gold_unparseable':False}}


@pytest.mark.parametrize('mutation',['none','config','code','packages','rows','incomplete'])
def test_source_cache_checks(tmp_path,monkeypatch,mutation):
    source=tmp_path/'old'
    code=source/'code';code.mkdir(parents=True)
    names=['worker.py','core.py','grading.py','chat_template.jinja','input_manifest.json']
    for name in names:(code/name).write_text(name)
    hashes={name:sha(code/name) for name in names}
    monkeypatch.setattr(suite_utils,'LEGACY_CODE',{k:v for k,v in hashes.items() if k!='input_manifest.json'})
    cfg={k:1 for k in suite_utils.GENERATION_KEYS};cfg['generations']=4
    rows=[{'id':'a'}];dev=[{'id':'b'}]
    packages={'train':{'torch':'2.8'},'eval':{'vllm':'0.10'}}
    write(source/'context.json',{'config':cfg,'code':hashes})
    write(source/'train_rows.json',rows);write(source/'eval_rows.json',dev)
    for label,value in packages.items():write(source/f'packages_{label}.json',value)
    write(source/'generations/b0_train.json',{'a':[output(i) for i in range(4)]})
    write(source/'generations/b0_eval.json',{'b':[output()]})
    if mutation=='config':cfg=dict(cfg,temperature=9)
    if mutation=='code':(code/'worker.py').write_text('modified')
    if mutation=='packages':packages['eval']={'vllm':'different'}
    if mutation=='rows':rows=[{'id':'changed'}]
    if mutation=='incomplete':write(source/'generations/b0_train.json',{})
    if mutation=='none':
        assert suite_utils.validate_source(source,cfg,hashes,packages,rows,dev)
    else:
        with pytest.raises(ValueError):suite_utils.validate_source(source,cfg,hashes,packages,rows,dev)


@pytest.mark.parametrize('mutation',['none','settings','missing','environment'])
def test_recovery_pool_is_validated_before_reuse(tmp_path,monkeypatch,mutation):
    source=tmp_path/'old';code=source/'code';code.mkdir(parents=True)
    names=['worker.py','core.py','grading.py','chat_template.jinja','input_manifest.json']
    for name in names:(code/name).write_text(name)
    hashes={name:sha(code/name) for name in names}
    monkeypatch.setattr(suite_utils,'RECOVERABLE_WORKER',hashes['worker.py'])
    cfg={k:1 for k in suite_utils.GENERATION_KEYS};cfg['generations']=4
    write(source/'suite_context.json',{'config':cfg,'code':hashes})
    packages={'train':{},'eval':{}}
    for k,v in packages.items():write(source/f'packages_{k}.json',v)
    rows=[{'id':str(i)} for i in range(2048)];dev=[{'id':'dev'}]
    baseline={'dev':[output()]}
    pool=source/'pools/2048'
    write(pool/'context.json',{'config':cfg})
    write(pool/'train_rows.json',rows);write(pool/'eval_rows.json',dev)
    gen={r['id']:[output(i) for i in range(4)] for r in rows}
    if mutation=='missing':del gen['0']
    write(pool/'generations/b0_train.json',gen)
    write(source/'shared/generations/b0_eval.json',baseline)
    write(source/'audit/integrity.json',{'identical':True})
    if mutation=='settings':write(pool/'context.json',{'config':dict(cfg,temperature=9)})
    if mutation=='environment':packages['eval']={'vllm':'changed'}
    if mutation=='none':
        assert len(suite_utils.validate_recovery(source,cfg,hashes,packages,rows,dev,baseline))==2048
    else:
        with pytest.raises(ValueError):suite_utils.validate_recovery(source,cfg,hashes,packages,rows,dev,baseline)


def setup_suite(tmp_path,monkeypatch,improve=True,fail=None,audit_variability=False):
    workspace=tmp_path/'work';inputs=workspace/'inputs';inputs.mkdir(parents=True)
    train=[{'problem':f'train {i}','dataset':'prm800k' if i<4500 else 'gsm8k','answer':'1'} for i in range(6000)]
    (inputs/'train.jsonl').write_text('\n'.join(json.dumps(r) for r in train))
    held=[{'problem':f'held {i}','id':question_id(f'held {i}'),
           'dataset':'gsm8k' if i%4==0 else 'prm800k','answer':'1'} for i in range(800)]
    split={'dev':held[:400],'final':held[400:],'quick_ids':[r['id'] for r in held[:128]]}
    write(inputs/'split.json',{'split':split,'split_sha256':digest(split)})
    code=tmp_path/'code';code.mkdir()
    write(code/'configs/ada_dpo.json',read(Path(suite.__file__).parent/'configs/ada_dpo.json'))
    write(code/'input_manifest.json',{'files':[{'path':p.name,'sha256':sha(p)} for p in inputs.iterdir()]})
    (code/'worker.py').write_text('# simulated worker')
    monkeypatch.setattr(suite,'HERE',code)
    monkeypatch.setattr(suite.subprocess,'check_output',lambda *a,**k:'{}')
    monkeypatch.setattr(suite.shutil,'disk_usage',lambda _:SimpleNamespace(free=200*2**30))
    calls=[]
    def model(path):
        path.mkdir(parents=True,exist_ok=True)
        (path/'weights.safetensors').write_bytes(b'weights')
        write(path/'artifacts.json',{'weights.safetensors':sha(path/'weights.safetensors')})
    def command(self,cmd,name):
        calls.append(name)
        if '--context' not in cmd:return 1
        ctx=read(cmd[cmd.index('--context')+1]);root=Path(ctx['run']);cfg=ctx['config']
        phase=cmd[3];branch=cmd[cmd.index('--candidate')+1]
        purpose=cmd[cmd.index('--purpose')+1];count=int(cmd[cmd.index('--count')+1])
        if fail and name==fail:raise suite.PhaseFailure('simulated candidate failure')
        if phase=='merge':
            if branch=='b0':model(root/'models/stage1')
            model(root/'models'/branch)
        elif phase=='generate':
            rows=read(root/{'train':'train_rows.json','eval':'eval_rows.json','final':'final_rows.json'}[purpose])[:count]
            if purpose=='train':
                outputs={r['id']:[output(i,correct=(i<2 or j%3==0)) for i in range(4)] for j,r in enumerate(rows)}
            else:
                threshold=7 if improve and branch!='b0' and root.name!='audit' else 6
                # Final is deliberately worse for the candidate: selection must remain frozen.
                if purpose=='final' and branch!='b0':threshold=5
                outputs={r['id']:[output(correct=j%10<threshold)] for j,r in enumerate(rows)}
                if audit_variability and root.name=='audit':
                    outputs[rows[0]['id']]=[output(correct=False)]
            write(root/f'generations/{branch}_{purpose}.json',outputs)
            write(root/f'timings/{branch}_{purpose}.json',[{'questions':count,'seconds':count*.01}])
        elif phase=='pairs':
            rows=read(root/'train_rows.json');gen=read(root/'generations/b0_train.json')
            pairs,stats=pairs_from_outputs(rows,gen,cfg['seed'],cfg.get('data_mode','mixed'))
            for p in pairs:
                p['chosen_tokens']={'input_ids':[1,2,3],'labels':[-100,2,3]}
                if 'rejected' in p:p['rejected_tokens']=p['chosen_tokens']
            write(root/'pairs.json',pairs);write(root/'pair_summary.json',dict(stats,pairs=len(pairs)))
        elif phase=='audit':write(root/'audit.json',{'ok':True})
        elif phase=='audit_weights':write(root/'integrity.json',{'identical':True})
        elif phase=='reference':write(root/'reference.json',{'values':{}})
        elif phase=='bench':write(root/f'timings/bench_{branch}.json',{
            'microbatch':cfg['microbatch'],'seconds_per_update':1/cfg['microbatch'],
            'updates':math.ceil(len(read(root/'pairs.json'))/16)})
        elif phase=='train':model(root/f'training/{branch}/final')
        return 1
    monkeypatch.setattr(suite.Suite,'command',command)
    args=SimpleNamespace(workspace=workspace,run_id='test',source_run=None,budget_minutes=480,
        started_at=None,train_python=sys.executable,eval_python=sys.executable,resume=False)
    return suite.Suite(args),calls,args


@pytest.mark.parametrize('improve,fail',[(True,None),(False,None),(True,'e1_rft_mixed_train_rft_eval_128')])
def test_suite_runs_independent_trials_and_freezes_selection_before_final(tmp_path,monkeypatch,improve,fail):
    runner,calls,args=setup_suite(tmp_path,monkeypatch,improve,fail)
    runner.run()
    result=read(runner.summary_path)
    assert len([n for n in result['trials'] if n in {x[0] for x in suite.EXPERIMENTS}])==7
    assert result['baseline']['n']==400
    assert result['status']==('suite_complete' if improve else 'suite_complete_no_improvement')
    if improve:
        assert result['selection']['candidate'] is not None
        assert result['final']['candidate']['comparison']['delta_accuracy']<0
        assert not result['final']['candidate']['improvement_supported']
        assert result['recommendation']=='keep_b0_improvement_not_confirmed'
    else:
        assert result['selection']['candidate'] is None
        assert not any('_seed43' in n for n in result['trials'])
    if fail:
        assert result['trials']['e1_rft_mixed']['status']=='failed'
        assert result['trials']['e5_dpo_nll']['status']=='complete'
    # Selection survives resume: final results cannot cause another candidate to be selected.
    selection=read(runner.root/'selection.json')
    args.resume=True
    resumed=suite.Suite(args);resumed.run()
    assert read(runner.root/'selection.json')==selection


def test_budget_cutoff_stops_before_spawning_gpu_process(tmp_path,monkeypatch):
    runner,calls,args=setup_suite(tmp_path,monkeypatch)
    runner.cutoff=0
    # Restore the actual command method: the subprocess must never be launched.
    with pytest.raises(suite.PhaseTimeout):
        REAL_COMMAND(runner,['unused'],'late_phase')


REAL_COMMAND=suite.Suite.command


def test_identical_model_with_generation_variability_continues_and_records_it(tmp_path,monkeypatch):
    runner,calls,args=setup_suite(tmp_path,monkeypatch,improve=False,audit_variability=True)
    runner.run()
    report=read(runner.summary_path)
    assert report['status']=='suite_complete_no_improvement'
    assert len(report['inference_variability']['changed_outputs'])==1
    assert len(report['inference_variability']['changed_correctness'])==1
    assert 'inference executions' in report['uncertainty_scope']


def test_timed_out_child_is_terminated_and_status_remains_recoverable(tmp_path,monkeypatch):
    runner,calls,args=setup_suite(tmp_path,monkeypatch)
    runner.cutoff=time.time()+100
    signals=[]
    class Child:
        pid=98765
        waits=0
        def wait(self,timeout):
            self.waits+=1
            if self.waits==1:raise subprocess.TimeoutExpired('gpu',timeout)
            return 0
    child=Child()
    monkeypatch.setattr(suite.subprocess,'Popen',lambda *a,**k:child)
    monkeypatch.setattr(suite.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    with pytest.raises(suite.PhaseTimeout,match='Time limit'):
        REAL_COMMAND(runner,['fake_gpu'],'timed_out')
    assert signals and signals[0][0]==child.pid
    assert runner.active is None
    assert read(runner.summary_path)['active_phase']=='timed_out'


def test_resume_rejects_code_change(tmp_path,monkeypatch):
    runner,calls,args=setup_suite(tmp_path,monkeypatch)
    args.resume=True
    (suite.HERE/'worker.py').write_text('changed')
    with pytest.raises(ValueError,match='Immutable'):
        suite.Suite(args)


def test_recovery_reuses_completed_trials_and_retrains_only_failed_trial(tmp_path,monkeypatch):
    runner,calls,args=setup_suite(tmp_path,monkeypatch,improve=False,
                                 fail='e4_dpo_train_dpo_eval_128')
    runner.run()
    source=runner.root
    original_report=(source/'suite_report.json').read_bytes()
    original_command=suite.Suite.command
    def command(self,cmd,name):
        # Remove the injected failure for the recovery run.
        if name=='e4_dpo_train_dpo_eval_128':
            ctx=read(cmd[cmd.index('--context')+1]);root=Path(ctx['run'])
            dest=root/'training/dpo/final';dest.mkdir(parents=True,exist_ok=True)
            (dest/'weights.safetensors').write_bytes(b'weights')
            write(dest/'artifacts.json',{'weights.safetensors':sha(dest/'weights.safetensors')})
            calls.append(name)
            return 1
        return original_command(self,cmd,name)
    monkeypatch.setattr(suite.Suite,'command',command)
    monkeypatch.setattr(suite,'validate_recovery',lambda *a:read(source/'pools/2048/generations/b0_train.json'))
    args.run_id='recovery';args.source_suite=source
    calls.clear()
    recovered=suite.Suite(args);recovered.run()
    assert 'e4_dpo_train_dpo_eval_128' in calls
    assert 'e3_dft_positive_train_rft_eval_128' not in calls
    assert all(t['status']=='complete' for t in recovered.summary['trials'].values())
    assert recovered.summary['trials']['e3_dft_positive']['imported_from']
    assert (source/'suite_report.json').read_bytes()==original_report
