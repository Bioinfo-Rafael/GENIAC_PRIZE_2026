"""CPU orchestration tests: simulated GPU children, real manifests/status/reports."""
import json
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import run
from core import write,sha,digest,read


@pytest.mark.parametrize('expire',[False,True])
def test_pipeline_and_budget_preserve_artifacts(tmp_path,monkeypatch,expire):
    code=tmp_path/'code';code.mkdir()
    (code/'worker.py').write_text('# simulated GPU worker')
    (code/'fetch_inputs.py').write_text('# simulated download')
    workspace=tmp_path/'work';inputs=workspace/'inputs'
    inputs.mkdir(parents=True)
    (inputs/'stage2').mkdir()
    (inputs/'stage2/adapter_model.safetensors').write_bytes(b'fake weights')
    train=[{'problem':f'train {i}','dataset':'prm800k' if i<4500 else 'gsm8k','answer':'1'} for i in range(6000)]
    (inputs/'train.jsonl').write_text('\n'.join(json.dumps(r) for r in train))
    from core import question_id
    held=[{'problem':f'held {i}','dataset':'prm800k' if i%4 else 'gsm8k',
           'id':question_id(f'held {i}'),'answer':'1'} for i in range(800)]
    split={'dev':held[:400],'final':held[400:],'quick_ids':[r['id'] for r in held[:128]]}
    write(inputs/'split.json',{'split':split,'split_sha256':digest(split)})
    write(code/'input_manifest.json',{'files':[{'path':str(p.relative_to(inputs)),'sha256':sha(p)}
          for p in inputs.rglob('*') if p.is_file()]})
    cfg=read(Path(run.__file__).parent/'configs/ada_dpo.json')
    cfg.update(prompts=128,stage2_sha256=sha(inputs/'stage2/adapter_model.safetensors'))
    write(code/'config.json',cfg)
    monkeypatch.setattr(run,'HERE',code)
    monkeypatch.setattr(run.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**k:'{}')
    monkeypatch.setattr(run.os,'killpg',lambda *a:None)
    phases=[]
    class Child:
        pid=99999
        def __init__(self,cmd,**kw):
            self.timeout=False;self.done=False
            if len(cmd)>3 and cmd[2].endswith('worker.py'):
                phase=cmd[3];phases.append(phase)
                ctx=read(cmd[cmd.index('--context')+1]);root=Path(ctx['run'])
                branch=cmd[cmd.index('--candidate')+1]
                purpose=cmd[cmd.index('--purpose')+1];count=int(cmd[cmd.index('--count')+1])
                if phase=='generate':
                    rows=read(root/('train_rows.json' if purpose=='train' else 'eval_rows.json'))[:count]
                    outputs={r['id']:[{'text':'1','token_ids':[1],'finish_reason':'stop',
                        'grade':{'correct':True,'extraction_failed':False,'grader_error':False,'gold_unparseable':False}}] for r in rows}
                    write(root/f'generations/{branch}_{purpose}.json',outputs)
                    write(root/f'timings/{branch}_{purpose}.json',[{'questions':count,'seconds':1}])
                elif phase=='pairs':write(root/'pair_summary.json',{'pairs':64})
                elif phase=='bench':
                    self.timeout=expire
                    write(root/f'timings/bench_{branch}.json',{'updates':8,'seconds_per_update':1})
        def wait(self,timeout=None):
            if self.timeout:
                self.timeout=False
                raise subprocess.TimeoutExpired('fake GPU phase',timeout)
            self.done=True;return 0
        def poll(self):return 0 if self.done else None
    monkeypatch.setattr(run.subprocess,'Popen',Child)
    monkeypatch.setattr(sys,'argv',['run.py','--workspace',str(workspace),'--run-id','test',
        '--config',str(code/'config.json'),'--eval-python',sys.executable])
    run.main()
    root=workspace/'runs/test'
    status=read(root/'status.json')['status']
    assert status==('budget_exhausted' if expire else 'dev_comparison_complete')
    assert (root/'context.json').exists() and (root/'report.json').exists()
    if not expire:
        assert phases.count('train')==2
        assert read(root/'report.json')['candidates']['dpo']['n']==400
