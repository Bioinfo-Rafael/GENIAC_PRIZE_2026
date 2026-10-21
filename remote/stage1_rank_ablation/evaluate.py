"""Only the pinned public dev400. Reuse the established generation and grading path."""
import argparse
import importlib.metadata
import shutil
from common import *
from worker import generate
from suite_utils import summarize, LEGACY_CODE


def contract(root,name):
    adapter=root/'inputs/stage2' if name=='A' else root/'conditions'/name/'final'
    return dict(model_sha=sha(adapter/'adapter_model.safetensors'),
        dev_sha=digest(read(root/'data/dev.json')),template_sha=sha(LEGACY/'chat_template.jinja'),
        worker_sha=sha(LEGACY/'worker.py'),grader_sha=sha(LEGACY/'grading.py'),
        packages={k:importlib.metadata.version(k) for k in ['vllm','torch','transformers','math-verify']},
        config=dict(base_model=BASE,base_revision=REVISION,seed=42,generations=1,
                    temperature=0,top_p=1,max_tokens=2048,context=4096,eval_chunk=32,vllm_memory=.8))



def reusable_outputs(reference,c,rows):
    if not reference:return None
    if (reference/'contract.json').exists() and read(reference/'contract.json')==c:
        file=reference/'generations/b0_eval.json'
    elif (reference/'context.json').exists():
        # Recognize only the archived, audited pilot implementation, never arbitrary legacy output.
        ctx=read(reference/'context.json');old=ctx['config']
        for key in ['base_model','base_revision','seed','max_tokens','context','eval_chunk','vllm_memory']:
            if old.get(key)!=c['config'][key]:return None
        if old.get('stage2_sha256')!=c['model_sha']:return None
        for key,expected in LEGACY_CODE.items():
            file=reference/'code'/key
            if not file.exists() or sha(file)!=expected:return None
        if sha(reference/'code/grading.py')!=c['grader_sha']:return None
        if read(reference/'eval_rows.json')!=rows:return None
        packages={k.lower():v for k,v in read(reference/'packages_eval.json').items()}
        if any(packages.get(k.lower())!=v for k,v in c['packages'].items()):return None
        # Legacy max_num_seqs multiplied by generations even for greedy evaluation.
        if old.get('generations')!=c['config']['generations']:return None
        validate_inputs(Path(ctx['inputs']))
        file=reference/'generations/b0_eval.json'
    else:return None
    if not file.exists():return None
    outputs=read(file)
    if any(len(v)!=1 for v in outputs.values()):return None
    summarize(outputs,rows)
    return outputs

def evaluate(root,name,reference=None,reuse_only=False):
    c=contract(root,name);dest=root/'evaluation'/name
    immutable_json(dest/'contract.json',c)
    rows=read(root/'data/dev.json')
    existing=dest/'generations/b0_eval.json'
    outputs=reusable_outputs(dest,c,rows) if existing.exists() and len(read(existing))==400 else None
    if outputs is None and name=='A':
        try:outputs=reusable_outputs(reference,c,rows)
        except (ValueError,KeyError,FileNotFoundError):outputs=None
    if outputs is not None:
        write(existing,outputs);write(dest/'metrics.json',summarize(outputs,rows))
        write(dest/'reused.json',dict(source=str(reference or dest)))
        return
    if reuse_only:raise SystemExit(3)
    immutable_json(dest/'eval_rows.json',rows)
    link=dest/'models/b0';link.parent.mkdir(parents=True,exist_ok=True)
    if not link.exists():link.symlink_to((root/'merged'/name).resolve())
    generate(dict(run=str(dest),config=c['config']),'b0','eval',400)
    outputs=read(dest/'generations/b0_eval.json')
    write(dest/'metrics.json',summarize(outputs,rows))
    print('RESULT',name,read(dest/'metrics.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--condition',choices=list(CONDITIONS),required=True);p.add_argument('--reference',type=Path);p.add_argument('--reuse-only',action='store_true')
    a=p.parse_args();evaluate(a.root,a.condition,a.reference,a.reuse_only)
