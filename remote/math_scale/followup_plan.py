"""Prospective queue; no score, official question or prediction is an input."""
from shared import *


def build_plan():
    chains=[]
    def add(key,rank,start,lr,targets,switch=None,role='main'):
        chains.append(dict(key=key,rank=rank,alpha=2*rank,start_questions=start,learning_rate=lr,
            switch_step=switch,targets=targets,role=role,
            experiments=[dict(name=f'{key}_{n}',target=n) for n in targets]))
    # First measure how gently the successful 3k adapter should be continued.
    for key,lr in [('C3_R8_L2e6',2e-6),('C3_R8_L1e6',1e-6),('C3_R8_L5e7',5e-7)]:
        add(key,8,3000,lr,[4000,6000,10000])
    # Capacity and fresh-Stage2 LR controls, same order/labels/parent/seed.
    add('F_R16_L5e5',16,0,5e-5,[3000,6000,10000],65)
    add('F_R32_L2e5',32,0,2e-5,[3000,6000,10000],65)
    add('F_R8_L1e5',8,0,1e-5,[3000,6000,10000],65)
    # Earlier continuation origin: original 300k / 1040-question adapter.
    add('C1_R8_L2e6',8,1040,2e-6,[3000,6000,10000])
    add('C1_R8_L1e6',8,1040,1e-6,[3000,6000,10000])
    add('F_R16_L2e5',16,0,2e-5,[3000,6000,10000],65)
    # Short, independently useful, predetermined tails for the final time window.
    for key,lr in [('TAIL_R8_L15e7',1.5e-6),('TAIL_R8_L75e8',7.5e-7),('TAIL_R8_L3e6',3e-6)]:
        add(key,8,3000,lr,[3500],role='short_reserve')
    return dict(version='followup_v1',chains=chains,
        execution_order=[e['name'] for c in chains for e in c['experiments']],
        maximum_additional_submissions=30,
        rule='Fixed order. Runtime/disk only may skip a chain remainder; never alter recipes using public/official scores.',
        deadline='Inherit preceding suite absolute deadline; no additional ten hours.',
        common=dict(seed=42,effective_batch=16,microbatch=1,context=8192,response_plus_eos_nll=True,
            teacher_token_target={'prm800k':.8,'limo':.2},repeat_questions=False),
        history_difference='Existing adapter histories, fresh initialization, cosine horizons and optimizer resets differ. Not a rank-only causal comparison. Exact 4000 checkpoint creates a partial effective batch; count boundaries are frozen.',
        storage='Keep every new adapter, full milestone optimizer checkpoint, JSON, hashes and evaluation. Only new followup merged weights are temporary and reproducible from shared parent+adapter.')


def schedule_for(full,chain):
    end=min(chain['targets'][-1],len(full));start=chain['start_questions']
    rows=full[start:end]
    bounds=[]
    if chain['switch_step'] is not None:
        bounds.append(('warmup1040',1040))
    for exp in chain['experiments']:
        target=min(exp['target'],len(full))
        if target<=start or (bounds and target-start<=bounds[-1][1]):continue
        bounds.append((exp['name'],target-start))
    return rows,count_plan(rows,bounds)


def remaining_disk_bytes(root,chain):
    """One FP16 merge + safety + all remaining full-state snapshots in this chain."""
    pending=sum(not (root/e['name']/'training_complete.json').exists() for e in chain['experiments'])
    # r32 AdamW+FP32 LoRA is ~0.8 GiB per full checkpoint. Keep conservative room.
    return int((12+pending*.25*(chain['rank']/8)+2*(chain['rank']/8))*2**30)
