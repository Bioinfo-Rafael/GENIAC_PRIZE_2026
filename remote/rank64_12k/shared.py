"""Import the existing rank experiment without accessing inference questions."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
RANK=HERE.parent/'stage1_rank_ablation'
sys.path.insert(0,str(RANK))
from common import *
# common.HERE is the rank helper directory; retain our own directory explicitly.
HERE=Path(__file__).resolve().parent
NAME='R12'
COUNTS={'prm800k':9000,'gsm8k':3000}
RECIPE=dict(base_model=BASE,base_revision=REVISION,stage1='run01 pretrained adapter merged in FP16',
            initialization='fresh math LoRA',rank=64,alpha=128,counts=COUNTS,epochs=1,
            updates=750,learning_rate=5e-5,effective_batch=16,max_length=2048,
            optimizer='PagedAdamW8bit',scheduler='cosine',warmup_ratio=.03,warmup_updates=23,
            weight_decay=0.,loss='all shifted nonpadding tokens including prompt; target-weighted effective batch',
            quantization='NF4 double quantization',compute_dtype='float16',seed=42)


def groups_for_12k(order):
    if len(order)!=12000 or sorted(order)!=list(range(12000)):
        raise ValueError('Expected exactly 12000 unique training indices')
    return [order[i:i+16] for i in range(0,12000,16)]


def configure_engine():
    import engine
    engine.TOTAL_STEPS=750
    engine.CONDITIONS[NAME]={'stage1':True,'rank':64,'alpha':128}
    return engine
