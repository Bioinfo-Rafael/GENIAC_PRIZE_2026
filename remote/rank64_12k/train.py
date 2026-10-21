"""One fresh rank64 adapter. Shared engine owns optimizer, resume, loss and merge."""
import argparse
from shared import *

if __name__=='__main__':
    import signal
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','merge','reference-merge'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--condition',choices=['A','D']);a=p.parse_args()
    cfg=read(a.root/'selected_backend.json')
    if cfg['backend']=='unsloth':
        import os
        os.environ['UNSLOTH_RETURN_LOGITS']='1'
        import unsloth
    engine=configure_engine()
    signal.signal(signal.SIGTERM,engine.stop);signal.signal(signal.SIGINT,engine.stop)
    if a.phase=='train':engine.train(a.root,NAME,cfg)
    elif a.phase=='merge':engine.merge(a.root,NAME)
    else:engine.merge(a.root,a.condition)
