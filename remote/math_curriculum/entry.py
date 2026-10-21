import argparse
from pathlib import Path
from prepare import prepare
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    prepare(p.parse_args().root)
