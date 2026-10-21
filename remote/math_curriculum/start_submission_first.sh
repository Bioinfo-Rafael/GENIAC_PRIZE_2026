#!/usr/bin/env bash
# Switch a running original workflow safely; never kill unrelated GPU jobs.
set -euo pipefail
MC_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MC_WORKSPACE="${GENIAC_WORKSPACE:-$HOME/geniac_math}"
MC_ID="${1:?Usage: start_submission_first.sh math_curriculum_RUN_ID}"
MC_SOURCE="${2:-}"
[[ -z "$MC_SOURCE" || "$MC_SOURCE" =~ ^math_curriculum_[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$MC_ID" =~ ^math_curriculum_[a-zA-Z0-9_-]+$ ]] || exit 2
MC_ROOT="$MC_WORKSPACE/math_curriculum/$MC_ID"
mkdir -p "$MC_WORKSPACE"
MC_EXIT=0
python3 - "$MC_ROOT" "$MC_SOURCE" <<'PY' || MC_EXIT=$?
import os,shlex,signal,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve()
old=str(root/'code/math_curriculum/run.py')
new=str(root/('control_continuation_first/run.py' if sys.argv[2] else 'control_submission_first/run.py'))
def processes():
    found=[]
    for line in subprocess.check_output(['ps','-ww','-eo','pid=,args='],text=True).splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2:continue
        try:args=shlex.split(fields[1])
        except ValueError:continue
        if len(args)<5 or args[1]!='-u' or args[2] not in (old,new):continue
        if '--root' not in args or args[args.index('--root')+1]!=str(root):continue
        found.append((int(fields[0]),args[2]))
    return found
found=processes()
if any(script==new for pid,script in found):
    print('Submission-first controller is already running.',flush=True);sys.exit(20)
if len(found)>1:raise SystemExit('Multiple matching coordinators; refusing to signal them')
if found:
    pid=found[0][0]
    print(f'Saving current training state and stopping old coordinator PID={pid}',flush=True)
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
    for _ in range(180):
        if not any(p==pid for p,_ in processes()):break
        time.sleep(1)
    else:raise SystemExit('Still saving after 180 seconds; retry later. No forced termination performed.')
PY
if [[ "$MC_EXIT" == 20 ]]; then exit 0; fi
if [[ "$MC_EXIT" != 0 ]]; then exit "$MC_EXIT"; fi
MC_LOG="$MC_WORKSPACE/$MC_ID.console.log"
nohup bash "$MC_HERE/launch.sh" "$MC_ID" "$MC_SOURCE" >> "$MC_LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$MC_LOG"
