"""Colab launcher utilities; safe bundle restoration and isolated child processes."""
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def restore_bundle(bundle, workspace, expected_sha256):
    """Exact trusted archive only. Verify every member before writing; never overwrite differences."""
    if sha256_file(bundle) != expected_sha256:
        raise RuntimeError('ZIPの内容が用意したものと異なります。run02_colab_inputs.zipを確認してください。')
    workspace = Path(workspace).resolve()
    with ZipFile(bundle) as z:
        manifest = json.loads(z.read('bundle_manifest.json'))
        if sorted(z.namelist()) != sorted(['bundle_manifest.json', *manifest['files']]):
            raise RuntimeError('ZIP内のファイル一覧がmanifestと一致しません。')
        pending = []
        for name, checksum in manifest['files'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('不正なZIP内パスです。')
            destination = workspace.joinpath(*relative.parts)
            if not destination.resolve().is_relative_to(workspace):
                raise RuntimeError('保存先がworkspace外です。')
            payload = z.read(name)
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise RuntimeError('ZIP内ファイルのハッシュが一致しません。')
            if destination.exists():
                if sha256_file(destination) != checksum:
                    raise RuntimeError('Driveに別内容のファイルがあります。上書きしません: '+name)
            else:
                pending.append(name)
        # All conflicts checked before any writes.
        for name in pending:
            target = workspace/name
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name+'.run02_tmp')
            tmp.write_bytes(z.read(name))
            os.replace(tmp, target)
        return {'added': len(pending), 'already_present': len(manifest['files'])-len(pending)}


def run_logged(args, log_path, echo=True):
    """Fresh interpreter means no manual kernel restart; SIGINT terminates its whole process group."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', buffering=1) as log:
        log.write('\n=== process start ===\n')
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line)
                if echo:
                    print(line, end='', flush=True)
            rc = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            process.stdout.close()
        if rc:
            raise RuntimeError(f'処理が終了コード{rc}で停止しました。ログ: {path}')


def completed_training(workspace, run_id):
    root = Path(workspace)
    report_path = root/'experiments'/run_id/'training_summary.json'
    adapter = root/'adapters'/(run_id+'_stage2')/'adapter_model.safetensors'
    if not report_path.exists():
        return False
    report = json.loads(report_path.read_text())
    if report.get('complete') is not True or not adapter.is_file():
        raise RuntimeError('完了記録とStage 2の保存状態が一致しません。')
    if sha256_file(adapter) != report['stage2_weights_sha256']:
        raise RuntimeError('完了時からStage 2の重みが変更されています。')
    return True
