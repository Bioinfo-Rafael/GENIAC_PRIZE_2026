"""Synthetic files only: never open the official question dataset."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import submit_d as d


def source(tmp_path):
    root=tmp_path/'rank';dest=tmp_path/'submission'
    d.write(root/'conditions/D/complete.json',dict(complete=True,steps=375))
    d.write(root/'selected_backend.json',dict(backend='unsloth'))
    for directory in ['conditions/D/final','parents/stage1','merged/D']:
        path=root/directory;path.mkdir(parents=True,exist_ok=True)
        (path/'adapter_model.safetensors').write_bytes(b'synthetic')
        d.write(path/'adapter_config.json',dict(r=64,lora_alpha=128))
        d.write(path/'artifacts.json',{n:d.sha(path/n) for n in
              ['adapter_model.safetensors','adapter_config.json']})
    return root,dest


def test_d_identity_and_resume(tmp_path):
    root,dest=source(tmp_path);d.prepare(root,dest)
    first=(dest/'manifest.json').read_bytes();d.prepare(root,dest)
    assert (dest/'manifest.json').read_bytes()==first
    m=d.read(dest/'manifest.json')
    assert m['candidate']=='D'
    assert '375 updates' in m['training_recipe'] and '4500' in m['training_recipe']
    assert m['config']['max_tokens']==2048


def test_incomplete_training_rejected(tmp_path):
    root,dest=source(tmp_path)
    d.write(root/'conditions/D/complete.json',dict(complete=True,steps=374))
    with pytest.raises(ValueError,match='375'):d.prepare(root,dest)


def test_changed_merged_weights_rejected(tmp_path):
    root,dest=source(tmp_path)
    (root/'merged/D/adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):d.prepare(root,dest)


def test_wrong_rank_rejected(tmp_path):
    root,dest=source(tmp_path)
    d.write(root/'conditions/D/final/adapter_config.json',dict(r=8,lora_alpha=16))
    with pytest.raises(ValueError,match='rank64'):d.prepare(root,dest)
