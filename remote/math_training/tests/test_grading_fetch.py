import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import fetch_inputs
from core import digest,sha,write
from grading import grade,numeric_answer,final_candidate


def test_does_not_pick_matching_intermediate_number():
    row={'dataset':'gsm8k'}
    result=grade(row,'We first get 42. Final computation is different.\nAnswer: 41',numeric_answer('42'))
    assert not result['correct'] and not result['extraction_failed']
    assert grade(row,'42',numeric_answer('42'))['correct']
    assert grade(row,'No answer',numeric_answer('42'))['extraction_failed']


def test_only_pinned_files_downloaded_and_corruption_rejected(tmp_path,monkeypatch):
    source=tmp_path/'source';source.write_bytes(b'known')
    manifest={'files':[{'path':'stage2/adapter_model.safetensors','drive_id':'example','sha256':sha(source)}]}
    monkeypatch.setattr(fetch_inputs,'read',lambda _:manifest)
    calls=[]
    def downloader(id,output,quiet):
        calls.append(id);Path(output).write_bytes(b'known');return output
    fetch_inputs.fetch(tmp_path/'inputs',downloader)
    fetch_inputs.fetch(tmp_path/'inputs',downloader)
    assert calls==['example']
    target=tmp_path/'inputs/stage2/adapter_model.safetensors';target.write_bytes(b'bad')
    def broken(id,output,quiet):Path(output).write_bytes(b'wrong');return output
    with pytest.raises(ValueError,match='hash'):fetch_inputs.fetch(tmp_path/'inputs',broken)


def test_manifest_cannot_escape_destination(tmp_path,monkeypatch):
    monkeypatch.setattr(fetch_inputs,'read',lambda _:{'files':[{'path':'../escape','drive_id':'x','sha256':'x'}]})
    with pytest.raises(ValueError,match='path'):fetch_inputs.fetch(tmp_path/'inputs',lambda **_:None)
