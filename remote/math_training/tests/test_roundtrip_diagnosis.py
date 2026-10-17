from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import save_file
from diagnose_roundtrip import compare_tensors,verify_model_equivalence


def test_comparison_ignores_sharding_but_detects_actual_weight_changes(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    a.mkdir();b.mkdir()
    save_file({'x':torch.tensor([1.,2.]),'y':torch.tensor([3.])},a/'model.safetensors')
    save_file({'x':torch.tensor([1.,2.])},b/'part1.safetensors')
    save_file({'y':torch.tensor([3.])},b/'part2.safetensors')
    assert compare_tensors(a,b)['changed_tensor_count']==0
    assert verify_model_equivalence(a,b)['identical']
    save_file({'y':torch.tensor([3.5])},b/'part2.safetensors')
    report=compare_tensors(a,b)
    assert report['changed_tensor_count']==1
    assert report['changed_tensors'][0]['max_abs_difference']==.5
    assert report['changed_tensors'][0]['changed_elements']==1
    assert not verify_model_equivalence(a,b)['identical']


def test_inference_config_changes_fail_even_with_identical_tensors(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    a.mkdir();b.mkdir()
    for p in (a,b):save_file({'x':torch.tensor([1.])},p/'model.safetensors')
    (a/'generation_config.json').write_text('{"eos_token_id": 1}')
    (b/'generation_config.json').write_text('{"eos_token_id": 2}')
    assert not verify_model_equivalence(a,b)['identical']
