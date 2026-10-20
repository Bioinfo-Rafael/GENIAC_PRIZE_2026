import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from accuracy_utils import (advantages, balanced_subset, matched_plan, select_accuracy,
                            rollout_tokens, online_groups, coverage)
from accuracy_data import split_public
from c_utils import validate_plan
from core import read, write


def rows(n=160):
    return [{'id': str(i), 'dataset': 'gsm8k' if i % 4 == 0 else 'prm800k',
             'chosen_tokens': {'input_ids': [1, 2, 3, 4], 'labels': [-100, -100, 3, 4]}}
            for i in range(n)]


def test_fresh_holdout_never_contains_old_training_or_exploration():
    items = rows(400)
    old = {str(i) for i in range(80)}
    excluded = {str(i) for i in range(80, 120)}
    train, confirm, final = split_public(items, old, excluded, n=40)
    a, b, c = ({r['id'] for r in x} for x in (train, confirm, final))
    assert len(b) == len(c) == 40
    assert not (a & b or a & c or b & c)
    assert not ((b | c) & (old | excluded))
    assert not a & excluded
    assert old <= a


def test_matched_schedules_nested_subsets_and_capacity():
    small = balanced_subset(rows(), 40)
    large = balanced_subset(rows(), 80)
    assert {r['id'] for r in small} <= {r['id'] for r in large}
    a, b = matched_plan(small, 160), matched_plan(large, 160)
    assert a['actual_tokens'] == b['actual_tokens'] == 160
    validate_plan(small, a); validate_plan(large, b)
    assert a['examples'] == b['examples']
    with pytest.raises(ValueError): matched_plan(small, 10000)
    with pytest.raises(ValueError): balanced_subset(rows(), 200)


def test_accuracy_is_primary_even_when_repetition_is_not_halved():
    base = {'n': 400, 'accuracy': .63}
    trials = {
        'short': {'status': 'complete', 'metrics': dict(n=400, accuracy=.635, correct=254, unknown=0, repeated=1)},
        'accurate': {'status': 'complete', 'metrics': dict(n=400, accuracy=.66, correct=264, unknown=2, repeated=140)}}
    assert select_accuracy(base, trials) == 'accurate'
    assert select_accuracy(base, {'short': trials['short']}) is None


def test_online_mask_keeps_generated_eos_and_does_not_invent_it():
    assert rollout_tokens([1, 2], [3, 9, 0, 0], 9, 20) == {
        'input_ids': [1, 2, 3, 9], 'labels': [-100, -100, 3, 9]}
    assert rollout_tokens([1, 2], [3, 4], 9, 20)['labels'][-1] == 4
    with pytest.raises(ValueError): rollout_tokens([1], [], 9, 20)
    with pytest.raises(ValueError): rollout_tokens([1, 2], [3, 4], 9, 3)


def test_advantages_and_sampling_not_conditioned_on_answers():
    assert advantages([1, 1, 1]) == [0, 0, 0]
    assert advantages([0, 0, 0]) == [0, 0, 0]
    adv = advantages([0, 1, 1, 0])
    assert abs(sum(adv)) < 1e-9 and adv[0] < 0 < adv[1]
    a = online_groups(rows(40), 20, 4, 42)
    assert a == online_groups(rows(40), 20, 4, 42)
    assert a != online_groups(rows(40), 20, 4, 43)
    assert len(set(i for group in a[:10] for i in group)) == 40


def test_coverage_rejects_missing_answers():
    out = {'a': [{'grade': {'correct': True}}, {'grade': {'correct': False}}],
           'b': [{'grade': {'correct': False}}, {'grade': {'correct': False}}]}
    result = coverage(out, [{'id': 'a'}, {'id': 'b'}], 2)
    assert result['sampled_pass_at_1'] == .25
    assert result['observed_pass_at_k'] == .5
    with pytest.raises(ValueError): coverage(out, [{'id': 'a'}], 2)


def test_online_gradient_increases_good_response_and_decreases_bad():
    import torch
    from online_worker import policy_objective
    values = torch.tensor([-2., -2.], requires_grad=True)
    loss = policy_objective(values, torch.tensor([2, 2]), advantages([1, 0]), 2)
    loss.backward()
    assert values.grad[0] < 0 < values.grad[1]
    assert values.grad.sum().abs() < 1e-6


def test_tiny_qwen_online_update_and_checkpoint(tmp_path):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    from losses import sequence_logps
    from online_worker import policy_objective
    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8)
    base = Qwen3ForCausalLM(config)
    initial = copy.deepcopy(base.state_dict())
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'],
                                          task_type='CAUSAL_LM'))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]])
    labels = ids.clone(); labels[:, :2] = -100
    before, counts = sequence_logps(model(ids).logits, labels)
    loss = policy_objective(before, counts, advantages([1, 0]), 2)
    loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        after, _ = sequence_logps(model(ids).logits, labels)
    assert after[0]-after[1] > before[0]-before[1]
    model.save_pretrained(tmp_path/'adapter')
    new_base = Qwen3ForCausalLM(config); new_base.load_state_dict(initial)
    loaded = PeftModel.from_pretrained(new_base, tmp_path/'adapter'); loaded.eval()
    with torch.no_grad():
        restored, _ = sequence_logps(loaded(ids).logits, labels)
    assert torch.allclose(after, restored, atol=1e-6)


def test_final_selection_frozen_and_no_candidate_uses_baseline(tmp_path):
    from accuracy_suite import AccuracySuite
    from suite_utils import summarize
    suite = AccuracySuite.__new__(AccuracySuite)
    suite.root = tmp_path
    suite.inputs = tmp_path/'inputs'
    suite.final = [{'id': 'a', 'dataset': 'gsm8k'}]
    output = {'a': [{'text': '1', 'token_ids': [1], 'finish_reason': 'stop',
                     'grade': {'correct': True, 'extraction_failed': False,
                               'grader_error': False, 'gold_unparseable': False}}]}
    suite.summary = {'baseline': {'n': 400, 'accuracy': .63}, 'trials': {}, 'training': {}}
    suite.save = lambda: None
    suite.baseline = lambda split, rows: output
    suite.finish()
    assert read(tmp_path/'selection.json')['candidate'] is None
    assert suite.summary['recommendation'] == 'keep_run03'
    assert suite.summary['final']['baseline']['correct'] == 1
    # Resuming cannot retroactively pick a different dev candidate after final.
    suite.summary['trials']['later'] = {'status': 'complete', 'metrics': {
        'n': 400, 'accuracy': .9, 'correct': 360, 'unknown': 0}}
    suite.finish()
    assert read(tmp_path/'selection.json')['candidate'] is None


@pytest.mark.parametrize('confirm_low,final_low,recommendation', [
    (-.01, .02, 'keep_run03'),
    (.01, -.01, 'keep_run03_final_gain_unconfirmed'),
    (.01, .01, 'accuracy_gain_supported')])
def test_confirmation_and_final_gate_are_separate(tmp_path, confirm_low, final_low, recommendation):
    from accuracy_suite import AccuracySuite
    suite = AccuracySuite.__new__(AccuracySuite)
    suite.root, suite.inputs = tmp_path, tmp_path/'inputs'
    suite.final = [{'id': 'a', 'dataset': 'gsm8k'}]
    output = {'a': [{'text': '1', 'token_ids': [1], 'finish_reason': 'stop',
                     'grade': {'correct': True, 'extraction_failed': False,
                               'grader_error': False, 'gold_unparseable': False}}]}
    suite.summary = {'baseline': {'n': 400, 'accuracy': .63}, 'training': {},
        'trials': {'best': {'status': 'complete', 'adapter': '/candidate', 'recipe': {},
                           'metrics': {'n': 400, 'accuracy': .67, 'correct': 268, 'unknown': 0}}}}
    suite.save = lambda: None
    suite.baseline = lambda split, rows: output
    evaluated = []
    def evaluate(name, adapter, recipe, split):
        evaluated.append(split)
        return {'comparison': {'paired_bootstrap_95': [
            confirm_low if split == 'confirmation' else final_low, .05]}}
    suite.evaluate = evaluate
    suite.finish()
    assert suite.summary['recommendation'] == recommendation
    assert evaluated == (['confirmation'] if confirm_low <= 0 else ['confirmation', 'final'])


def test_accuracy_command_has_no_legacy_time_cutoff(tmp_path):
    from accuracy_suite import AccuracySuite
    from suite import PhaseFailure
    suite = AccuracySuite.__new__(AccuracySuite)
    suite.root = tmp_path
    suite.summary = {}
    suite.save = lambda: None
    suite.cutoff = suite.deadline = 0  # An old-style suite would stop immediately.
    suite.command([sys.executable, '-c', 'print("ok")'], 'unlimited')
    assert 'ok' in (tmp_path/'logs/unlimited.log').read_text()
    with pytest.raises(PhaseFailure):
        suite.command([sys.executable, '-c', 'raise SystemExit(1)'], 'failure')
