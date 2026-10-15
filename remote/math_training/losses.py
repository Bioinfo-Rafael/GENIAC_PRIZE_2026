"""Explicit shifted completion likelihoods shared by SFT, reference and DPO."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def collate(rows, pad_id, device):
    length = max(len(r['input_ids']) for r in rows)
    ids, labels, masks = [], [], []
    for r in rows:
        pad = length - len(r['input_ids'])
        ids.append(r['input_ids'] + [pad_id] * pad)
        labels.append(r['labels'] + [-100] * pad)
        masks.append([1] * len(r['input_ids']) + [0] * pad)
    return {k: torch.tensor(v, device=device) for k, v in
            [('input_ids', ids), ('labels', labels), ('attention_mask', masks)]}


def sequence_logps(logits, labels, chunk=64):
    # Chunk+checkpoint the vocabulary normalization; avoid a full FP32 logits copy.
    logits, labels = logits[:, :-1], labels[:, 1:]
    total = logits.new_zeros(logits.shape[0], dtype=torch.float32)
    count = (labels != -100).sum(-1)
    for start in range(0, labels.shape[1], chunk):
        part, target = logits[:, start:start+chunk], labels[:, start:start+chunk]
        def reduce(x, y):
            valid = y != -100
            values = -F.cross_entropy(x.float().reshape(-1, x.shape[-1]),
                                      y.reshape(-1), ignore_index=-100, reduction='none')
            return (values.reshape_as(y) * valid).sum(-1)
        total = total + (checkpoint(reduce, part, target, use_reentrant=False)
                         if torch.is_grad_enabled() else reduce(part, target))
    return total, count


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    return -F.logsigmoid(beta * ((chosen - ref_chosen) - (rejected - ref_rejected)))
