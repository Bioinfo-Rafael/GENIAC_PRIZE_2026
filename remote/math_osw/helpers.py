"""CPU-only OSW contracts. No competition inputs are read here."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'math_scale'))
from shared import (read, write, sha, digest, counts, count_plan, CONFIG, SEED,
                    immutable_json, link_readonly, file_receipt, verify_files, lr_factor)


def weight(correct):
    if len(correct) != 2 or any(type(v) is not bool for v in correct):
        raise ValueError('Exactly two boolean rewards are required')
    return 1. - sum(correct) / 2


def prefix_ids(tokens):
    ids, labels = tokens['input_ids'], tokens['labels']
    if len(ids) != len(labels) or len(ids) > 8192:
        raise ValueError('Invalid complete training sequence')
    first = next((i for i, v in enumerate(labels) if v != -100), None)
    if first is None or first < 1 or labels[first:] != ids[first:]:
        raise ValueError('Expected response-only labels')
    if first + 2048 > 8192:
        raise ValueError('Training prompt cannot fit the unchanged rollout budget')
    return ids[:first]


def can_start_batch(now, train_deadline, previous_seconds):
    # The first batch is additionally interruptible at token boundaries.
    required = max(previous_seconds or [60.]) * 1.3 + 30.
    return now + required < train_deadline


def candidate_name(questions):
    return f'OSW_C3_R8_L1e6_q{3000 + questions}'


def validate_order(full, order, evaluation, maximum):
    ids = [r['id'] for r in full]
    if ids != order or len(ids) != len(set(ids)) or len(ids) < 3000 + maximum:
        raise ValueError('Frozen unique 10k order changed or too short')
    if set(ids) & {r['id'] for r in evaluation}:
        raise ValueError('Fixed evaluation overlaps training')
    if maximum < 16 or maximum > 128 or maximum % 16:
        raise ValueError('Pilot size must be 16..128 in full batches of 16')
    rows = full[3000:3000 + maximum]
    for row in rows:
        if row['dataset'] not in ('prm800k', 'limo'):
            raise ValueError('Unapproved training source')
        prefix_ids(row['chosen_tokens'])
    return rows
