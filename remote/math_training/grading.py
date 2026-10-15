"""Pure grading helpers copied from colab_t4/experiments/dev_eval.py; no data I/O."""
import re
import math
import statistics
from collections import Counter
from fractions import Fraction

def last_boxed(text):
    matches = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if i and text[i - 1] == '\\':
            continue
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def final_candidate(text):
    # Never search all intermediate numbers for a matching gold answer.
    if re.search(r'\\(?:boxed|fbox)\b', text):
        return last_boxed(text), 'boxed'
    matches = list(re.finditer(r'(?:^|\n)\s*(?:####|(?:final\s+)?answer\s*(?:is|:))\s*([^\n]+)',
                               text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip(), 'answer_marker'
    # A response consisting only of a number is also an unambiguous final answer.
    if numeric_answer(text) is not None:
        return text.strip(), 'numeric_only'
    return None, 'missing'


def numeric_answer(text):
    if text is None:
        return None
    s = text.strip().strip('$').strip().replace('−', '-')
    if s.endswith('.'):
        s = s[:-1]
    m = re.fullmatch(r'\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}', s)
    if m:
        s = m[1] + '/' + m[2]
    # Remove only well-formed thousands separators; don't turn a list into a number.
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', s):
        s = s.replace(',', '')
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?', s):
        return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def parse_math(answer):
    from math_verify import parse, LatexExtractionConfig
    # Input is ONLY the selected final expression, never the reasoning trace.
    answer = answer.strip()
    boxed = last_boxed(answer)
    if boxed is not None:
        answer = boxed
    if answer.startswith('$') and answer.endswith('$'):
        answer = answer.strip('$')
    return parse('$' + answer + '$', extraction_config=[LatexExtractionConfig()],
                 extraction_mode='first_match', fallback_mode='no_fallback', parsing_timeout=3)


def prepare_gold(rows):
    gold, failures = {}, Counter()
    for row in rows:
        value = numeric_answer(row['answer']) if row['dataset'] == 'gsm8k' else parse_math(row['answer'])
        if value is None or value == []:
            failures[row['dataset']] += 1
        gold[row['id']] = value
    return gold, dict(failures)


def grade(row, text, gold):
    candidate, method = final_candidate(text)
    result = {'correct': False, 'extraction_failed': True, 'answer_method': method,
              'grader_error': False, 'gold_unparseable': gold is None or gold == []}
    if not candidate:
        return result
    try:
        if row['dataset'] == 'gsm8k':
            pred = numeric_answer(candidate)
            result['extraction_failed'] = pred is None
            result['correct'] = pred is not None and pred == gold
        else:
            from math_verify import verify
            pred = parse_math(candidate)
            result['extraction_failed'] = not bool(pred)
            result['correct'] = (not result['gold_unparseable'] and bool(pred)
                                 and bool(verify(gold, pred, strict=True, timeout_seconds=3, raise_on_error=True)))
    except Exception:
        result['grader_error'] = True
    return result


def wilson(k, n):
    if not n:
        return [None, None]
    z = 1.95996398454
    p, d = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, center - radius), min(1, center + radius)]


def metrics(rows):
    n = len(rows)
    if not n:
        return {'n': 0}
    lengths = sorted(r['completion_tokens'] for r in rows)
    correct = sum(r['correct'] for r in rows)
    return {'n': n, 'correct': correct, 'accuracy': correct / n,
            'wilson95': wilson(correct, n),
            'gold_unparseable_count': sum(r['gold_unparseable'] for r in rows),
            'gradable_n': sum(not r['gold_unparseable'] for r in rows),
            'gradable_accuracy': (correct / sum(not r['gold_unparseable'] for r in rows)
                                  if any(not r['gold_unparseable'] for r in rows) else None),
            'extraction_failure_rate': sum(r['extraction_failed'] for r in rows) / n,
            'length_stop_rate': sum(r['finish_reason'] == 'length' for r in rows) / n,
            'grader_error_count': sum(r['grader_error'] for r in rows),
            'completion_tokens_mean': statistics.mean(lengths),
            'completion_tokens_p95': lengths[math.ceil(.95 * n) - 1],
            'answer_methods': dict(Counter(r['answer_method'] for r in rows)),
            'finish_reasons': dict(Counter(r['finish_reason'] for r in rows))}
