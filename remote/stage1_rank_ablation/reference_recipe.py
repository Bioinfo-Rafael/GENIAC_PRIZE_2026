"""Exact pure boxed parser extracted from the executed run03 notebook cell."""
import re

def boxed_answer(text):
    matches = list(re.finditer('\\\\boxed\\s*\\{', text))
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
