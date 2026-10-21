"""Gold-independent answer selectors; all methods use the existing verifier."""
from collections import defaultdict
import re

METHODS=('current','first_box','last_box','explicit_final')
BOX=re.compile(r'\\(?:boxed|fbox)\s*\{')
FINAL=re.compile(r'(?:\bfinal\s+answer\s*(?:is\b|:|=)|^\s*####)\s*([^\n]+)',re.I|re.M)


def box(text,first=False):
    matches=list(BOX.finditer(text))
    if not matches:return None
    m=matches[0] if first else matches[-1];depth=1
    for i in range(m.end(),len(text)):
        if i and text[i-1]=='\\':continue  # Same escaping rule as existing last_boxed.
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[m.end():i].strip()
    return None


def select(text,method):
    if method=='first_box':return box(text,True)
    if method=='last_box':return box(text)
    if method=='explicit_final':
        matches=list(FINAL.finditer(text))
        if not matches:return None
        candidate=matches[-1].group(1).strip()
        return box(candidate) if BOX.search(candidate) else candidate
    raise ValueError(method)


def length_bin(n):
    for boundary in (256,512,1024,2048):
        if n<=boundary:return f'<= {boundary}'
    return '> 2048'


def summary(records):
    result={}
    for method in METHODS:
        values=[r['methods'][method] for r in records];n=len(values)
        result[method]=dict(n=n,correct=sum(v['correct'] for v in values),
            accuracy=sum(v['correct'] for v in values)/n if n else None,
            missing_selection=sum(not v['candidate'] for v in values),
            extraction_failure=sum(v['extraction_failed'] for v in values),
            grader_error=sum(v['grader_error'] for v in values),
            gold_unparseable=sum(v['gold_unparseable'] for v in values))
    both=[r for r in records if not any(r['methods'][m]['extraction_failed'] for m in ('first_box','last_box'))]
    result['transitions']=dict(n=len(records),both_extractable=len(both),
        first_correct_last_wrong=sum(r['methods']['first_box']['correct'] and not r['methods']['last_box']['correct'] for r in records),
        first_wrong_last_correct=sum(not r['methods']['first_box']['correct'] and r['methods']['last_box']['correct'] for r in records),
        first_correct_last_wrong_both_extractable=sum(r['methods']['first_box']['correct'] and not r['methods']['last_box']['correct'] for r in both))
    return result


def summarize(records):
    groups=defaultdict(list)
    for r in records:
        for label in ('all','dataset:'+r['dataset'],'subject:'+r['subject'],
                      'difficulty:'+r['dataset']+':'+str(r['level']),'length:'+length_bin(r['tokens'])):
            groups[label].append(r)
    return {k:summary(v) for k,v in groups.items()}
