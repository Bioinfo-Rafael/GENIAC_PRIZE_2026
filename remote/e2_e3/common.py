"""CPU contracts shared by the isolated E2/E3 runner."""
import ast
from fractions import Fraction
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent/'math_training'
sys.path.insert(0, str(LEGACY))
from core import read, write, sha, digest, question_id
from suite_utils import immutable_json
INDEPENDENT_SHA = '7eafe2c0c46678af1483ea724948ccd394914771eb8362d4990ef4e91cf87d70'
POLICY = dict(quick='exclude only paired bootstrap upper bound < 0',
              dev='maximum correct; ties prefer A, then E2, then E3-16, then E3-32',
              independent='freeze one dev winner; adopt only if paired 95% lower bound > 0; otherwise A',
              bootstrap_seed=42, bootstrap_draws=2000)


def arithmetic(text):
    """Small exact arithmetic grammar, never eval/sympify untrusted text."""
    if not isinstance(text,str) or len(text)>120: raise ValueError('expression_size')
    tree=ast.parse(text.strip(),mode='eval')
    if len(list(ast.walk(tree)))>60: raise ValueError('expression_size')
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            v=Fraction(ast.get_source_segment(text.strip(),n))
        elif isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            v=visit(n.operand)*(1 if isinstance(n.op,ast.UAdd) else -1)
        elif isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Add):v=a+b
            elif isinstance(n.op,ast.Sub):v=a-b
            elif isinstance(n.op,ast.Mult):v=a*b
            else:v=a/b
        else:raise ValueError('unsupported_expression')
        if abs(v.numerator)>10**20 or v.denominator>10**20:raise ValueError('value_size')
        return v
    return visit(tree.body)


def validate_generated(item,text):
    import json,re
    expected=arithmetic(item['rhs'])
    if arithmetic(item['lhs'])!=expected:return 'source_equation_wrong'
    try:
        if item['form']=='json':
            obj=json.loads(text)
            if set(obj)!={'expression','value'} or obj['expression']!=item['lhs'] or not isinstance(obj['value'],str):
                return 'schema_or_expression'
            value=obj['value']
        elif item['form']=='equation':
            parts=text.strip().split('=')
            if len(parts)!=2 or parts[0].strip()!=item['lhs']:return 'equation_format'
            value=parts[1].strip()
        else:
            m=re.fullmatch(r'\\boxed\{([^{}]+)\}',text.strip())
            if not m:return 'boxed_format'
            value=m[1]
        if arithmetic(value)!=expected:return 'wrong_value'
    except (ValueError,TypeError,SyntaxError,ZeroDivisionError):return 'unverifiable'
    return None


def centered(rewards):
    if not rewards:raise ValueError('empty group')
    mean=sum(rewards)/len(rewards)
    return [r-mean for r in rewards]


def choose(results):
    order=['A','E2','E3-16','E3-32']
    eligible=[n for n in order if results.get(n,{}).get('status')=='complete']
    if 'A' not in eligible:raise ValueError('Full baseline required')
    return max(eligible,key=lambda n:results[n]['metrics']['correct'])
