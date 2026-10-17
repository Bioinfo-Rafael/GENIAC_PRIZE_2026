"""Read-only CPU comparison of the baseline and no-update audit model."""
import argparse
import json
from pathlib import Path


def tensor_index(directory):
    from safetensors import safe_open
    index = {}
    for path in sorted(Path(directory).glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                if key in index:
                    raise ValueError('Duplicate tensor: '+key)
                index[key] = path
    if not index:
        raise ValueError('No safetensors found in '+str(directory))
    return index


def compare_tensors(left_dir, right_dir):
    import torch
    from safetensors import safe_open
    left, right = tensor_index(left_dir), tensor_index(right_dir)
    changed, dtype_changes = [], []
    for key in sorted(left.keys() & right.keys()):
        with safe_open(left[key], framework='pt', device='cpu') as a, \
             safe_open(right[key], framework='pt', device='cpu') as b:
            x, y = a.get_tensor(key), b.get_tensor(key)
            if x.dtype != y.dtype:
                dtype_changes.append({'tensor':key,'before':str(x.dtype),'after':str(y.dtype)})
            if x.shape != y.shape:
                changed.append({'tensor':key,'shapes':[list(x.shape),list(y.shape)]})
            elif not torch.equal(x,y):
                count, maximum = 0, 0.
                for start in range(0,x.numel(),1_000_000):
                    p = x.reshape(-1)[start:start+1_000_000].to(torch.float64)
                    q = y.reshape(-1)[start:start+1_000_000].to(torch.float64)
                    count += int((p != q).sum())
                    delta = (p-q).abs()
                    maximum = max(maximum,float(delta.nan_to_num(nan=float('inf')).max()))
                changed.append({'tensor':key,'changed_elements':count,
                                'elements':x.numel(),'max_abs_difference':maximum})
            del x, y
    return {'left_tensor_count':len(left),'right_tensor_count':len(right),
            'missing_on_right':sorted(left.keys()-right.keys()),
            'extra_on_right':sorted(right.keys()-left.keys()),
            'dtype_changes':dtype_changes,'changed_tensor_count':len(changed),
            'changed_tensors':changed}


def json_diff(left, right):
    if not left.exists() or not right.exists():
        return {'left_exists':left.exists(),'right_exists':right.exists()}
    a, b = json.loads(left.read_text()), json.loads(right.read_text())
    return {key:{'before':a.get(key),'after':b.get(key)}
            for key in sorted(a.keys()|b.keys()) if a.get(key)!=b.get(key)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,required=True)
    args = parser.parse_args()
    root = args.suite.expanduser().resolve()
    baseline, audit = root/'shared/models/b0',root/'audit/models/rft'
    print('Comparing merged tensors on CPU (no training or GPU use)...',flush=True)
    result = {'merged_weights':compare_tensors(baseline,audit),
              'config_differences':{name:json_diff(baseline/name,audit/name) for name in
                  ('config.json','generation_config.json','tokenizer_config.json','special_tokens_map.json')}}
    source = json.loads((root/'suite_context.json').read_text()).get('source')
    if source:
        result['generation_environment_differences'] = json_diff(
            Path(source)/'generation_environment.json',root/'audit/generation_environment.json')
    path = root/'roundtrip_diagnosis.json'
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print('Saved:',path)


if __name__ == '__main__':
    main()
