"""Print the results table from runs/final-*.jsonl: every model, every task, one harness."""
import json, glob, collections
R = collections.defaultdict(dict)
for f in glob.glob('runs/final-*.jsonl'):
    L = [json.loads(l) for l in open(f)]
    meta = L[0].get('_run', {}); d = [r for r in L if r.get('arm') == 'herd']
    if not d: continue
    src = meta.get('data', '')
    task = 'chains' if 'vt_' in src else 'several facts' if 'multivalue' in src else 'one fact'
    R[meta['model']][task] = (sum(r['correct'] for r in d), len(d),
                              sum(r['cost_usd'] for r in d) / len(d))
cols = ['one fact', 'several facts', 'chains']
print(f"{'model':<26}" + ''.join(f"{c:>16}" for c in cols) + f"{'$/chain':>10}")
print('-' * 76)
for m in sorted(R, key=lambda m: -R[m].get('chains', (0, 1))[0]):
    line = f"{m.split('/')[-1]:<26}"
    for c in cols:
        v = R[m].get(c); line += f"{(f'{v[0]}/{v[1]}' if v else '-'):>16}"
    ch = R[m].get('chains'); line += f"{(f'{ch[2]:.3f}' if ch else '-'):>10}"
    print(line)
