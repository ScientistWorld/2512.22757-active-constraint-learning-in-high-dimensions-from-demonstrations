"""Compare two evaluator-owned frozen raw comparison files (packager check).

Usage: python scripts/compare_frozen.py OLD NEW. Reports, per frozen method
and experiment, how many instances have identical predictions and identical
oracle ledgers (requested endpoints and returned demonstrations). Exits 1 on
any difference.
"""
import json
import sys

old=json.load(open(sys.argv[1]));new=json.load(open(sys.argv[2]))
different=0
for name in ('gp_acl','random_baseline'):
    for exp,records in old[name]['records'].items():
        fresh=new[name]['records'][exp]
        same_pred=sum(a['predictions']==b['predictions'] for a,b in zip(records,fresh))
        same_ledger=sum(a['ledger']==b['ledger'] and a['initial_demos']==b['initial_demos'] for a,b in zip(records,fresh))
        print(f'{name} {exp}: {len(records)} vs {len(fresh)} instances; identical predictions {same_pred}, identical ledgers {same_ledger}',flush=True)
        different+=(len(records)!=len(fresh))+(len(records)-same_pred)
print('frozen records reproduced exactly' if not different else f'{different} differences',flush=True)
sys.exit(1 if different else 0)
