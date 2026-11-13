"""Write a benchmark setup into the files that define it (packager only).

Usage: python scripts/apply_setup.py --initial N [--rho R] [--sides train,test,method]

  train/test: INITIAL_CROSSINGS in eval/<side>/tasks.py (shared initial
              demonstration set of every method on that split);
  method:     the default RHO of method/gp_acl.py and the --rho of the
              reference and random-control manifests (scripts/method.sh,
              scripts/baseline.sh).

Each edit replaces exactly one canonical line; anything else is an error.
Idempotent: applying the current values leaves every file byte-identical.
"""
import argparse
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def replace_one(path, pattern, text):
    source=path.read_text()
    matches=re.findall(pattern,source,flags=re.M)
    if len(matches)!=1:
        raise SystemExit(f'{path}: expected one match of {pattern!r}, found {len(matches)}')
    updated=re.sub(pattern,text,source,count=1,flags=re.M)
    if updated!=source:
        path.write_text(updated)
    return updated!=source


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--initial',type=int,required=True,choices=[1,2,3,4])
    parser.add_argument('--rho',type=float,default=None)
    parser.add_argument('--sides',default='train,test,method')
    args=parser.parse_args()
    sides=set(args.sides.split(','))
    if not sides<= {'train','test','method'}:
        raise SystemExit('unknown side')
    changed=[]
    for side in ('train','test'):
        if side in sides:
            path=ROOT/'eval'/side/'tasks.py'
            if replace_one(path,r'^INITIAL_CROSSINGS = \d+$',f'INITIAL_CROSSINGS = {args.initial}'):
                changed.append(str(path.relative_to(ROOT)))
    if 'method' in sides:
        if args.rho is None or not args.rho>0:
            raise SystemExit('--rho (positive) is required for the method side')
        rho=repr(float(args.rho))
        path=ROOT/'method'/'gp_acl.py'
        if replace_one(path,r'^RHO = [0-9.e-]+$',f'RHO = {rho}'):
            changed.append(str(path.relative_to(ROOT)))
        for script in ('method.sh','baseline.sh'):
            path=ROOT/'scripts'/script
            if replace_one(path,r'"--prior","tied","--rho","[0-9.e-]+"\]',f'"--prior","tied","--rho","{rho}"]'):
                changed.append(str(path.relative_to(ROOT)))
    print(f'applied initial={args.initial} rho={args.rho} sides={sorted(sides)} changed={changed}',flush=True)


if __name__=='__main__':
    main()
