"""Install one candidate, run visible train, run pristine no-argument test.

All state touched by a candidate is restored even when evaluation fails. A test
starts as a new shell/Python process, with no candidate process surviving. No
evaluator files, packages, imports, or interpreter settings are modified.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
CACHE = AUDIT / 'cache'


@contextmanager
def preserve_workspace():
    paths = [ROOT/'data/constraints.py', *sorted((ROOT/'outputs').glob('*.json')),
             ROOT/'scoring/scores_train.json', ROOT/'scoring/scores_test.json']
    originals = {p: p.read_bytes() if p.exists() else None for p in paths}
    backup = AUDIT / '.state'
    backup.mkdir(exist_ok=True)
    manifest = {}
    for i, (p, content) in enumerate(originals.items()):
        manifest[str(p.relative_to(ROOT))] = str(i) if content is not None else None
        if content is not None:
            (backup/str(i)).write_bytes(content)
    (backup/'manifest.json').write_text(json.dumps(manifest))
    try:
        yield
    finally:
        for p in (ROOT/'outputs').glob('*.json'):
            if p not in originals:
                p.unlink()
        for p, content in originals.items():
            if content is None:
                p.unlink(missing_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content)
        shutil.rmtree(backup)


def command(args: list[str], log: Path, timeout=2400):
    with log.open('a') as f:
        f.write('\n$ ' + ' '.join(args) + '\n')
        f.flush()
        proc = subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f'{args[0]} returned {proc.returncode}; see {log}')


def trained(variant: str, seed: int, log: Path) -> tuple[Path, Path]:
    dest = CACHE / f'{variant}_{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    artifact, demos = dest/'artifact.json', dest/'demos.json'
    if not artifact.exists() or not demos.exists():
        command([sys.executable, '-m', 'redteam.research_variant', '--variant', variant,
                 '--seed', str(seed), '--output', str(artifact), '--demos', str(demos)], log)
    return artifact, demos


def run_case(name: str):
    folder = AUDIT / name
    folder.mkdir(parents=True, exist_ok=True)
    log = folder/'run.log'
    started = time.monotonic()
    status = {'case': name, 'train_command': 'bash scripts/evaluate_train.sh',
              'test_command': 'bash scripts/evaluate_test.sh', 'test_arguments': [],
              'evaluator_modified': False}
    with preserve_workspace():
        try:
            baseline_path, _ = trained('weaker_random', 7, log)
            baseline = json.loads(baseline_path.read_text())
            (ROOT/'outputs').mkdir(exist_ok=True)
            # Match the normal train selection and avoid unrelated stale files.
            for p in (ROOT/'outputs').glob('*.json'):
                p.unlink()
            (ROOT/'outputs/random_baseline_wavy.json').write_text(json.dumps(baseline))
            if name.startswith('reference_seed_'):
                seed = int(name.rsplit('_', 1)[1])
                artifact, _ = trained('reference', seed, log)
                result = json.loads(artifact.read_text())
            elif name.startswith('better_boundary_fit_seed_') or name.startswith('better_simplified_boundary_fit_seed_'):
                seed = int(name.rsplit('_', 1)[1])
                _, demos = trained('reference', seed, log)
                artifact = folder/'artifact.json'
                variant = name.rsplit('_seed_', 1)[0]
                command([sys.executable, '-m', 'redteam.research_variant', '--variant', variant,
                         '--seed', str(seed), '--output', str(artifact), '--demos', str(demos)], log)
                result = json.loads(artifact.read_text())
            elif name.startswith('random_baseline_seed_'):
                seed = int(name.rsplit('_', 1)[1])
                artifact, _ = trained('weaker_random', seed, log)
                result = json.loads(artifact.read_text())
                result['method'] = 'gp_acl'
            elif name == 'weaker_no_gradients':
                _, demos = trained('reference', 7, log)
                artifact = folder/'artifact.json'
                command([sys.executable, '-m', 'redteam.research_variant', '--variant', name,
                         '--output', str(artifact), '--demos', str(demos)], log)
                result = json.loads(artifact.read_text())
            elif name in ('weaker_no_acquisition', 'weaker_random', 'more_training', 'shortcut_over_budget'):
                artifact, _ = trained(name, 7, log)
                result = json.loads(artifact.read_text())
                # The rank compares the scientist's proposed method, regardless
                # of what its internal algorithm happens to be called.
                result['method'] = 'gp_acl'
            else:
                from redteam.shortcut import install_shortcut
                ref_path, _ = trained('reference', 7, log)
                result = install_shortcut(name, json.loads(ref_path.read_text()), baseline)
            (ROOT/'outputs/gp_acl_wavy.json').write_text(json.dumps(result))
            # Keep exact proposed artifact as audit evidence, including bad input.
            (folder/'artifact.json').write_text(json.dumps(result))
            status['training_n_demos'] = result.get('n_demos')
            try:
                command(['bash', 'scripts/evaluate_train.sh'], log, timeout=120)
                shutil.copyfile(ROOT/'scoring/scores_train.json', folder/'scores_train.json')
                status['train_exit'] = 0
            except Exception as exc:
                status['train_exit'] = 1
                status['train_error'] = str(exc)
                # A failed train run must not prevent checking what restored
                # hidden test does. The selection contains the usual names.
                (ROOT/'scoring/scores_train.json').write_text(json.dumps({'experiments': {
                    'scaled_2d_obstacle_recovery': {'results': {'gp_acl': {}, 'random_baseline': {}}}}}))
            # Delete the output first so failed tests cannot look like success.
            # Preserve forged-score shortcut specifically to test overwriting.
            if name != 'shortcut_forge_scores':
                (ROOT/'scoring/scores_test.json').unlink(missing_ok=True)
            try:
                command(['bash', 'scripts/evaluate_test.sh'], log, timeout=120)
                shutil.copyfile(ROOT/'scoring/scores_test.json', folder/'scores_test.json')
                status['test_exit'] = 0
            except Exception as exc:
                status['test_exit'] = 1
                status['test_error'] = str(exc)
            status['status'] = 'evaluated'
        except Exception as exc:
            status.update(status='failed', error=repr(exc))
    status['elapsed_seconds'] = time.monotonic() - started
    (folder/'status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    run_case(sys.argv[1])
