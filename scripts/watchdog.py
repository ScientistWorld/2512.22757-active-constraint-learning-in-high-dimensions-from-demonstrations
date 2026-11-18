"""Job-only diagnostics: periodically log memory and long-running Python
processes from /proc (the container has no ps/free). A submitted producer
(a method command, not an evaluator) still running after --abort seconds is
sent SIGABRT; with PYTHONFAULTHANDLER=1 its traceback lands in the failure
record the evaluator prints. Normal instances finish in a few minutes, so
this only fires on a genuine hang; it never touches evaluator processes.

Usage: python scripts/watchdog.py [--every 600] [--abort 2700]
"""
import argparse
import os
import signal
import time

parser = argparse.ArgumentParser()
parser.add_argument('--every', type=float, default=600)
parser.add_argument('--abort', type=float, default=2700)
args = parser.parse_args()
TICK = os.sysconf('SC_CLK_TCK')
PRODUCERS = ('method/gp_acl.py', 'scripts/shortcut_probe.py')


def processes():
    uptime = float(open('/proc/uptime').read().split()[0])
    for pid in os.listdir('/proc'):
        if not pid.isdigit(): continue
        try:
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace').strip()
            stat = open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()
            status = open(f'/proc/{pid}/status').read()
        except OSError:
            continue
        if 'python' not in cmd: continue
        rss = next((int(line.split()[1])//1024 for line in status.splitlines() if line.startswith('VmRSS')), 0)
        cpu = (int(stat[11])+int(stat[12]))/TICK
        age = uptime-int(stat[19])/TICK
        yield int(pid), stat[0], age, cpu, rss, cmd


while True:
    time.sleep(args.every)
    meminfo = dict(line.split(':', 1) for line in open('/proc/meminfo').read().splitlines() if ':' in line)
    rows = sorted(processes(), key=lambda r: -r[2])
    print(f'== watchdog {time.strftime("%H:%M:%S")} MemAvailable {meminfo.get("MemAvailable", "?").strip()} '
          f'MemTotal {meminfo.get("MemTotal", "?").strip()} python processes {len(rows)}', flush=True)
    for pid, state, age, cpu, rss, cmd in rows[:20]:
        print(f'   pid {pid} state {state} age {age:.0f}s cpu {cpu:.0f}s rss {rss}MB {cmd[:150]}', flush=True)
        if age > args.abort and any(p in cmd for p in PRODUCERS) and 'evaluate' not in cmd:
            print(f'   watchdog: producer {pid} running {age:.0f}s; sending SIGABRT for a traceback', flush=True)
            try: os.kill(pid, signal.SIGABRT)
            except OSError: pass
