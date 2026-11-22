"""Wait for a running evaluate_all PID, then run post_train_report."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def pid_alive(pid: int) -> bool:
    import ctypes

    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    kernel32.CloseHandle(h)
    return bool(ok and code.value == 259)  # STILL_ACTIVE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True, help="evaluate_all python PID")
    parser.add_argument("--poll-seconds", type=int, default=90)
    args = parser.parse_args()
    print(f"Waiting for PID {args.pid} to finish …")
    while pid_alive(args.pid):
        time.sleep(args.poll_seconds)
        print(f"  still running ({args.pid}) …")
    print("evaluate_all finished; running post_train_report")
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "post_train_report.py")], cwd=REPO)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())