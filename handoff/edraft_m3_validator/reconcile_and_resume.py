"""Reviewed post-m3_tune state reconciliation and SIGHUP-immune resume helper.

Lives strictly in /tmp/edraft_m3_validator/ so zero files inside /home/catid/drafter
are created or modified while the 36 source/checkpoint/report bindings are frozen.

Modes:
  --self-test   Read-only CPU verification against live state schema, SIGHUP double-fork
                signal mask in /tmp, and GPU query parser (0 writes in /home/catid/drafter).
  --check-gpus  Read-only check of GPUs 0..7 against all idle predicates.
  --execute     Run only AFTER PID 2946875 (m3_tune) has exited cleanly and @verifier
                has signed off: runs 8 pre-reconciliation gates, backs up state,
                atomically reconciles reference_continuation_state.json, enforces
                3x15s GPU 0..7 idle gate, double-forks SIGHUP-immune unmodified
                reference_continuation_source.txt, and verifies new driver PID & lock.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import validate

ROOT = Path('/home/catid/drafter')
ARTIFACTS = ROOT / 'artifacts'
PYTHON = ROOT / '.venv/bin/python'
DRIVER_SOURCE = ARTIFACTS / 'reference_continuation_source.txt'
STATE_PATH = ARTIFACTS / 'reference_continuation_state.json'
BACKUP_STATE_PATH = Path('/tmp/edraft_m3_validator/reference_continuation_state.pre_reconcile.json')
LOCK_PATH = ARTIFACTS / 'reference_continuation.lock'
M3_TUNE_LOG = ARTIFACTS / 'reference_continuation_logs/m3_tune.log'
DRIVER_STDOUT_LOG = ARTIFACTS / 'reference-continuation.log'

EXPECTED_DRIVER_SHA256 = '798a272ff45774e056c0fbacd41366898ca8b0252b00d0267edd80321c73f5d0'
EXPECTED_PLAN_SHA256 = 'a5bc136c88537c1923d372eec1af657092444d42ca248dcf65e84fd0f813ec3f'
EXPECTED_ATTEMPT_ID = 'fa8c9571901b4cd38b16fb411102a292'
EXPECTED_INFLIGHT_PID = 2946875
EXPECTED_START_TICKS = 179201686
EXPECTED_FINAL_STDOUT = {'stage': 'm3', 'scale': 'reference', 'mode': 'tune', 'status': 'settings_frozen'}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def proc_info(pid: int) -> dict | None:
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        return {
            'pid': int(pid),
            'state': fields[0],
            'process_group': int(fields[2]),
            'session': int(fields[3]),
            'start_ticks': int(fields[19]),
        }
    except (OSError, ValueError, IndexError):
        return None


def process_scan(group: int) -> list[dict]:
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        info = proc_info(int(entry.name))
        if info is not None and info['process_group'] == group:
            found.append(info)
    return sorted(found, key=lambda item: item['pid'])


def sig_ign_mask(pid: int) -> int:
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        if line.startswith('SigIgn:'):
            return int(line.split(':', 1)[1].strip(), 16)
    raise RuntimeError(f'Missing SigIgn in /proc/{pid}/status')


def query_gpus_idle(indices=tuple(range(8))) -> dict:
    """Read-only GPU 0..7 check combining assert_idle + bench_adapter.assert_idle."""
    gpu_rows = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu,name',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    gpus = {}
    for row in gpu_rows:
        idx_s, uuid_s, mem_s, util_s, name_s = [part.strip() for part in row.split(',', 4)]
        gpus[int(idx_s)] = {
            'index': int(idx_s),
            'uuid': uuid_s,
            'memory_used_MiB': int(mem_s),
            'utilization_percent': int(util_s),
            'name': name_s,
        }
    require(set(indices) <= gpus.keys(), f'Missing planned GPU indices: {sorted(set(indices) - gpus.keys())}')
    uuid_to_idx = {gpus[i]['uuid']: i for i in indices}

    app_rows = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    compute_apps = []
    for row in app_rows:
        if not row.strip():
            continue
        parts = [p.strip() for p in row.split(',', 3)]
        if parts[0] in uuid_to_idx:
            compute_apps.append({
                'gpu_index': uuid_to_idx[parts[0]],
                'gpu_uuid': parts[0],
                'pid': int(parts[1]),
                'process_name': parts[2],
                'used_memory_MiB': int(parts[3]) if parts[3].isdigit() else parts[3],
            })

    violations = []
    for i in indices:
        g = gpus[i]
        if g['memory_used_MiB'] > 128:
            violations.append(f"gpu {i}: memory.used={g['memory_used_MiB']} MiB > 128 MiB")
        if g['utilization_percent'] > 2:
            violations.append(f"gpu {i}: utilization.gpu={g['utilization_percent']}% > 2%")
    for app in compute_apps:
        violations.append(f"gpu {app['gpu_index']}: active compute pid={app['pid']} ({app['process_name']})")

    return {
        'checked_at_utc': now_utc(),
        'all_idle': not violations,
        'violations': violations,
        'gpus': [gpus[i] for i in indices],
        'compute_apps': compute_apps,
    }


def verify_frozen_bindings(state: dict) -> None:
    binding = state.get('frozen_artifact_binding', {})
    require(len(binding) == 36, f'Expected 36 frozen bindings, found {len(binding)}')
    mismatches = []
    for path_str, expected_sha in binding.items():
        actual_sha = validate.digest(path_str)
        if actual_sha != expected_sha:
            mismatches.append((path_str, expected_sha, actual_sha))
    require(not mismatches, f'Frozen artifact binding drift detected: {mismatches}')
    require(validate.digest(DRIVER_SOURCE) == EXPECTED_DRIVER_SHA256, 'reference_continuation_source.txt SHA256 changed')
    require(state.get('plan_sha256') == EXPECTED_PLAN_SHA256, 'state plan_sha256 changed')


def check_m3_tune_log_clean_exit(log_path: Path, attempt_id: str) -> str:
    text = log_path.read_text(errors='replace')
    marker = f'attempt {attempt_id}'
    pos = text.rfind(marker)
    require(pos != -1, f'Attempt marker {marker!r} not found in {log_path}')
    tail = text[pos:]
    require('Traceback (most recent call last):' not in tail, 'Traceback present in m3_tune.log attempt tail')
    nonempty = [line.strip() for line in tail.splitlines() if line.strip()]
    require(bool(nonempty), 'Empty m3_tune.log tail')
    try:
        parsed = json.loads(nonempty[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'Final line of m3_tune.log is not terminal JSON status: {nonempty[-1]!r}') from exc
    require(parsed == EXPECTED_FINAL_STDOUT, f'Unexpected final stdout JSON in m3_tune.log: {parsed}')
    return nonempty[-1]


def build_reconciled_state(state: dict, m3_settings_sha256: str, finished_at: str) -> dict:
    updated = copy.deepcopy(state)
    require(isinstance(updated.get('steps'), list) and len(updated['steps']) == 8, 'state.steps must remain the 8-step list')
    if (updated.get('inflight') is None
            and updated.get('status') == 'reconciled_ready_to_resume'
            and updated.get('settings_sha256', {}).get('m3') == m3_settings_sha256):
        return updated
    inflight = updated.get('inflight')
    require(isinstance(inflight, dict) and inflight.get('attempt_id') == EXPECTED_ATTEMPT_ID
            and inflight.get('step') == 'm3_tune' and inflight.get('pid') == EXPECTED_INFLIGHT_PID,
            'Unexpected inflight record in reference_continuation_state.json')
    attempts = updated.get('attempts', [])
    require(len(attempts) >= 1 and attempts[-1].get('attempt_id') == EXPECTED_ATTEMPT_ID,
            'Final attempts entry does not match inflight attempt_id')

    attempts[-1]['exit_code'] = 0
    attempts[-1]['finished_at_utc'] = finished_at
    attempts[-1]['last_observed_at_utc'] = finished_at
    attempts[-1]['status'] = 'complete'

    updated['inflight'] = None
    updated['status'] = 'reconciled_ready_to_resume'
    updated.setdefault('settings_sha256', {})['m3'] = m3_settings_sha256
    updated['post_m3_tune_reconciliation'] = {
        'reconciled_at_utc': finished_at,
        'reconciled_attempt_id': EXPECTED_ATTEMPT_ID,
        'reconciled_pid': EXPECTED_INFLIGHT_PID,
        'reconciled_start_ticks': EXPECTED_START_TICKS,
        'm3_settings_sha256': m3_settings_sha256,
        'reason': (
            'Parent driver PID 2946860 exited during prior session teardown while child m3_tune '
            'PID 2946875 (pgrp/session 2946875) completed all 20/20 bundles (140/140 requests) '
            'and froze artifacts/m3/settings.json cleanly.'
        ),
    }
    updated['updated_at_utc'] = finished_at
    return updated


def verify_driver_init_invariants(reconciled_state: dict) -> None:
    """Mirror Driver.__init__ lines 225-240 without touching files or locks."""
    require(reconciled_state['plan_sha256'] == EXPECTED_PLAN_SHA256, 'plan_sha256 mismatch')
    inflight = reconciled_state.get('inflight')
    require(inflight is None, 'inflight must be None before Driver.__init__')
    require(isinstance(reconciled_state['settings_sha256'].get('m3'), str)
            and len(reconciled_state['settings_sha256']['m3']) == 64, 'missing m3 settings_sha256')
    require(reconciled_state['attempts'][-1]['status'] == 'complete'
            and reconciled_state['attempts'][-1]['exit_code'] == 0, 'last attempt not marked complete/0')


def spawn_sighup_immune_detached(argv: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> int:
    """Double-fork + setsid + SIG_IGN(SIGHUP) + SIG_DFL(SIGINT/SIGTERM) + execv."""
    read_fd, write_fd = os.pipe()
    pid1 = os.fork()
    if pid1 > 0:
        os.close(write_fd)
        data = b''
        while chunk := os.read(read_fd, 64):
            data += chunk
        os.close(read_fd)
        _, status = os.waitpid(pid1, 0)
        require(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 and data,
                f'Intermediate fork failed with status={status}')
        return int(data.decode().strip())

    try:
        os.close(read_fd)
        os.setsid()
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        pid2 = os.fork()
        if pid2 > 0:
            os.write(write_fd, f'{pid2}\n'.encode())
            os.close(write_fd)
            os._exit(0)
        os.close(write_fd)
        os.setpgid(0, 0)
        os.chdir(str(cwd))
        devnull_fd = os.open(os.devnull, os.O_RDONLY)
        log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(devnull_fd, 0)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        for std_fd in (0, 1, 2):
            os.set_inheritable(std_fd, True)
        max_fd = os.sysconf('SC_OPEN_MAX')
        os.closerange(3, max_fd if max_fd > 3 else 1024)
        target_env = os.environ.copy() if env is None else env
        os.execve(argv[0], argv, target_env)
    except BaseException:
        os._exit(1)
    return -1


def run_self_test() -> int:
    state = json.loads(STATE_PATH.read_text())
    verify_frozen_bindings(state)
    fake_sha = 'a' * 64
    reconciled = build_reconciled_state(state, fake_sha, now_utc())
    verify_driver_init_invariants(reconciled)

    # Test SIGHUP-immune double-fork launcher strictly in /tmp with a 0.4s CPU python sleeper
    with tempfile.TemporaryDirectory(prefix='edraft_sighup_test_') as tmpdir:
        tmp_log = Path(tmpdir) / 'daemon.log'
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        grandchild_pid = spawn_sighup_immune_detached(
            [str(PYTHON), '-c',
             'import os, signal, time; '
             'print({"pid": os.getpid(), "pgrp": os.getpgrp(), '
             '"sighup_ign": signal.getsignal(signal.SIGHUP) == signal.SIG_IGN, '
             '"fds_012_open": all(os.path.exists(f"/proc/self/fd/{fd}") for fd in (0, 1, 2))}, flush=True); '
             'time.sleep(0.4)'],
            Path(tmpdir),
            tmp_log,
            env=env,
        )
        info = proc_info(grandchild_pid)
        require(info is not None and info['process_group'] == grandchild_pid and info['session'] != grandchild_pid,
                f'Detached test process unexpected group/session: {info}')
        ign = sig_ign_mask(grandchild_pid)
        require(bool(ign & 0x1), f'SIGHUP bit 0x1 not ignored in /proc/{grandchild_pid}/status SigIgn={hex(ign)}')
        # Send actual SIGHUP to prove immunity!
        os.kill(grandchild_pid, signal.SIGHUP)
        time.sleep(0.15)
        require(proc_info(grandchild_pid) is not None, 'Detached process died on SIGHUP!')
        time.sleep(0.45)
        require(proc_info(grandchild_pid) is None, 'Detached test process did not finish cleanly')
        logged = tmp_log.read_text().strip()

    gpu_snapshot = query_gpus_idle()
    print(json.dumps({
        'status': 'passed',
        'frozen_bindings_verified': 36,
        'reconciled_schema_verified': True,
        'sighup_double_fork_verified': True,
        'sighup_sigign_hex': hex(ign),
        'daemon_log_sample': logged,
        'current_gpu_idle_snapshot': {
            'all_idle': gpu_snapshot['all_idle'],
            'violations': gpu_snapshot['violations'],
        },
    }, indent=2))
    return 0


def run_execute(tune_audit_output: Path, idle_polls: int = 3, poll_interval_sec: int = 15,
                reconcile_only: bool = False, startup_timeout_sec: float = 30.0) -> int:
    # Gate 1: Inflight PID 2946875 and process group 2946875 completely gone
    info = proc_info(EXPECTED_INFLIGHT_PID)
    require(info is None, f'PID {EXPECTED_INFLIGHT_PID} is still alive: {info}')
    survivors = process_scan(EXPECTED_INFLIGHT_PID)
    require(not survivors, f'Process group {EXPECTED_INFLIGHT_PID} still has survivors: {survivors}')

    # Gate 2: Lock is free right now (proves no driver or child holds flock)
    require(LOCK_PATH.is_file(), f'Expected lock file {LOCK_PATH} is missing')
    with LOCK_PATH.open('r+b') as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('reference_continuation.lock is still held by another process') from exc
        finally:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            except OSError:
                pass

    # Gate 3: State & 36 frozen bindings & driver/plan SHA256 intact
    state = json.loads(STATE_PATH.read_text())
    verify_frozen_bindings(state)

    # Gate 4: Clean terminal stdout line in m3_tune.log after attempt header
    final_line = check_m3_tune_log_clean_exit(M3_TUNE_LOG, EXPECTED_ATTEMPT_ID)

    # Gate 5: Independent raw M3 tune validator (settings.json + session + 20 bundles + 140 rows)
    cfg = SimpleNamespace(scale='reference', scope='tune', repo=ROOT, artifacts=ARTIFACTS, seed=2026)
    audit = validate.Audit(cfg)
    try:
        audit.run()
    except Exception as exc:
        raise RuntimeError(f'M3 tune audit failed: {exc}') from exc
    require(not audit.failures and audit.counts['tune'] == 140, f'M3 tune audit failed: {audit.failures}')
    validate.atomic_json(tune_audit_output, {
        'status': 'passed',
        'scope': 'tune',
        'scale': 'reference',
        'checks': audit.checks,
        'counts': dict(audit.counts),
        'final_stdout_line': final_line,
        'checked_at_utc': now_utc(),
    })

    # Gate 6 (if launching): Multi-sample GPU 0..7 idle check (3x15s, never signaling external jobs)
    samples = []
    if not reconcile_only:
        for sample_idx in range(idle_polls):
            if sample_idx > 0:
                time.sleep(poll_interval_sec)
            snap = query_gpus_idle()
            samples.append(snap)
            require(snap['all_idle'],
                    f'GPU 0..7 idle gate failed on sample {sample_idx + 1}/{idle_polls}: {snap["violations"]}')

    # Gate 7: Backup original state & atomically write reconciled state (idempotent if already reconciled)
    settings_sha = validate.digest(ARTIFACTS / 'm3/settings.json')
    finished_at = now_utc()
    reconciled = build_reconciled_state(state, settings_sha, finished_at)
    verify_driver_init_invariants(reconciled)
    if state.get('inflight') is not None and not BACKUP_STATE_PATH.exists():
        validate.atomic_json(BACKUP_STATE_PATH, state)
    if reconciled != state:
        validate.atomic_json(STATE_PATH, reconciled)

    if reconcile_only:
        print(json.dumps({
            'status': 'reconciled_only',
            'm3_settings_sha256': settings_sha,
            'm3_tune_audit_checks': audit.checks,
        }, indent=2))
        return 0

    # Gate 8: Launch unmodified reference_continuation_source.txt with SIGHUP immunity and clean CUDA visibility
    driver_env = os.environ.copy()
    driver_env.pop('CUDA_VISIBLE_DEVICES', None)
    argv = [str(PYTHON), str(DRIVER_SOURCE), '--scale', 'reference']
    new_pid = spawn_sighup_immune_detached(argv, ROOT, DRIVER_STDOUT_LOG, env=driver_env)

    deadline = time.monotonic() + startup_timeout_sec
    live_state = {}
    entered_running = False
    while time.monotonic() < deadline:
        time.sleep(0.5)
        new_info = proc_info(new_pid)
        require(new_info is not None,
                f'Newly launched driver PID {new_pid} exited during startup; check {DRIVER_STDOUT_LOG}')
        live_state = json.loads(STATE_PATH.read_text())
        if live_state.get('driver_pid') == new_pid and live_state.get('status') in ('failed', 'interrupted'):
            raise RuntimeError(
                f'Driver PID {new_pid} entered {live_state.get("status")}: {live_state.get("last_error")}')
        if (live_state.get('driver_pid') == new_pid
                and live_state.get('status') == 'running'
                and isinstance(live_state.get('inflight'), dict)):
            entered_running = True
            break

    require(entered_running,
            f'Driver PID {new_pid} did not reach status="running" with active inflight step within '
            f'{startup_timeout_sec}s (last status={live_state.get("status")!r})')
    new_info = proc_info(new_pid)
    require(new_info is not None, f'Driver PID {new_pid} exited unexpectedly right after startup')
    ign = sig_ign_mask(new_pid)
    require(bool(ign & 0x1), f'Driver PID {new_pid} missing SIGHUP ignore mask: {hex(ign)}')

    print(json.dumps({
        'status': 'launched',
        'new_driver_pid': new_pid,
        'new_driver_proc': new_info,
        'inflight_step': live_state['inflight'].get('step'),
        'inflight_pid': live_state['inflight'].get('pid'),
        'sigign_hex': hex(ign),
        'm3_settings_sha256': settings_sha,
        'm3_tune_audit_checks': audit.checks,
        'idle_gate_samples': len(samples),
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test', action='store_true')
    group.add_argument('--check-gpus', action='store_true')
    group.add_argument('--reconcile-only', action='store_true')
    group.add_argument('--execute', action='store_true')
    parser.add_argument('--tune-audit-output', type=Path,
                        default=Path('/tmp/edraft_m3_validator/m3_tune_reference_validation.json'))
    parser.add_argument('--idle-polls', type=int, default=3)
    parser.add_argument('--poll-interval-sec', type=int, default=15)
    parser.add_argument('--startup-timeout-sec', type=float, default=30.0)
    args = parser.parse_args()
    if args.self_test:
        return run_self_test()
    if args.check_gpus:
        print(json.dumps(query_gpus_idle(), indent=2))
        return 0
    return run_execute(args.tune_audit_output, args.idle_polls, args.poll_interval_sec,
                       reconcile_only=args.reconcile_only, startup_timeout_sec=args.startup_timeout_sec)


if __name__ == '__main__':
    raise SystemExit(main())
