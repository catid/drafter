# Takeover Prompt: Complete DFlash Causal Energy-Adapter Project (`Phase A` -> `Phase B --scale full`)

You are taking over an active, rigorously verified 8× NVIDIA B300 research and engineering run in `/home/catid/drafter` implementing the complete DFlash speculative-decoding causal energy-adapter specification in `/home/catid/drafter/prompt.md`.

 Follow the plan **all the way through** until both **Phase A** (`edraft.post_m4_suite --scale reference`) and **Phase B** (remaining secondary ablations + `--scale full` M1–M4 pipeline) are 100% executed, audited with zero failures, documented, committed, and pushed to `origin/main` (`git@github.com:catid/drafter.git`).

---

## 1. Mandatory First Reads (Before Running Anything)

Read these four files first:
1. `/home/catid/drafter/AGENTS.md` — mandatory rules (optimize hot paths before long runs; parallel independent pre-experiment bug-check gate with `0` open bugs; periodic 20-minute bug-hunt & workflow-effectiveness sweeps; zero smoke/pilot substitutions; claim completion only with current verified artifact evidence).
2. `/home/catid/drafter/prompt.md` — authoritative specification (`M0`–`M5`, secondary ablations, `--scale full`, statistical/calibration/provenance rules).
3. `/home/catid/drafter_handoff/RESUME_HANDOFF.md` — full technical state, verified script SHA256s, scientific findings so far, and copy-paste command runbook.
4. `/home/catid/drafter_handoff/bug_check_ledger.md` — complete log of 65 bug-check sweeps and all caught/resolved bugs.

If you have multi-agent / subagent / swarm capabilities, use dedicated parallel verifiers/bug-hunters on CPU (`CUDA_VISIBLE_DEVICES=''`, pinned to CPUs `224-255` via `nice -n 19 taskset -c 224-255 OMP_NUM_THREADS=2`, wrapping shell checks in `timeout --foreground -k 10 180s ... < /dev/null`) so at least one bug-hunting agent periodically audits live outputs, manifests, and upcoming transitions every ~20 minutes without touching GPUs `0–7`.

---

## 2. Critical Safety Invariants Right Now

1. **DO NOT kill the live 8-GPU Phase-A daemon (`PID 2011387`) or modify any git-tracked file in `/home/catid/drafter` while Phase A is running!**
   - Check `/home/catid/drafter/artifacts/logs/post_m4_suite_state.json` and `ps -ef | grep post_m4_suite` first (`CUDA_VISIBLE_DEVICES=''`).
   - While `status == "running"`, `/home/catid/drafter` MUST stay on `main` @ `b92f0fbfc685ad92034e8da1f54072f7a18ae8be` with `git -C /home/catid/drafter status --short -uno` completely empty (`""`).
   - Always pass `CUDA_VISIBLE_DEVICES=''` to every CPU audit, test, or inspection command while GPU suites are running so zero stray CUDA contexts touch GPUs `0..7` (`GPU g -> CPUs [16g..16g+15]`, `numa_node = g // 2`).
2. **Keep all 28 upstream M0–M4 reference artifacts in `/home/catid/drafter/artifacts/` 100% byte-immutable.**
   - Phase A writes exclusively under `/home/catid/drafter/artifacts/post_m4_suite/` (and `/home/catid/drafter/artifacts/logs/`).
   - Phase B writes exclusively under `/home/catid/drafter/artifacts/full_scale/`.
3. **If `/tmp/edraft_*` was cleared (e.g., after a machine reboot)**:
   - Restore `/tmp/edraft_{reporting,m3_validator,09a5469_source,bughunt}` bit-for-bit from `/home/catid/drafter_handoff/` and verify checksums:
     ```bash
     (cd /home/catid/drafter_handoff && sha256sum -c SHA256SUMS.txt)
     rsync -a /home/catid/drafter_handoff/edraft_reporting/ /tmp/edraft_reporting/
     rsync -a /home/catid/drafter_handoff/edraft_m3_validator/ /tmp/edraft_m3_validator/
     rsync -a /home/catid/drafter_handoff/edraft_09a5469_source/ /tmp/edraft_09a5469_source/
     mkdir -p /tmp/edraft_bughunt && rsync -a /home/catid/drafter_handoff/edraft_bughunt/ /tmp/edraft_bughunt/
     ```
   - If the machine rebooted and `PID 2011387` is no longer running while `post_m4_suite_state.json["status"] != "complete"`, re-launch `/home/catid/drafter/.venv/bin/python /tmp/edraft_m3_validator/post_m4_suite_driver.py --spawn` (every stage/shard resumes idempotently via SHA256-verified manifests).
4. **Always push to GitHub using the latest active `tmux` SSH agent socket**:
   ```bash
   for s in $(tmux list-sessions -F '#{session_activity} #{session_name}' 2>/dev/null | sort -rn | awk '{print $2}'); do
     line=$(tmux show-environment -t "$s" SSH_AUTH_SOCK 2>/dev/null || true)
     sock="${line#SSH_AUTH_SOCK=}"
     if [ -S "$sock" ] && SSH_AUTH_SOCK="$sock" ssh-add -l >/dev/null 2>&1; then export SSH_AUTH_SOCK="$sock"; break; fi
   done
   ```

---

## 3. Current State Snapshot (GitHub & Local Disk)

- **GitHub (`origin` = `git@github.com:catid/drafter.git`)**:
  - `origin/main` = `b92f0fbfc685ad92034e8da1f54072f7a18ae8be` (Mandatory Reference M0–M4 complete + 24 Post-M4/M5 commits; matches live `/home/catid/drafter` `HEAD`).
  - `origin/feat/phase-b-full-scale` = `6967ad50cca2c03cadeae9c7963aa01daf95bda6` (also local `feat/phase-b-full-scale` & `handoff/phase-b-verified-6967ad50`, plus `/home/catid/drafter_handoff/phase_b_verified_6967ad50.bundle`): **100% coded, hotpath-optimized, and quad-verified on CPU (`0` open bugs, `0` conflicts with Phase-A doc/manifest files)**.
  - `origin/handoff/state-2026-11-17` = GitHub backup of `RESUME_HANDOFF.md`, `bug_check_ledger.md`, all 22 verified validator/driver/reporting `.py` scripts, and completed Phase-A/M5 JSON snapshots.
- **Phase A (`post_m4_suite --scale reference`, `PID 2011387`)**:
  - **Step 1 (`m2_refit_calibration.json`)**: COMPLETE & QUAD-AUDITED (`exit_code=0`).
  - **Step 2 (`online_head_to_head.json`)**: COMPLETE & QUAD-AUDITED (`exit_code=0`).
  - **Step 3 (`m5_eval`, workspace `m5_eval_ws_9343c88a80db`)**: Offline Fit A (`m5_two_sided_pl.json`), Offline Fit B (`m5_exact_pairs.json`), `offline_kl_eval`, `380/380` `m5/tune` (`m5/settings.json`), and `4,800/4,800` `m5/evaluate` (`m5/summary.json`, `m5/greedy_output_comparisons.json`) are **100% COMPLETE & AUDITED (`52,791/52,791` checks passed)**; final sub-stage 8-GPU `m5/timing` (`14,400` rows) is LIVE now, followed automatically by **Step 4 (`fastpath_bench`)** and **Step 5 (`all` -> `post_m4_suite_summary.json`)**.

---

## 4. End-to-End Execution Plan to Complete the Project

### Stage 1 — Monitor & Audit Remaining Phase-A Steps Until `status == "complete"`
1. Check `post_m4_suite_state.json` and `nvidia-smi` (`CUDA_VISIBLE_DEVICES=''`).
2. Run progressive & post-step CPU audits (`CUDA_VISIBLE_DEVICES='' nice -n 19 taskset -c 224-255 OMP_NUM_THREADS=2`):
   - Suite 3 (`m5_eval`): `/home/catid/drafter/.venv/bin/python /tmp/edraft_bughunt/audit_live_suite3_m5_eval.py`
   - Suite 4 (`fastpath_bench`) & Step 5 (`post_m4_suite_summary`): `/home/catid/drafter/.venv/bin/python /tmp/edraft_bughunt/audit_live_suite4_fastpath_bench.py`
3. Wait until `/home/catid/drafter/artifacts/logs/post_m4_suite_state.json` reports `"status": "complete"` (all 5 steps `exit_code == 0`) and all 8 B300 GPUs are at `0 MiB`.

### Stage 2 — Zero-Idle-Gap Phase-A Finalize, Commit, Push, & Immediate Phase-B 8-GPU Launch
As soon as Phase A reaches `"status": "complete"` and GPUs `0..7` are idle (`0 MiB`):
1. Run the Phase-A doc/manifest updater and strict CPU validators in `/home/catid/drafter`:
   ```bash
   cd /home/catid/drafter
   test -z "$(git status --short -uno)"

   CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/edraft_reporting/update_docs_post_m4.py --repo /home/catid/drafter --scale reference
   CUDA_VISIBLE_DEVICES='' .venv/bin/python artifacts/m4_evidence_validator/validate.py --scale reference --artifacts artifacts
   CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/edraft_m3_validator/validate.py \
     --scale reference --scope full \
     --repo /tmp/edraft_09a5469_source --source_repo /tmp/edraft_09a5469_source \
     --artifacts /home/catid/drafter/artifacts
   ```
2. Selectively stage **ONLY** the 12 Phase-A files (**NEVER `git add -A`**, never raw `*ws*`/`.pt`/`.duckdb`/`.log`, and leave `resume.md` unstaged):
   ```bash
   git add -- \
     AGENTS.md README.md NOTES.md progress.md reports/SUMMARY.md \
     artifacts/milestones.json artifacts/implementation_audit.json \
     artifacts/post_m4_suite/m2_refit_calibration.json \
     artifacts/post_m4_suite/online_head_to_head.json \
     artifacts/post_m4_suite/m5_eval.json \
     artifacts/post_m4_suite/fastpath_bench.json \
     artifacts/post_m4_suite/post_m4_suite_summary.json

   git diff --cached --stat
   git commit -m "Complete Phase-A reference Post-M4 & M5 suite, calibration/online ablations, and fastpath benchmarks"

   CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/edraft_reporting/audit_post_m4_suite.py \
     --scale reference --require_complete --check_implementation_audit \
     --output_dir /home/catid/drafter/artifacts
   ```
3. Push `main` to `origin/main` using the latest `tmux` `SSH_AUTH_SOCK` snippet from §2.4 (`git push origin main`).
4. Immediately merge the quad-verified Phase-B branch (`6967ad50cca2c03cadeae9c7963aa01daf95bda6`), run the quick CPU sanity gate, and spawn the detached 8-GPU Phase-B suite driver (`--timing_repeats 2 --bf16_gemm --fast_decode` across GPUs `0..7`):
   ```bash
   cd /home/catid/drafter
   git merge --no-edit feat/phase-b-full-scale
   test -z "$(git status --short -uno)"

   CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_full_scale_io_fastpaths.py tests/test_phase_b_suite.py
   /home/catid/drafter/.venv/bin/python /tmp/edraft_m3_validator/phase_b_suite_driver.py --spawn
   ```

### Stage 3 — Shepherd & Progressively Audit All 5 Phase-B Steps (`~29.4h` on 8× B300)
Monitor `/home/catid/drafter/artifacts/full_scale/logs/phase_b_suite_state.json` across all 5 steps:
1. `bigram_eviction` (`--scale reference`: `online_bigram` `k_ret=None` vs `kret256` vs `kret128`, `~1.9h`)
2. `full_vocab_lookahead` (`--scale reference`: `lookahead_{static,rerelax}_full` vs `dense_full` + matched-`C0[j]` arms, `~3.1h`)
3. `pure_t1_fit` (`--scale reference`: `--allow_pure_t1_ablation` fit + 8-GPU dual `primary`/`pure_t1` KL + 8-GPU closed-loop eval, `~1.85h`)
4. `full_scale` (`--scale full --allow_large_run`: `2M–4M` primary train tokens, `>=32` dev + `>=96` test `long_thinking` M1 traces, 8-GPU DDP M2 fit, `128×3` std + `64×3` long closed-loop M3 & 9-arm M4 `tune`/`evaluate`/`timing`/`calibration`, `~22.5h`)
5. `all` (`<5s` cache hit -> `artifacts/full_scale/phase_b_suite_summary.json`)

Every ~20–30 minutes while Phase B runs, keep `git status --short -uno == ""` and run progressive CPU audits (`CUDA_VISIBLE_DEVICES=''`):
```bash
CUDA_VISIBLE_DEVICES='' nice -n 19 taskset -c 224-255 OMP_NUM_THREADS=2 \
  /home/catid/drafter/.venv/bin/python /tmp/edraft_reporting/audit_phase_b_full_scale.py \
  --repo /home/catid/drafter --phase_b_dir /home/catid/drafter/artifacts/full_scale --allow_partial
```

### Stage 4 — Finalize, Audit, Commit, & Push Phase B (`--scale full` + Secondary Ablations)
Once `/home/catid/drafter/artifacts/full_scale/logs/phase_b_suite_state.json` reports `"status": "complete"` and all 8 GPUs are at `0 MiB`:
1. Run the Phase-B auto-doc updater + quad post-write validator (`1,304 files/sec`, `<60s` across `70,000+` files) and full test suite:
   ```bash
   cd /home/catid/drafter
   CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/edraft_reporting/update_docs_phase_b.py \
     --repo /home/catid/drafter --scale full --suite all
   CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q tests
   ```
2. Selectively stage updated docs/manifests and **ONLY** the 5 compact top-level `artifacts/full_scale/*.json` reports (**NEVER** stage raw traces, `.pt` checkpoints, `.duckdb`, or per-rank workspaces), remove temporary `/home/catid/drafter/resume.md` if appropriate, commit, run the post-commit audit gate, and push `main` to `origin/main` via the latest `tmux` `SSH_AUTH_SOCK`:
   ```bash
   git add -- \
     AGENTS.md README.md NOTES.md progress.md reports/SUMMARY.md \
     artifacts/milestones.json artifacts/implementation_audit.json \
     artifacts/full_scale/bigram_eviction.json \
     artifacts/full_scale/full_vocab_lookahead.json \
     artifacts/full_scale/pure_t1_fit.json \
     artifacts/full_scale/full_scale.json \
     artifacts/full_scale/phase_b_suite_summary.json

   git diff --cached --stat
   git commit -m "Complete Phase-B secondary M5 ablations and full-scale M1-M4 evaluation"

   CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/edraft_reporting/audit_phase_b_full_scale.py \
     --repo /home/catid/drafter --phase_b_dir /home/catid/drafter/artifacts/full_scale \
     --require_complete --check_implementation_audit

   git push origin main
   ```
3. Double-check `prompt.md` end-to-end to confirm zero remaining gaps, and present the complete M0–M5 + Secondary Ablations + Full-Scale results with exact numbers, bootstrap CIs, and file links.
