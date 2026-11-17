# Resumption Runbook & Complete State Handoff (`DFlash Causal Energy Adapter`)

**Generated UTC**: `2026-11-17T19:22:00Z`  
**Authoritative Spec**: `/home/catid/drafter/prompt.md` + `/home/catid/drafter/AGENTS.md`  
**Primary Repository**: `/home/catid/drafter` (`origin`: `git@github.com:catid/drafter.git`)  
**Pushed GitHub Branches (`origin`)**:
- `origin/main` (`b92f0fbfc685ad92034e8da1f54072f7a18ae8be`) — exact live Phase-A `post_m4_suite` codebase running on GPUs `0..7`
- `origin/feat/phase-b-full-scale` (`6967ad50cca2c03cadeae9c7963aa01daf95bda6`) — quad-verified Phase-B secondary ablations + `--scale full` fastpath codebase (`0` conflicts with Phase-A doc/manifest commit)
- `origin/handoff/state-2026-11-17` — complete GitHub-hosted snapshot containing `RESUME_HANDOFF.md`, `handoff/bug_check_ledger.md`, `handoff/edraft_{reporting,m3_validator,bughunt}/*.py`, `artifacts/post_m4_suite/{m2_refit_calibration,online_head_to_head}.json`, and `handoff/m5_completed_so_far/*.json`
**Persistent Local Handoff Bundle**: `/home/catid/drafter_handoff/` (non-`/tmp` mirror of all drivers, validators, reporting scripts, git bundle, and `SHA256SUMS.txt`)

---

## 0. CRITICAL FIRST RULES FOR THE RESUMING AGENT

1. **DO NOT KILL `PID 2011387` (`post_m4_suite_driver.py`) OR MODIFY TRACKED FILES IN `/home/catid/drafter` WHILE PHASE A IS STILL RUNNING!**
   - The detached 8-GPU Phase-A driver (`PID 2011387`) is **LIVE RIGHT NOW** across all **8× NVIDIA B300 GPUs (`0..7`)** running on `/home/catid/drafter` (`main` @ `b92f0fbfc685ad92034e8da1f54072f7a18ae8be`, `git status --short -uno == ""`).
   - While `PID 2011387` is running, keep `CUDA_VISIBLE_DEVICES=''` on all CPU inspection/validation commands and keep `/home/catid/drafter` tracked files untouched (`git -C /home/catid/drafter status --short -uno` MUST stay empty) until `post_m4_suite_state.json["status"] == "complete"`.
2. **If `/tmp/edraft_*` was cleared (e.g., after a reboot)**:
   - Restore `/tmp/edraft_{reporting,m3_validator,09a5469_source,bughunt}` identically from `/home/catid/drafter_handoff/`:
     ```bash
     (cd /home/catid/drafter_handoff && sha256sum -c SHA256SUMS.txt)
     rsync -a /home/catid/drafter_handoff/edraft_reporting/ /tmp/edraft_reporting/
     rsync -a /home/catid/drafter_handoff/edraft_m3_validator/ /tmp/edraft_m3_validator/
     rsync -a /home/catid/drafter_handoff/edraft_09a5469_source/ /tmp/edraft_09a5469_source/
     mkdir -p /tmp/edraft_bughunt && rsync -a /home/catid/drafter_handoff/edraft_bughunt/ /tmp/edraft_bughunt/
     ```
   - And if `PID 2011387` ever stopped mid-run (e.g., machine reboot), simply re-run `/home/catid/drafter/.venv/bin/python /tmp/edraft_m3_validator/post_m4_suite_driver.py --spawn` — every stage and shard is idempotently resumable via SHA256-verified manifests!

---

## 1. Exact Current Project State (What Is Done, What Is Live, What Remains)

### A. Mandatory Reference M0–M4 (`--scale reference`) — 100% COMPLETE & PUSHED
- Committed & pushed to `origin/main` at `586826ee99fb4b0017e3a19f22e8eb22c6ad58ed` (`64,141,807` M3+M4 validator checks passed, `0` failures).
- Local `main` in `/home/catid/drafter` is `b92f0fbfc685ad92034e8da1f54072f7a18ae8be` (2 commits ahead of `origin/main` fixing `post_m4_suite.py` multi-GPU idle assertion & workspace reuse).
- All 28 upstream M0–M4 reference artifacts in `/home/catid/drafter/artifacts/` are frozen and verified byte-identical before and after every suite.

### B. Phase A — Reference Post-M4 & M5 Suite (`post_m4_suite --scale reference`) — LIVE ON ALL 8× B300 GPUs (`~5–6.5h` Remaining)
- **Detached Driver**: `/tmp/edraft_m3_validator/post_m4_suite_driver.py` (`PID 2011387`)
- **Live State File**: `/home/catid/drafter/artifacts/logs/post_m4_suite_state.json`
- **Step-by-Step Progress**:
  1. **Step 1 (`m2_refit_calibration`) — COMPLETE & QUAD-AUDITED (`exit_code=0`)**
     - Output: `/home/catid/drafter/artifacts/post_m4_suite/m2_refit_calibration.json` (`sha256: 3a311f848645c653b8f1f240d39d3f57c06c1b90a118b0c5e8769c5a758611dc`).
     - Key result: Block-shuffled + weight-decayed M2 refit + `beta=0.20` calibration flips held-out HumanEval test grouped KL positive (`1.2170 -> 1.1820` dense `-2.88%`, `1.1642` dense+sparse `-4.34%`) and statically improves OOD `long_thinking` `E[τ]` (`2.4764 -> 2.5159` `+1.60%` dense, `2.5677` `+3.69%` dense+sparse, `0` loops).
  2. **Step 2 (`online_head_to_head`) — COMPLETE & QUAD-AUDITED (`exit_code=0`)**
     - Output: `/home/catid/drafter/artifacts/post_m4_suite/online_head_to_head.json` (`sha256: 53d62acf3119ed039876ce6433eff2d8ecde3633814b43a00a320d984d6344a3`).
     - Key result: `zero_offline_online_only` (`W=A=b=0`) unlocks **`E[τ] = 3.0294` (`+0.5375*`, `+21.7%`)** overall on `long_thinking` and **`ΔE[τ] = +1.1617*` (`+63.6%`) with `paired_isolated_tps_ratio = 1.1571×` (`+15.71%` net wall-clock speedup!) in deep `8K–16K`**; `online_dbS_zero_init` improves both `std@0.0` (`+0.0404*`) and `long@0.6` (`+0.1029`, `+0.3060` in `8K–16K`) with `0 / 480` loops.
  3. **Step 3 (`m5_eval`, workspace `m5_eval_ws_9343c88a80db`, `PID 2536581`) — FINAL SUB-STAGE (`m5/timing`) LIVE NOW**:
     - Offline Fit A (`m5_two_sided_pl.json`, `sha256: 19b6a97f...`): **100% DONE & AUDITED** (`+11.57%` calibrated dev `live_full` KL gain at `optimal_beta=0.15`).
     - Offline Fit B (`m5_exact_pairs.json`, `sha256: ef258bc4...`): **100% DONE & AUDITED** (`0` collisions, `0` overflow, `151,095` active keys = `0.901%` of `2^24`).
     - Held-Out `offline_kl_eval` (`result-4374dd1b2c.json`, `sha256: b64ef284...`): **100% DONE & AUDITED** (`ExactObservedPairs` vs `SignedHash` KL within `2.52e-5` dev / `6.45e-5` test nats = `>99.997%` collision cancellation).
     - 10-Arm Dev `m5/tune` (`380 / 380` rows) & `m5/settings.json` (`sha256: 8d76f0b1...`): **100% DONE & DUAL `0-ULP` AUDITED**.
     - 10-Arm Held-Out `m5/evaluate` (`4,800 / 4,800` rows) & `m5/greedy_output_comparisons.json` (`1,280 / 1,280`): **100% DONE & AUDITED (`52,791 / 52,791` checks passed!)** — on `long_thinking` (`n=32`), `signed_feature_hash_online` (`2.4954`, **`+0.6644` / `1.1831×` in `8K–16K`**) and `offline_online_stride1` (`2.4925`, **`+0.6490` / `1.3064×` in `8K–16K`**) beat `plain_dflash` (`2.4764`), `exact_observed_pairs_online` (`2.4653`, `+0.3160` in `8K–16K`) matches `offline_online_stride1` (`3.1958 == 3.1958`) on `std@0.0`, and `lookahead_{rerelax,static}_candidate` (`2.2569`, `2.2302`) beat matched-`C0[j]` `dense_candidate` (`1.9802`) by **`+14.0%` / `+12.6%`**.
     - 10-Arm Isolated `m5/timing` (`14,400` rows across GPUs `0..7`, `PIDs 2833462..2833469`): **LIVE NOW (`~55 / 14,400` rows done on front-loaded `long_thinking` prompts, ~3.5–4.5h left)** -> writes `/home/catid/drafter/artifacts/post_m4_suite/m5_eval.json`.
  4. **Step 4 (`fastpath_bench`)**: Automatically launched by `PID 2011387` right after `m5_eval` (~1.5–2.0h) -> writes `/home/catid/drafter/artifacts/post_m4_suite/fastpath_bench.json`.
  5. **Step 5 (`all`)**: Automatically launched by `PID 2011387` right after `fastpath_bench` (`<2s` cache hit) -> writes `/home/catid/drafter/artifacts/post_m4_suite/post_m4_suite_summary.json` and sets `post_m4_suite_state.json["status"] = "complete"`.

### C. Phase B — Secondary Ablations + Full Scale (`feat/phase-b-full-scale` @ `6967ad50cca2c03cadeae9c7963aa01daf95bda6`) — 100% CODED, OPTIMIZED, & QUAD-VERIFIED ON CPU (`0` Open Bugs)
- **Git Branch in `/home/catid/drafter`**:
  - `feat/phase-b-full-scale` -> **`6967ad50cca2c03cadeae9c7963aa01daf95bda6`**
  - Backup ref: `handoff/phase-b-verified-6967ad50` -> **`6967ad50cca2c03cadeae9c7963aa01daf95bda6`**
  - Offline git bundle: `/home/catid/drafter_handoff/phase_b_verified_6967ad50.bundle`
  - Touches ONLY `13` `src/edraft/*.py` + `5` `tests/*.py` files (`0` overlap with the 12 Phase-A doc/manifest files — merges cleanly with `0` conflicts after the Phase-A commit!).
- **Validated Fastpaths Included on `6967ad50`**:
  1. `--fast_decode`: `PreallocatedDynamicLayer` + native PyTorch 2.13 `causal_lower_right` 8-KV-head FlashAttention-2 GQA on B300 `sm_103` (`12.46×` verification attention speedup at `L=12,288`; `2.37×` end-to-end `long_thinking` decode speedup) across all closed-loop stages AND `--scale full` M1a `collect.py`!
  2. `--bf16_gemm` + zero-host-sync `CompactProposal` across BOTH `candidate_masked` & `full_vocabulary` `_FullVocabLookaheadCache` (accelerates all `32` `[K, 151936]` GEMMs/block on B300 TensorCores and eliminates `15` D2H `.item()` syncs/block).
  3. `--timing_repeats 2` (`-14,144` redundant 3rd-repeat generations = `25%` timing compute savings while running `2` rotated repeats + `1` profile across all slots).
  4. Quadratic-KV Condition-Stratified Greedy LPT (`0.46–0.51%` 8-GPU spread) across `pure_t1_fit`, `compute_tune_sharding_plan`, and `compute_timing_sharding_plan`, plus 8-GPU LPT `dual_offline_kl_eval` (`7.31×` speedup).
  5. Fused M1a `capture(block)` (`6 -> 2` GPU syncs/block, single `[15, 151936]` softmax, `0-ULP` bit-exact `compact_block` + full CPU `validate_block`) + CUDA-buffered M2 `SparseAccessRecorder(flush_every=128)` (`42,155 -> ~330` Rank-0 `fsync` Parquet chunks, `0-ULP` crash/resume & collision stats).
- **Detached Phase-B Driver**: `/tmp/edraft_m3_validator/phase_b_suite_driver.py` (`sha256: d57f149d2816d843279a0b1fb5e01bcd43ab5599308390ac51420a60c61e3f2d`) — runs all 5 steps isolated under `/home/catid/drafter/artifacts/full_scale`:
  - Step 1: `bigram_eviction` (`--scale reference`, `~1.9h`)
  - Step 2: `full_vocab_lookahead` (`--scale reference`, `~3.1h`)
  - Step 3: `pure_t1_fit` (`--scale reference`, `~1.85h`)
  - Step 4: `full_scale` (`--scale full --allow_large_run`: `2M–4M` primary tokens, `>=32` dev + `>=96` test long M1 traces, 8-GPU DDP M2, `128×3` std + `64×3` long closed-loop M3 & 9-arm M4 `tune`/`evaluate`/`timing`/`calibration`, `~22.5h`)
  - Step 5: `all` (`<5s` cache hit -> `artifacts/full_scale/phase_b_suite_summary.json`)
  - **Total Projected 8× B300 Wall-Clock for Phase B**: **`~29.4 hours`**.

---

## 2. Exact Step-by-Step Commands to Resume & Complete Everything

### Step A — Monitor & Progressively Audit Live Phase A (`PID 2011387`) Until `status == "complete"`
Run strictly with `CUDA_VISIBLE_DEVICES=''` (never touch `/home/catid/drafter` tracked files while running):
```bash
# 1. Check driver state & GPU utilization
CUDA_VISIBLE_DEVICES='' cat /home/catid/drafter/artifacts/logs/post_m4_suite_state.json
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader

# 2. Fast (<15s) progressive / complete CPU audit of Suite 3 (m5_eval)
CUDA_VISIBLE_DEVICES='' nice -n 19 taskset -c 224-255 OMP_NUM_THREADS=2 \
  /home/catid/drafter/.venv/bin/python /tmp/edraft_bughunt/audit_live_suite3_m5_eval.py

# 3. Fast (<5s) progressive / complete CPU audit of Suite 4 (fastpath_bench) & Step 5 (post_m4_suite_summary)
CUDA_VISIBLE_DEVICES='' nice -n 19 taskset -c 224-255 OMP_NUM_THREADS=2 \
  /home/catid/drafter/.venv/bin/python /tmp/edraft_bughunt/audit_live_suite4_fastpath_bench.py
```

### Step B — Finalize, Validate, Commit, & Push Phase A (`post_m4_suite --scale reference`)
As soon as `/home/catid/drafter/artifacts/logs/post_m4_suite_state.json` shows `"status": "complete"` (all 5 steps `exit_code == 0`) and all 8 GPUs are at `0 MiB`:
```bash
cd /home/catid/drafter

# 1. Remove any temporary handoff note in artifacts/logs if desired (untracked, optional) and verify clean tracked tree
git status --short -uno   # MUST be empty

# 2. Run auto-doc updater + full 4-suite PostM4SuiteAuditor + M0-M4 immutability + implementation_audit generator (<60s)
#    (Automatically updates AGENTS.md, README.md, NOTES.md, progress.md, reports/SUMMARY.md,
#     artifacts/milestones.json, and artifacts/implementation_audit.json!)
CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/edraft_reporting/update_docs_post_m4.py --repo /home/catid/drafter --scale reference

# 3. Re-verify upstream M4 & M3 reference validators
CUDA_VISIBLE_DEVICES='' .venv/bin/python artifacts/m4_evidence_validator/validate.py --scale reference --artifacts artifacts
CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/edraft_m3_validator/validate.py \
  --scale reference --scope full \
  --repo /tmp/edraft_09a5469_source --source_repo /tmp/edraft_09a5469_source \
  --artifacts /home/catid/drafter/artifacts

# 4. Selectively stage ONLY the 12 Phase-A files (NEVER `git add -A` or raw workspaces/*.pt/*.duckdb/*.log!)
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

# 5. Post-commit audit gate (verifies _is_post_audit_doc_commit + all SHA256s at new HEAD!)
CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/edraft_reporting/audit_post_m4_suite.py \
  --scale reference --require_complete --check_implementation_audit \
  --output_dir /home/catid/drafter/artifacts

# 6. Push Phase A to origin/main (discover active tmux SSH_AUTH_SOCK if needed)
for s in $(tmux list-sessions -F '#{session_activity} #{session_name}' 2>/dev/null | sort -rn | awk '{print $2}'); do
  line=$(tmux show-environment -t "$s" SSH_AUTH_SOCK 2>/dev/null || true)
  sock="${line#SSH_AUTH_SOCK=}"
  if [ -S "$sock" ] && SSH_AUTH_SOCK="$sock" ssh-add -l >/dev/null 2>&1; then export SSH_AUTH_SOCK="$sock"; break; fi
done
git push origin main
```

### Step C — Merge Pre-Verified Phase B (`6967ad50`) & Immediately Launch 8-GPU Phase-B Driver
Right after `git push origin main` succeeds (zero conflict — `rehearse_sweep56` already proved `0` conflicts between the 12 Phase-A files and `6967ad50`!):
```bash
cd /home/catid/drafter

# 1. Merge quad-verified Phase-B branch (6967ad50cca2c03cadeae9c7963aa01daf95bda6) into main
git merge --no-edit handoff/phase-b-verified-6967ad50
test -z "$(git status --short -uno)"

# 2. Quick pre-launch CPU sanity gate
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_full_scale_io_fastpaths.py tests/test_phase_b_suite.py

# 3. Launch detached 8-GPU Phase-B driver (--timing_repeats 2 --bf16_gemm --fast_decode across GPUs 0..7)
/home/catid/drafter/.venv/bin/python /tmp/edraft_m3_validator/phase_b_suite_driver.py --spawn
```

### Step D — Monitor, Audit, Document, Commit, & Push Phase B (`--scale full` + Secondary Ablations)
1. **Monitor live Phase-B state & logs**:
   - State file: `/home/catid/drafter/artifacts/full_scale/logs/phase_b_suite_state.json`
   - Progressive CPU audit at any point (`1,304 files/sec`, zero GPU usage):
     ```bash
     CUDA_VISIBLE_DEVICES='' nice -n 19 taskset -c 224-255 OMP_NUM_THREADS=2 \
       /home/catid/drafter/.venv/bin/python /tmp/edraft_reporting/audit_phase_b_full_scale.py \
       --repo /home/catid/drafter --phase_b_dir /home/catid/drafter/artifacts/full_scale --allow_partial
     ```
2. **When `phase_b_suite_state.json["status"] == "complete"` (`all` 5 steps `exit_code == 0`)**:
   ```bash
   cd /home/catid/drafter

   # 1. Run Phase-B auto-doc updater & quad post-write validator (<60s for 70,000+ files!)
   CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/edraft_reporting/update_docs_phase_b.py \
     --repo /home/catid/drafter --scale full --suite all

   # 2. Run full CPU test suite
   CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q tests

   # 3. Selectively stage code/tests, docs, manifests, and ONLY the 5 top-level Phase-B summary JSONs
   #    (NEVER stage raw trace shards, .pt checkpoints, .duckdb, or per-rank closed-loop workspaces!)
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

   # 4. Post-commit audit check & push to origin/main
   CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/edraft_reporting/audit_phase_b_full_scale.py \
     --repo /home/catid/drafter --phase_b_dir /home/catid/drafter/artifacts/full_scale \
     --require_complete --check_implementation_audit

   git push origin main
   ```

---

## 3. Inventory of Verified Scripts & SHA256s (`/tmp/` & `/home/catid/drafter_handoff/`)

| Path (in `/tmp/` and mirrored under `/home/catid/drafter_handoff/`) | SHA256 Prefix | Role |
|---|---|---|
| `edraft_m3_validator/post_m4_suite_driver.py` | *(live `PID 2011387`)* | Detached 5-step 8-GPU Phase-A driver |
| `edraft_m3_validator/phase_b_suite_driver.py` | `d57f149d2816d843...` | Detached 5-step 8-GPU Phase-B driver (`--timing_repeats 2 --bf16_gemm --fast_decode`) |
| `edraft_m3_validator/validate.py` | *(frozen M3 gate)* | M3 reference validator (`--repo /tmp/edraft_09a5469_source --source_repo /tmp/edraft_09a5469_source`) |
| `edraft_reporting/build_and_audit_reports.py` | `147e67df8b295cda...` | M0–M4 report builder & auditor (`0-ULP` memoized) |
| `edraft_reporting/audit_post_m4_suite.py` | `33ccf72550d475f4...` | Phase-A 4-suite + master summary auditor (`ThreadPoolExecutor(8)`) |
| `edraft_reporting/update_docs_post_m4.py` | `655cce12e1be8289...` | Phase-A idempotent doc/manifest/`AGENTS.md` updater |
| `edraft_reporting/build_implementation_audit.py` | `aac6054f487b3e75...` | AST symbol + SHA256 + `_is_post_audit_doc_commit` verifier |
| `edraft_reporting/audit_phase_b_full_scale.py` | `4a4356a6b05c58b8...` | Phase-B `70,000+`-file auditor (`1,304 files/sec`) |
| `edraft_reporting/update_docs_phase_b.py` | `cd5769bac63143cd...` | Phase-B idempotent doc/manifest updater |
| `edraft_bughunt/audit_live_suite3_m5_eval.py` | *(ProcessPool 20w)* | `<8s` progressive & complete Suite-3 `m5_eval` auditor |
| `edraft_bughunt/audit_live_suite4_fastpath_bench.py` | `ab3f4ad669079f95...` | `<5s` progressive & complete Suite-4 + Step-5 auditor |
| `phase_b_verified_6967ad50.bundle` | `b84e56afe65aa476...` | Complete git bundle of `main` + `feat/phase-b-full-scale` (`6967ad50`) |
| `bug_check_ledger.md` | *(Sweeps #1–#65)* | Full bug-hunt audit trail & caught/closed bug log |
