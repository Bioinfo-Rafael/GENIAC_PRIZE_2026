# Stage 1 / rank comparison (B → C,D timing → C → D)

A is the **existing run03 adapter**, never retrained. B/C/D use newly initialized math adapters:

| Condition | Parent | Rank / alpha | Math training |
|---|---|---|---|
| A | run01 Stage 1 | 8 / 16 | Existing run03 only |
| B | Qwen3-4B-Base | 8 / 16 | 375 updates |
| C | run01 Stage 1 | 32 / 64 | 375 updates |
| D | run01 Stage 1 | 64 / 128 | 375 updates |

## Contract and scope

Only the pinned Qwen3-4B-Base revision, run01/run03 adapters, and original public training cache / fixed public dev400 are accessed. No competition inference files or organizer DOCX are read. Existing experiment directories remain untouched. New code is contained in this directory and imports existing generation, grading, tokenizer and statistics helpers.

`run03_reference.json` records the executed notebook configuration. The cache must match the original SHA256, PRM4,500 + GSM1,500 examples, hash-based cached order, **1,598,551 tokens / maximum length 1,923**. No truncation, filtering, or generated answers are added. Every shifted nonpadding token (including the prompt) is supervised: 1,592,551 targets. Loss is normalized across the fixed effective batch of 16 examples, independently of microbatch size.

One epoch; LR 5e-5; seed42; max sequence2048; paged AdamW8bit; cosine decay; 12 warmup updates; weight decay0; gradient clipping1; same seven projection modules, dropout0. NF4 double quantization and **FP16 compute** preserve run03's quantization/compute choice. Trainable adapter parameters use FP32, with dynamic loss scaling. AMP overflow retries the same batch without advancing optimizer/scheduler. Nonfinite loss or persistent nonfinite gradients fail explicitly.

Epoch-zero data order reconstructs the original Trainer's seeded sampler. Original emitted sampler indices are unavailable, so this is not a bitwise historical replay. The software/GPU differ from the old T4 run; A is a historical control, and small A–C/D differences cannot be attributed exclusively to rank. A's historical training time/VRAM are left blank rather than compared as if measured on Ada.

## Short speed selection and separate C/D estimates

A bounded common-backend search runs disposable rank64 warmup/memory checks plus one measured effective update per candidate. This ensures the selected method fits D before B starts. It compares available HF SDPA / FlashAttention / Unsloth, microbatch2/4, and one no-checkpoint candidate. All are 4bit QLoRA. The fastest passing configuration is frozen for B/C/D. A short loss/gradient audit checks reduction semantics. Padding is retained: cross-document packing changes are excluded to minimize validation time and preserve the historical objective.

After **B training and final dev400 evaluation**, C and D each get their own fresh, disposable benchmark: longest-batch warmup, then three real effective batches near the 17th/50th/83rd percentiles of token count. Benchmark updates never enter the final adapters. Each JSON/log/console report includes seconds/update, supervised tokens/sec, peak VRAM, and **approximate training-only time for 375 updates**. The min/max-sample projection is a descriptive range, not a confidence interval. Setup, audit, warmup, checkpoint saving, merge and evaluation are excluded from the estimate. This short sample can misestimate full-epoch time.

C then D are trained and evaluated, final checkpoints only. Each update prints a cumulative measured training ETA and writes `remaining_training_seconds`. `phase_times.jsonl`, `setup_time.json`, and training `wall_seconds` separate overhead from optimizer-update timing. Checkpoint saving is included in wall time, not `train_seconds`.

## Pull and background launch (remote RTX 6000 Ada 48GB)

```bash
cd "$HOME/geniac-remote"
git pull --ff-only origin main
# If using a sparse checkout, include both helper and new experiment directories:
if [ "$(git config --bool core.sparseCheckout)" = true ]; then
  git sparse-checkout add remote/math_training remote/stage1_rank_ablation
fi
mkdir -p "$HOME/geniac_math"
RANK_ID="rank_$(TZ=Asia/Tokyo date +%Y%m%d_%H%M%S)"
printf '%s\n' "$RANK_ID" > "$HOME/geniac_math/latest_rank_id.txt"
nohup bash remote/stage1_rank_ablation/launch.sh "$RANK_ID" \
  --reference "$HOME/geniac_math/runs/ada_20261016_221122" \
  > "$HOME/geniac_math/$RANK_ID.console.log" 2>&1 < /dev/null &
echo "PID=$! RUN=$RANK_ID"
```

Omit `git sparse-checkout add` for a normal full checkout. The launcher uses Python3.10–3.12 (or provisions 3.11), creates a separate training environment and reuses the established evaluation environment read-only. It attempts pinned optional Unsloth and a binary-only FlashAttention wheel; unavailable accelerators are skipped, not compiled for hours. An incompatible existing evaluation environment fails instead of silently changing it; set `GENIAC_EVAL_PYTHON` to a compatible executable if necessary. Allow approximately 80 GiB of free disk for parent/merged models and resumable adapters.

An existing workspace launch lock prevents overlap with an active old suite. Stop that suite separately before starting this one if it is still running. This launcher does not terminate other work.

A's saved outputs are reused only when model, questions, generation settings, known implementation and package metadata match. Legacy greedy runs with a different `max_num_seqs` (including the old four-generation configuration) are conservatively re-inferred **once**, never retrained. A previous matching new experiment's `evaluation/A` directory is also accepted by `--reference`.

## Resume and inspect

After the previous process has exited, rerun the same ID. It uses the immutable original code snapshot, restores adapter/optimizer/scheduler/scaler/RNG state from the last 25-update checkpoint, and skips completed training/evaluation. At most 24 unsaved updates repeat after an abrupt kill. Repeated pulls do not change a running or resumed experiment. Changed package/config/data signatures are rejected.

```bash
cd "$HOME/geniac-remote"
RANK_ID="$(cat "$HOME/geniac_math/latest_rank_id.txt")"
nohup bash remote/stage1_rank_ablation/launch.sh "$RANK_ID" \
  --reference "$HOME/geniac_math/runs/ada_20261016_221122" \
  >> "$HOME/geniac_math/$RANK_ID.console.log" 2>&1 < /dev/null &
echo "PID=$!"
```

```bash
RANK_ID="$(cat "$HOME/geniac_math/latest_rank_id.txt")"
RUN="$HOME/geniac_math/rank_ablation/$RANK_ID"
tail -f "$HOME/geniac_math/$RANK_ID.console.log"
# Once C/D benchmarks finish:
cat "$RUN/benchmarks/C_estimate.json" "$RUN/benchmarks/D_estimate.json"
# At completion (also written on a trial failure):
cat "$RUN/report.json"
cat "$RUN/comparison.csv"
```

Outputs: `conditions/{B,C,D}/final/` adapters; `evaluation/{A,B,C,D}/generations/b0_eval.json` full raw responses/tokens/grades; `comparison.csv`, `report.json`, `question_changes.csv`, `comparison.png/.svg`. All six model pairs get paired-bootstrap 95% intervals (2,000 resamples, same fixed questions); this measures question sampling uncertainty, not inference-execution variability. These are public dev scores, not official competition scores.

## Local checks

```bash
python -m pytest -q remote/stage1_rank_ablation/tests
```

CPU tests cover A training prohibition, quantization-only candidates, sampler coverage, causal-loss/gradient equivalence across microbatches, and paired statistics. Real local cache tokenization was verified against the recorded run03 counts. Actual CUDA throughput, backend compatibility and memory fit must pass the remote short benchmarks; they cannot be measured on this Mac.
