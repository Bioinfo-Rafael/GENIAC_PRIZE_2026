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
  git sparse-checkout set remote/math_training remote/stage1_rank_ablation
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

## Setup compatibility recovery

TorchAO is pinned to 0.13.0 for Torch 2.8.0; einops is explicit for the FlashAttention binary. Setup now tests Transformers/PEFT imports in addition to `pip check` before declaring the environment ready. This fixes the `ScalingType` ImportError caused by TorchAO 0.18.0.

If an older run failed during `parents`, pull this fix and launch with a **new run ID**. The old run has immutable code and package snapshots; changing its environment and resuming would correctly fail its package-identity check. No training has happened at `parents`; retain the failed directory for diagnosis. The new run reuses already downloaded inputs and Hugging Face model cache.

<a id="results-2026-10-21"></a>
## 実測結果（2026-10-21 JST）

出典：ユーザーが共有した本実験の最終 `report.json`。固定public dev400問（PRM800K 300問、GSM8K 100問）、最終checkpointで比較。Aは既存run03であり再学習なし。公式提出スコアではない。以下のCは本実験のrank32条件で、過去の別実験の「C」と区別する。

| 条件 | Stage 1 | rank / alpha | 正解数 | 正答率 | GSM8K | PRM800K | 正解かつ正常終了 |
|---|---|---|---:|---:|---:|---:|---:|
| A | あり | 8 / 16 | 251/400 | 62.75% | 84/100 | 167/300 | 191/400 |
| B | なし | 8 / 16 | 236/400 | 59.00% | 74/100 | 162/300 | 103/400 |
| C | あり | 32 / 64 | 243/400 | 60.75% | 84/100 | 159/300 | 130/400 |
| D | あり | 64 / 128 | 253/400 | 63.25% | 86/100 | 167/300 | 114/400 |

| 条件 | 反復数（率） | 非正常終了数（率） | unknown | 平均生成トークン |
|---|---:|---:|---:|---:|
| A | 123（30.75%） | 124（31.00%） | 26 | 760.8125 |
| B | 232（58.00%） | 250（62.50%） | 27 | 1374.8625 |
| C | 216（54.00%） | 224（56.00%） | 25 | 1221.8750 |
| D | 249（62.25%） | 259（64.75%） | 23 | 1389.7200 |

反復は同じ10語列が4回以上出現する回答、非正常終了は `finish_reason != 'stop'`。どちらも誤答と同義ではなく、正解の回答にも含まれる。unknownは抽出失敗・採点エラー・gold解析不能のいずれかを含む件数。正解かつ正常終了は補助指標であり、公式スコアではない。

| 条件 | 更新時間（秒） | 学習wall時間（秒） | ピークVRAM（bytes / GiB） | 教師target tokens/sec |
|---|---:|---:|---:|---:|
| A | 未計測 | 未計測 | 未計測 | 未計測 |
| B | 709.3571 | 716.7102 | 5,360,830,464 / 4.99 | 2245.06 |
| C | 721.9293 | 736.7361 | 5,771,679,744 / 5.38 | 2205.97 |
| D | 743.3908 | 769.7548 | 6,265,473,024 / 5.84 | 2142.28 |

更新時間は学習更新だけ、wall時間は学習ループとcheckpoint保存を含むが、モデル初期化・データ準備・merge・評価は含まない。B/C/D合計の更新時間は約36.24分で、実験全体の所要時間ではない。DはCより更新時間が約3.0%増えた。GPUはRTX 6000 Ada 48GB。最終選択backend・microbatchは今回のreportに含まれないため、この表から推測しない（`selected_backend.json`で確認可能）。

### 問題ごとの変化とpaired bootstrap

差は「右の条件 − 左の条件」、単位はパーセントポイント。2,000回の問題単位resamplingによる95%区間。誤→正／正→誤も左から右への変化。

| 比較 | 正答率差 | 95%区間 | 誤→正 | 正→誤 |
|---|---:|---:|---:|---:|
| A → B | −3.75 | [−7.50, 0.00] | 23 | 38 |
| A → C | −2.00 | [−5.50, +1.50] | 22 | 30 |
| A → D | +0.50 | [−3.00, +4.00] | 28 | 26 |
| B → C | +1.75 | [−2.00, +5.50] | 36 | 29 |
| B → D | +4.25 | [+0.25, +8.25] | 42 | 25 |
| C → D | +2.50 | [−0.75, +5.75] | 26 | 16 |

### 解釈と今回の判断

- **Aを維持する。** Dは正答数が最大だがAとの差はわずか2問。反復率はAの30.75%から62.25%、非正常終了率は31.00%から64.75%へ悪化し、生成長は約1.83倍。Aより良いモデルになったと判断する根拠は不足している。
- **rankを増やすだけの正答率改善は確認できない。** CはAより8問少なく、Dは2問多い。A対C・Dの区間はいずれも0を含む。これは「rankに効果がない」と証明したことではなく、このデータ・学習条件・400問では改善を確認できなかったという結果。
- **Stage 1は有効である可能性を支持するが、単独効果は未確定。** 同rankのA対BはAが15問多く正解し、終了挙動も良い。ただしAは旧環境で学習された既存モデルであり、backendやライブラリなどの差が残る。今回Aを再学習しない設計上の限界。
- B対Dのみ未補正95%区間が0を上回るが、Stage 1とrankの両方が変わり、6比較の多重性も補正していない。rank単独の効果として扱わない。
- D対Aのデータセット別の正答数差はGSM8Kが+2、PRM800Kが0。ただしPRMの同じ問題が全て同じ正誤という意味ではない。全体では28問改善・26問悪化が相殺されている。
- 信頼区間は問題の再抽出による不確実性のみを扱い、学習seedや推論実行の変動は含まない。本結果から公式提出スコアの上昇は推定しない。
