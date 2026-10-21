# Rank64: 6,000 → 12,000例の数学SFT

新規学習は **R12の1条件だけ**。既存 `stage1_rank_ablation` のDから高速化設定とStage 1親モデルを再利用し、数学LoRAは新規初期化する。Dの数学adapterを継続学習しない。A・Dとの比較の主指標は正答率で、反復・生成長・終了率を採用条件にしない。devが悪くても今回のモデルの提出JSONを生成する。Omnicampusには送信しない。

## 実行結果（2026-10-21）

R12の本番スコアはユーザー報告で **0.12394**。run IDは `rank64_12k_20261021_183439`。devでも既存Dを超えず、今回の12,000例への増量で正答率改善は確認できなかった。

| 条件 | dev正解 / 400 | GSM / 100 | PRM / 300 | 反復率 | non_stop率 | 平均生成token |
|---|---:|---:|---:|---:|---:|---:|
| A：rank8、6,000例 | 251（62.75%） | 84 | 167 | 30.75% | 31.00% | 760.81 |
| D：rank64、6,000例 | 253（63.25%） | 86 | 167 | 62.25% | 64.75% | 1,389.72 |
| R12：rank64、12,000例 | 248（62.00%） | 85 | 163 | 64.25% | 68.25% | 1,451.76 |

R12−Dは−1.25ポイント、R12−Aは−0.75ポイント。R12のpaired bootstrap・学習時間・VRAMは今回受領した提出JSONには含まれず、リモートの `report.json` を参照する。本番とdevは別指標であり、Dの本番成績は未確認。

提出metadataから、実際の高速化設定は **Unsloth / NF4 / attention auto / microbatch2 / gradient checkpointing有効 / packingなし**と確認。その他は下記レシピ。提出JSONは形式チェックのerror/warning/violationが全て0だった。

提出232回答では反復145件、実効上限到達156件、平均1,573.81 token。数学的に改善した例もあるが、以前の正しい最大公約数を誤答へ変える、正しい中間式から最終回答の係数を落とす、正答後に矛盾を反復する例が確認された。過去のfix7対象7件にも数学的誤りが残った。公式の問題別採点を得たわけではない。

[提出回答の詳細監査](../../research/20261021_rank64_12k_submission_audit.md)・[集計JSON](../../research/20261021_rank64_12k_submission_metrics.json)。問題ファイルは今回読まず、提出回答の計算と挙動を検査した。回答を編集せず、検査した本番例を学習には使用しない。

## 学習レシピ

| 項目 | 設定 |
|---|---|
| Base | Qwen/Qwen3-4B-Base、revision `906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| 親モデル | 既存run01のStage 1 adapterをFP16でmerge済みのモデル（Dの保存物を読取専用で参照） |
| Stage 2 | 新規LoRA、rank64 / alpha128、dropout0、biasなし |
| 対象modules | q/k/v/o_proj、gate/up/down_proj |
| 教材 | PRM800K 9,000例 + GSM8K 3,000例、run04と同じ固定revision・選択関数 |
| epoch / 更新 | 1 / 750 |
| LR / 実効batch / 最大系列長 | 5e-5 / 16 / 2,048 |
| optimizer | PagedAdamW8bit、betas(.9,.999)、eps1e-8、weight decay0 |
| scheduler | cosine、warmup比率.03 → **23更新**（Dは12更新。同じ比率を750更新へ適用） |
| loss | promptを含む全shifted nonpadding token。16例のtarget数で正規化 |
| 量子化 / 計算 | NF4 double quantization / FP16、学習パラメータFP32、動的loss scaling |
| seed / clipping | 42 / gradient norm1 |
| packing | なし |
| backend / microbatch / checkpointing | Dの `selected_backend.json` をそのまま固定。新規ベンチマークなし |

`run04_recipe.py` はrun04 notebookの関数をそのまま抽出したもの。GSMの `####` 分割・`<<...>>` 除去、boxed補完、固定chat template、末尾EOS、hash順選択・重複排除・全文長フィルタを引き継ぐ。run04のheldoutと固定dev/finalの全800問を除外する。今回この除外集合が同じであることも確認した。推論用問題は教材選択に使用しない。

公開教材によるローカル実測：**12,000例、3,235,460トークン、最長1,974、教師target 3,223,460**。全例EOSで終わり、既存run03/Dの6,000例を含むことを検査する。seed42のsamplerで1 epochを750個の16例groupへ分割する。長い例を切り詰めない。

## 再利用と実行順

1. 指定された既存rank実験のDが完了済みであること、コード・入力・親モデル・設定・実際の学習環境が対応することを検査。
2. 公開12,000例を準備し、A・Dの回答がdev質問・順序・モデルhash・採点コード・推論設定・package版と一致すればコピーして再利用。
3. `stage1_rank_ablation.engine.train` をimportし、R12を750更新で学習。optimizer/loss/AMP/再開処理は共有する。
4. 最終adapterだけmerge・dev400評価。比較対象の条件が一致しない場合だけA/Dも新しい出力先へ再推論する。元の保存物は変更しない。
5. 正答率、GSM/PRM別成績、全問題の正誤変化、A→R12・D→R12のpaired bootstrap 95%区間（2,000再標本化）、時間・throughput・VRAM・反復・正常終了・生成長を保存。
6. そのmerge済みR12を再利用して提出JSON生成・公式形式チェック。提出だけ失敗しても学習や完了済みdev推論は繰り返さない。

Aは旧環境で学習した既存モデル、Dは同じ高速化設定・環境の6,000例モデル。epoch固定なので12,000例条件は教師トークン・更新数も約2倍になる。データ量だけを同一計算量で比較する実験ではない。信頼区間は問題の再標本化に限られ、学習・推論の実行間変動を含まない。

## 提出処理

配布 `notebook2_inference.ipynb` の設定・推論・保存セル、`submission.py` / `launch_submission.sh`、公式 `inference_config.json` / `evalset.py` / `check_submission.py` を調査した。公式message生成関数を変更せず抽出して `official_prompt.py` に保持し、`llm.chat` に渡す。model/tokenizerは学習時と同じ固定templateを使用。1問1回答・greedy、best-of-Nも再ランキングもなし。実際のprompt token IDsの一致を検査し、回答の空白を含め出力をそのまま保存する。

配布物のconfigはモデル指定・vLLMパラメータ変更を許可している。今回の提出設定は **context8192、全task max_tokens2048、temperature0、seed42、chunk32、FP16、GPU memory0.8**。公式既定のolympiad上限4096から2048への変更を含むため、既定値の完全再現とは称さない。単一user turn・system指示なし・JSON形式は保持する。入力を切り詰めず、収まらない場合は停止する。推論環境は既存評価用venvを再利用し実際のpackage版を記録する。

共有 `submission.generate` に明示的なモデルpath・recipe・run ID・Stage 1名・message builderを渡すAPIを追加した。旧C-suite CLIの既定動作は保持するが、今回の提出metadataは **run01 Stage 1 + fresh rank64 / PRM9000 GSM3000 / 750更新** を記録し、C-suiteのモデル指定・履歴を流用しない。

推論用データの取得・読み込み・推論・形式検査はリモートプロセス内だけで実施する。Codexで実データをロードしていない。推論phaseのstdout/stderrは権限を限定した `private_logs/` へ保存し、メインconsoleには問題文・回答・例外の本文を転送しない。**`questions.json`、`partial.json`、提出JSON、private logをチャットへ貼らない。** 認証が必要なら、既に利用条件へ同意済みのHugging Faceアカウント/tokenをリモート内で利用する。

## Pullからバックグラウンド起動まで

```bash
(
  set -e
  cd "$HOME/geniac-remote"
  git pull --ff-only origin main
  if git sparse-checkout list >/dev/null 2>&1; then
    git sparse-checkout set remote/math_training remote/stage1_rank_ablation remote/rank64_12k
  fi
  SOURCE_RANK="$HOME/geniac_math/rank_ablation/$(cat "$HOME/geniac_math/latest_rank_id.txt")"
  test -f "$SOURCE_RANK/conditions/D/complete.json"
  NEW_ID="rank64_12k_$(TZ=Asia/Tokyo date +%Y%m%d_%H%M%S)"
  printf '%s\n' "$NEW_ID" > "$HOME/geniac_math/latest_rank64_12k_id.txt"
  nohup bash remote/rank64_12k/launch.sh "$NEW_ID" --source-rank "$SOURCE_RANK" \
    > "$HOME/geniac_math/$NEW_ID.console.log" 2>&1 < /dev/null &
  echo "PID=$! RUN=$NEW_ID"
)
```

`latest_rank_id.txt` が比較対象以外を指す場合は、`SOURCE_RANK` を完了したA〜D実験のパスに置き換える。今回の既存ログは `rank_20261021_165303`。Dの学習venvは再installせず、その保存済み `packages.json` と不一致なら停止する。提出データ用venvがない場合だけ既存 `requirements-submission-data.txt` で別途作成する。古いGitに合わせ `sparse-checkout set` を使用。旧実験と同じworkspace lockで二重起動を防ぐ。

```bash
NEW_ID="$(cat "$HOME/geniac_math/latest_rank64_12k_id.txt")"
tail -f "$HOME/geniac_math/$NEW_ID.console.log"
```

`R12 step=.../750` に経過学習時間と残り推定時間が出る。準備・merge・評価・提出時間は別途 `phase_times.jsonl` に保存する。

## 再開

前のprocessが終了した後、同じIDで起動する。元のコードsnapshotを保持し、学習は25更新ごとのcheckpointからadapter・optimizer・scheduler・scaler・RNGを復元する。突然の停止では最大24更新を再実行する。完了した学習・dev推論・提出回答は再生成しない。

```bash
(
  set -e
  cd "$HOME/geniac-remote"
  NEW_ID="$(cat "$HOME/geniac_math/latest_rank64_12k_id.txt")"
  SOURCE_RANK="$HOME/geniac_math/rank_ablation/$(cat "$HOME/geniac_math/latest_rank_id.txt")"
  nohup bash remote/rank64_12k/launch.sh "$NEW_ID" --source-rank "$SOURCE_RANK" \
    >> "$HOME/geniac_math/$NEW_ID.console.log" 2>&1 < /dev/null &
  echo "PID=$!"
)
```

## 結果

出力先：`$HOME/geniac_math/rank64_12k/$NEW_ID/`

- `README.md`, `report.json`, `comparison.csv`：実測値・比較。正答率を主指標に記録。
- `question_changes.json/.csv`：A/DからR12への同じdev問題の正誤変化。
- `conditions/R12/final/`：新規学習した数学LoRA。
- `evaluation/{A,D,R12}/generations/b0_eval.json`：public dev回答。
- `submission/submission.json`：公式形式の未編集回答。`submission/complete.json` と成功した `format_check.json` があることを確認する。
- `submission/manifest.json`, `generation_profile.json`, `hashes.json`：実験・モデル・設定・出力のhashと履歴。
- `status.json`：`complete`なら提出ファイル生成まで完了。Omnicampusへの送信は別途ユーザーが行う。

```bash
NEW_ID="$(cat "$HOME/geniac_math/latest_rank64_12k_id.txt")"
RUN="$HOME/geniac_math/rank64_12k/$NEW_ID"
cat "$RUN/status.json"
cat "$RUN/report.json"
cat "$RUN/comparison.csv"
```

実行後の実測結果は上記runディレクトリのREADMEへ自動保存する。本READMEの「実行結果」はユーザー報告のdev成績・本番スコアと、取得済み提出JSONの監査で追記した。ローカルでは公開教材の選択実測と合成データのテストを行い、コンペ問題を使った試験はしていない。

## ローカル検証記録

公開教材を実取得して上記12,000例・トークン数・元6,000例包含を検証し、準備処理の再実行ではcache再利用を確認した。合成データによる17テスト（既存提出処理・rank比較の回帰テストを含む）が通過。選択・重複/heldout排除、750更新、公式message builderの一致、回答無加工・1問1回答、正しいmetadata、完了提出の再生成防止、正答率優先の集計を検証した。CUDA学習と実際のコンペ推論はローカルでは実行していない。

```bash
python -m pytest -q remote/rank64_12k/tests remote/math_training/tests/test_submission.py remote/stage1_rank_ablation/tests
```
