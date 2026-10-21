# E2 形式対応SFT / E3 オンライン正誤報酬RL

run03（Qwen3-4B-Base + run01 Tulu Stage1 + rank8数学LoRA）を独立した親とする実験。E1/D/R12の重み・結果は更新しない。GPU実測と学習・400問評価は**未実行**。180分は処理を打ち切る上限であり、全候補の学習・評価完了や性能改善の保証ではない。

## リモートでpullしてバックグラウンド実行

既存の `$HOME/geniac-remote` / `$HOME/geniac_math` と、既存の train/eval/submission-data venvを利用する。パッケージを再インストールしない。HF認証とモデルキャッシュ、固定した独立400問、run03親モデルが準備済みであることが前提。空きディスク45GiB以上。必要に応じ `GENIAC_TRAIN_PYTHON` / `GENIAC_EVAL_PYTHON` / `GENIAC_DATA_PYTHON` を指定できる。

```bash
(
  set -e
  cd "$HOME/geniac-remote"
  git pull --ff-only origin main
  if git sparse-checkout list >/dev/null 2>&1; then
    { git sparse-checkout list; printf '%s\n' remote/e2_e3 remote/math_training remote/public_eval_diagnostics remote/rank64_12k; } |
      sort -u | git sparse-checkout set --stdin
  fi
  test -f "$HOME/geniac_math/runs/ada_20261016_221122/models/b0/artifacts.json"
  test -f "$HOME/geniac_math/diagnostics/rank64_12k_20261021_183439_public_eval_v1/independent_dev400.json"
  E23_ID=e2e3_20261021_v1
  E23_LOG="$HOME/geniac_math/$E23_ID.console.log"
  nohup bash remote/e2_e3/launch.sh "$E23_ID" >> "$E23_LOG" 2>&1 < /dev/null &
  echo "PID=$! LOG=$E23_LOG"
)
```

E1等が動いている場合、共通 `.launch.lock` / `.math_training.lock` またはGPUプロセス検査で停止する。既存ジョブをkillせず、終了後に同じコマンドを実行する。別workspaceのGPUジョブも検出するが、起動後にロックを守らず起動された外部ジョブとの競合までは排除できない。

初回コードを `e2_e3_runtime/RUN_ID/code/` に保存し、再開もそのコードを使用する。同じ起動コマンドで `--resume` を自動付与。run IDは `e2e3_` 必須。入力・コード・モデル・環境が変わる再開は拒否する。実験を意図的に新しく開始するときだけ別run IDを指定する。

run03の検証済み提出ディレクトリがある場合：

```bash
nohup bash remote/e2_e3/launch.sh e2e3_20261021_v1 \
  --baseline-submission /absolute/path/to/run03_submission_directory \
  >> "$HOME/geniac_math/e2e3_20261021_v1.console.log" 2>&1 < /dev/null &
```

`submission.json`, `complete.json`, `manifest.json` が必要。重みのartifact hash・run03 adapter hash・JSON hashと公式checkerが一致する場合だけ再利用する。`$HOME/geniac_math/submissions/*` も自動探索。出自を検証できないColab JSONを、ファイル名だけでrun03と見なすことはしない。

## 時間と停止

元の180分配分（合成25 / SFT15 / E2評価20 / RL45 / E3評価20 / 選択・提出40 / 予備15分）を上限として扱う。初期検証は予備から最大10分。全phaseの実測秒数・残り秒数・2秒間隔のデバイスVRAM最大値を `phase_times.jsonl` に保存。学習worker自身のピークVRAMも記録する。phaseの所要時間はロード・merge・採点を含む。

**提出を最優先するための順序上の例外**：run03 JSONを再利用できなければ学習より先に公式推論し、`fallback_submission/` に確保する。この時間は選択・提出40分に算入する。既にJSONが確保されてからE2→E3→最終評価→最終JSONの順で処理する。最終候補に改善がなければ、このrun03 JSONを再推論せずコピーする。候補の公式推論が失敗・時間切れでもrun03を残す。公式データの内容は学習・評価・選択ロジックへ渡さない。

合成の最初の16件、SFTの最長batch 1更新、RLの第1ラウンド、各評価の最初の8問で速度を測る。測定値の1.5倍と起動余裕を用いて、残り処理が収まるか判断する。追加RLをまず省略し、合成の追加batchも期限前に終了。補助評価は実装しない。E2は全1 epochが収まらなければスキップする。E3の16ラウンドが収まらなければ第1ラウンドのcheckpointを残し、候補として採用しない。未完の400問評価を400問の成績として使わない。

学習・通常評価は実行開始140分まで。最後の40分を最終処理に予約し、180分より30秒前に子プロセスへSIGTERM、最大25秒後にそのプロセスグループだけSIGKILLする。既存のcheckpoint・採点済み問題・合成batchは保存済み地点から再開。処理量は速度で調整するが、測定より極端に遅い処理、GPU/ディスク障害、OSの強制終了等の下でJSON完成は保証できない。run03推論だけで40分以内に完了しない場合は学習を開始せず、部分推論と理由を保存する。

`clock.json` の締切は再開しても延長しない。実行間の停止時間もwall-clockに含む。期限後は追加GPU処理を開始せず、確保済みJSONを最終成果物として残す。`partial_with_submission` は提出可能だが省略phaseがある状態。完了後の同一run IDは再実行しない。

## E2

許可ソースのうちSHA256・revision固定のGSM8K trainのみ使用する（PRM800K/LIMOは今回は使用しない）。元解説の `<<expression=result>>` を先頭から機械的に検査し、1原問題につき1式を抽出。既存dev/finalと既定の独立400問をexact/near-duplicate検査で除外する。固定seed順の最大900式からrun03が回答を生成し、検証済み300例を目指す。

- JSON（expression/value）、単一等式、欠けた右辺をboxedで回答する3形式。テンプレートは一般的な形式指示のみで、数学的な問題・正解は原文から自動導出する。
- 生成モデルは共有run03だけ。外部モデル/API・人手選別・本番問題は使用しない。
- 入力式と原文結果の同値性、生成回答の厳密な形式、生成結果の同値性を `Fraction` と制限付きASTで検証。任意コード実行・自由な数式パーサは使わない。余分な説明、変数、未対応の演算、検証不能例は不採用。
- 狭い算術の形式変換であり、一般的な証明生成の品質を保証するものではない。32例未満なら学習をスキップ。生成数・採用数・除外理由・元問題ID・元解説hash・実生成応答を保存する。
- run03 rank8 adapterの継続更新、回答部分のみNLL、実効batch16、1 epoch。既存workerのAdamW・cosine scheduleを再利用し、初期/最大LR 5e-6、BF16/NF4、maxlen4096。benchmarkの更新は本学習へ引き継がない。

## E3と既存方式との差分

E2の重みを参照せず、共有Stage1＋オリジナルrun03 Stage2 adapterから開始。固定seedのGSM8K train最大512問から4問×8回答/round。既存graderで最終回答を判定し、正解1・不正解0。オンライン生成はtemperature1/top_p1/top_k0、学習と同じ量子化policy、dropoutなし。

`online_worker.py` の既定動作は維持し、`dr_normalization=true` のときだけ次に切り替える。

```text
従来: advantage = (r - group_mean) / group_std
      loss = -sum(advantage * sequence_logprob / response_length) / batch_completions
E3:   advantage = r - group_mean
      loss = -sum(advantage * sequence_logprob) / (batch_completions * 2048)
```

2048は固定最大生成長。同一正誤グループのadvantageは0で、分母には含める。混在がないroundではoptimizerを更新しない。rolloutは更新前に保存し、round終了時にoptimizer/RNG/checkpointを保存。reward一覧・平均、混在group数、累積/当該roundの実optimizer更新回数、loss、各生成長、時間・VRAMを記録。

[Dr. GRPOの論文](https://arxiv.org/abs/2503.20783)の報酬標準偏差・応答長正規化の除去を参考にした実装。**論文実装の完全再現ではない**。既存の1 rollout batchにつき1回のon-policy REINFORCE更新を使い、PPO clipping・複数epoch・参照KLはない。LR 1e-6。16roundのdev400正答数がAより多い場合だけ、時間が許せば32roundへ進む。16/32のcheckpointを記録し、devで強いE3 checkpointが最終候補になり得る。

## 評価と選択

各候補とAを同じ固定dev先頭128問で予備評価。既存paired bootstrap 2,000回/seed42の差の95%CI上端が0未満なら明確な悪化として除外し、それ以外は400問で評価。8問の速度測定もこの同じsplitの先頭部分であり、別の評価セットではない。完了済み生成は問題ID単位で再利用する。

全dev400の正答数で比較し、同点はA→E2→E3-16→E3-32の順。生成長・反復率だけで落とさない。選択を `selection.json` に固定してから、その1候補とAだけを独立400問で比較。95%CI下端が0より大きい場合のみ改善確認とし、それ以外はrun03。結果を見て別候補を選び直さない。Aがdev勝者なら自己比較は省略する。独立splitは既存の固定hashを検査し、新しいsplitを生成しない。

全体/データセット別正答率、生成長、反復率、正常終了率は `suite_report.json`。問題単位の正誤変化は同JSONと `question_changes.csv`。繰り返し使用されたdevの探索バイアスと実行ばらつきはbootstrapに含まない。

## 公式JSONと結果確認

既存 `submission.py`、rank64_12kの配布message builder、公式inference_configとcheckerをimport。単一モデル・greedy・1問1回答、配布prompt、float16/maxlen4096/max_tokens2048/chunk58。生成回答は文字列の空白も含め編集しない。best-of-N/再ランキング/送信はしない。

```bash
tail -f "$HOME/geniac_math/e2e3_20261021_v1.console.log"
# 別ターミナル、またはtailをCtrl-Cで終了して確認
E23_OUT="$HOME/geniac_math/suites/e2e3_20261021_v1"
cat "$E23_OUT/status.json"
cat "$E23_OUT/adoption.json"
cat "$E23_OUT/submission/format_check.json"
```

| 保存先（`E23_OUT`以下） | 内容 |
|---|---|
| `submission/submission.json` | 最終提出用JSON |
| `submission/format_check.json` | 公式形式チェック。errors/violationsが空であること |
| `submission/manifest.json`, `hashes.json`, `complete.json` | モデル・設定・コード・提出ファイルhash/出自 |
| `fallback_submission/` | 先に確保したrun03提出。中断中でもこちらを確認可能 |
| `suite_report.json`, `status.json`, `clock.json`, `phase_times.jsonl`, `logs/` | 状態・失敗/省略理由・実測時間・VRAM・ログ |
| `data/synthesis_complete.json`, `synthesis_state.json` | 合成・検証の集計とbatch単位再開状態 |
| `training/E2/`, `training/E3/` | 独立したcheckpoint・学習ログ・RL rollout |
| `evaluations/`, `question_changes.csv`, `selection.json`, `adoption.json` | 評価・問題別変化・固定候補・最終採用 |

ローカル検証：

```bash
python -m pytest -q remote/e2_e3/tests remote/math_training/tests/test_accuracy.py remote/math_training/tests/test_submission.py
bash -n remote/e2_e3/launch.sh
```

CPUの単体/制御フローテストと小型モデルの既存RL回帰テストを実施。RTX 6000 Ada上での速度・VRAM・実スコアは起動後の記録で確認する。
