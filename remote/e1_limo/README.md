# E1: run03からLIMO混合SFTを1条件実行

既存`math_training`のrank8 adapter継続・NLL・保存/再開・vLLM推論・grader・paired bootstrapを再利用する。E2/E3、seed違い、推論設定違いの探索は実行しない。既存run03・過去の実験ディレクトリは変更しない。

## 固定条件と選択規則

- run03 = 共通Stage 1 + 既存rank8/alpha16数学adapter。新しいadapterを初期化しない。
- 許可済みpinは`math_training/accuracy_data.py`のPRMと`published_sources.json`のGSM8K/LIMO。取得時SHA検証、HFキャッシュ・既存PRMファイル再利用。
- 原文回答 + EOS、promptはmask。最大8192、長い回答は例ごと除外し切断しない。GSMの本文変換もしない。
- 教師トークン比PRM/GSM/LIMO = 約60/20/20（残りPRM:GSM=3:1もトークン比）。正規化問題・大文字小文字/空白・5-gram Jaccard 0.85で機械的重複除去。cross-source重複はLIMO→PRM→GSMの固定順で優先し、評価問題を必ず除外。評価正解やモデル正誤による教材選別なし。
- 既存dev/final、新独立400問、存在すれば過去accuracy suiteのconfirmation/finalも除外。基盤モデル事前学習やStage 1との完全な分離は保証しない。
- 初期LR 5e-6、既存のcosine scheduler/warmup5%、AdamW/weight decay .01、effective batch16、microbatch1、seed42。BF16/NF4、gradient checkpointing、PyTorch SDPA（対応時のflash kernel）を利用。Unsloth導入や既存環境更新はしない。
- 教師60万。1例最大3回。20万/60万を初めて超えた**完全な回答例の境界**で更新グループを閉じて保存。実量と比率は`data/data_report.json`。途中の教師例は切らず、少数トークン超過を許容。
- A/E1-200k/E1-600kを既存dev400で比較。正解数最大を選び、同点は200k。反復率/生成長は選択に使わない。
- 選択を`selection.json`へ凍結してから、作成済み独立400問をAと勝者のみで評価。落選checkpointの独立評価は行わない。
- 採用は **devでAを厳密に上回り、独立400問でpaired bootstrap 95%区間の下限>0** の場合のみ。それ以外はA。2,000再標本化、seed42。正解率以外の救済基準や再選択はない。
- 全内部評価は同じgreedy・max_tokens2048・context8192・chunk8・既存grader。公式scoreや過去の本番回答を選択に使わない。

## リモートで実行

ローカルではCPUテストのみ。GPUの学習時間・VRAM・改善量は実測前であり、結果は未生成。

以下をリモート端末へ貼り付ける。既存の3環境（`.venv-train` / `.venv-eval` / `.venv-submission-data`）とHF認証をそのまま使用する。追加installなし。既存GPUジョブがあればロックにより拒否し、停止させない。

```bash
(
  set -e
  cd "$HOME/geniac-remote"
  git pull --ff-only origin main
  if git sparse-checkout list >/dev/null 2>&1; then
    { git sparse-checkout list; printf '%s\n' remote/e1_limo remote/math_training remote/public_eval_diagnostics remote/rank64_12k; } |
      sort -u | git sparse-checkout set --stdin
  fi
  test -f "$HOME/geniac_math/runs/ada_20261016_221122/context.json"
  test -f "$HOME/geniac_math/diagnostics/rank64_12k_20261021_183439_public_eval_v1/independent_dev400.json"
  E1_ID=e1_limo_600k_v1
  nohup bash remote/e1_limo/launch.sh "$E1_ID" \
    >> "$HOME/geniac_math/$E1_ID.console.log" 2>&1 < /dev/null &
  echo "PID=$! LOG=$HOME/geniac_math/$E1_ID.console.log"
)
```

同じコマンド・IDで再開する。初回のコードを`e1_runtime/<ID>/code/`に保存し、後日のpullが実行中/再開コードを変えない。別の実験として開始するときだけIDを変える。入力パス変更は初回起動で`--source-run … --independent …`を追加できる。独立評価は既存セットのdigest `7eafe2c0c46678af1483ea724948ccd394914771eb8362d4990ef4e91cf87d70`との一致を必須とする。

```bash
tail -f "$HOME/geniac_math/e1_limo_600k_v1.console.log"
tail -f "$HOME/geniac_math/suites/e1_limo_600k_v1/logs/E1_train_rft_eval_128.log"
```

最長1更新のメモリ検査と代表3更新の短いbenchmarkのみ実施し、`ESTIMATED_TRAINING_MINUTES`を学習前に出力する。モデルreload・保存・評価・提出時間はこの予測に含まない。benchmark重みは捨て、学習はrun03から開始する。学習ログに経過/残り秒、教師tokens/sec、allocatorピークVRAMを記録。経過/ETAは再開後のセグメント基準、phase_timesは各起動の全時間を保持する。

SIGTERM/SIGINT時は更新完了境界でoptimizer/scheduler/RNGを保存して停止。20万/60万にも再開stateを保存。強制kill/電源断では最後に完成したstateまで戻る。通常は余分な定期checkpointを生成しない。GPUロックを保持したまま単一ジョブとして逐次実行する。

## 再利用と出力

過去の公開LIMO suiteはGSM/LIMO約50:50（PRMなし）、今回の60/20/20と一致せず流用しない。さらにコード・初期モデル・教材順/全token列・設定・packageまで一致するE1の完成済み学習があれば、検証した重みを読み取り専用linkで再利用する。判定は`reuse_audit.json`。評価はモデルartifact・全行・推論設定・コード・packageが一致した完成済み回答だけ再利用し、`evaluation_reuse.json`に記録する。自分自身の再開では完成済みphase/回答をスキップする。

`$HOME/geniac_math/suites/e1_limo_600k_v1/`以下：

| 出力 | 内容 |
|---|---|
| `README.md`, `report.json` | 採用結果、baseline/候補指標、bootstrap、phase時間・ピークVRAM |
| `comparison.csv` | 全体/GSM/PRM正答率、生成長、反復率、正常終了率、差とCI |
| `question_changes.csv` | 問題ごとの正誤と変化（問題本文なし） |
| `policy.json`, `selection.json`, `adoption.json` | 事前規則、dev勝者の凍結、独立評価後の採用 |
| `benchmark.json`, `phase_times.jsonl`, `logs/` | 学習時間予測、実時間、5秒間隔で測ったdevice VRAMピーク、ログ |
| `training_trials/E1/training/rft/` | 2地点のadapter、再開state、更新別token/VRAM/時間 |
| `baselines/`, `evaluations/` | 生の生成、既存graderによる正誤、評価再利用identity |
| `submission/submission.json` | 提出するJSON（生成後の回答編集なし） |
| `submission/manifest.json`, `hashes.json`, `format_check.json` | モデル・コード・設定・JSONのhash、公式チェッカー結果 |

提出処理は`math_training/submission.py`と`rank64_12k/official_prompt.py`をimport。運営のmessage builderを使用し、greedy・1問1回答・best-of-Nなし。学習・モデル採用が確定するまで大会問題を開かない。JSON/回答は手編集せず公式チェッカーで検査する。Omnicampus送信なし。

## ローカル検証

```bash
.venv-local/bin/python -m pytest remote/e1_limo/tests remote/math_training/tests -q
bash -n remote/e1_limo/launch.sh
```

混合トークン・全回答保持・prompt mask・二地点だけの保存計画・容量不足拒否・選択/採用境界・独立評価前の選択凍結・再開順序をCPUで検証。実際のGPU通し実行はリモートコマンドで行う。
