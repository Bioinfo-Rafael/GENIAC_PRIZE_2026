# 数学Curriculum：rank8 / rank32、累計6,000 / 10,000問

既存 `math_curriculum_20261021_3000_v1` を読み取り専用で利用する4条件のsuite。Stage 1の再学習・再merge・venv構築はしない。実行順は **R8_6k学習 → merge → 提出JSON・公式検査 → 固定公開評価 → R8_10k → R32_6k → R32_10k**。提出生成に失敗したら停止する。公開評価のスコアに関係なくJSONを保存し、Omnicampusには送信しない。

## 固定レシピ

- Qwen3-4B-Base revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`。共通親は既存GSM2,000問・rank8 merge済みモデル。
- Stage 2は既存のPRM800K/LIMO tokenize済みpool・split・IDを再利用。原文・解説を切断せず回答＋EOSのみNLL。seed42、context8192、microbatch1、実効batch16（件数境界の末尾8問だけ端数）。Unsloth/NF4/FP16、gradient checkpointing、AdamW weight decay .01、既存loss/scaler/checkpointエンジンを再利用。
- 元300kの実使用1,040問＋継続の実使用1,960問をschedule・完了署名・保存IDから順序込みで復元。以後の7,000問を最初に固定する。重複や評価集合との交差、教師ラベル/EOS、出自hashを検査。
- PRM/LIMOは累計教師トークン80:20へ近づくseed42順のsource先頭を選ぶ。意味・正答率・本番入力による選別なし。不足時は反復で埋めず実件数を記録する。6k以下しかない場合は異なる2地点を作れないため停止する。
- R8は学習済みrank8/alpha16をロード。追加3,000→10,000問全体に新しいAdamW＋5%warmup cosine、上限5e-6。6kでschedulerを作り直さない。
- R32は同じGSM親に新規rank32/alpha64。従来と同じq/k/v/o/gate/up/downの7 modules。R8と同じ固定10,000問・順序。最初の1,040問（65更新）は上限5e-5、以後は上限5e-6。各区間は5%warmup cosine、optimizerは全区間で一つ。65更新後にLR scheduleを低LR区間へ切り替え、6kでは連続させる。切替規則・全更新数は開始前に凍結する。
- rank32/rank64過去実験は上限5e-5だが、全token loss、context2048、別Stage 1/教材/optimizer等が異なる。今回は成功したCurriculumの回答NLL・8192・AdamWを優先する。

**rankだけの因果比較ではない。** R8は既存1,040問と1,960問でそれぞれ終了したcosine・3,000問地点のoptimizer再作成を継承する。R32は1,040問後の低LR区間を最終10,000問までとし、optimizerをリセットしない。rank以外にschedulerの履歴・optimizer履歴も異なる。全条件に同じ問題順・同じ共有Stage 1を使うが、学習履歴の差を各 `experiment.json` に残す。

既存 `math_curriculum.gpu.train/merge`、`math_training.worker.train`、`submission.generate`、`math_curriculum.evaluate` をimport。新規トレーナーなし。共通GPU loaderに任意のrank/alpha/固定recipe引数のみ追加した。既存呼び出しの既定値は維持する。

## 起動

ローカルでテスト・commit・push後、リモートで実行する。既存venvとHF cacheを使い、pip installはしない。

```bash
cd "$HOME/geniac-remote"
git pull --ff-only origin main
if git sparse-checkout list >/dev/null 2>&1; then
  { git sparse-checkout list; printf '%s\n' remote/math_scale remote/math_curriculum remote/math_training remote/rank64_12k; } | sort -u | git sparse-checkout set --stdin
fi
MS_ID=math_scale_20261021_v1
nohup bash remote/math_scale/launch.sh "$MS_ID" \
  > "$HOME/geniac_math/$MS_ID.console.log" 2>&1 < /dev/null &
echo "PID=$!"
```

既存共有lock2個を取得し、GPUがRTX 6000 Ada 48GB・空き状態、ディスク60GiB以上を検査する。4個のmergeモデル・checkpointを保持する。コードsnapshotとpackage一覧を保存する。訓練中と推論中のプロセスを分離し、前のプロセス終了後に次を起動する。

## 600分の上限・再開

launch開始を `budget.json` に保存し、**停止中の時間・失敗・再開を含めて36,000秒を上限**にする。各条件開始前に更新数と実測速度（初回は過去実測）から学習・merge・提出・評価の時間と余裕を計算する。提出時間まで確保できない未着手条件はスキップ。実行中は期限120秒前にSIGTERMで更新境界保存を促し、最終15秒前に未終了のphaseを強制終了。外側にも同じ期限のtimeoutを置く。異常なGPU/OS停止で安全保存が間に合わない場合も既存checkpointと完成済み提出を維持し、未完了を完成扱いしない。

通常の停止はcoordinatorに `kill -TERM PID`。同じID・同じコマンドで再開する（既存プロセス終了後）。期限は延長されない。完了phaseは全成果物hashを検査してskipする。学習は既存エンジンの10更新ごとのcheckpointとmilestoneからoptimizer・scheduler・AMP scaler・Python/Torch/CUDA RNGを復元。6kの状態は各条件の `checkpoint/` に別保存し、後続学習で削除されない。条件ごとのreceiptにより提出だけの再開も可能。途中停止で最新保存以降の更新だけは再計算し得る。

## 進捗と成果物

```bash
tail -f "$HOME/geniac_math/math_scale_20261021_v1.console.log"
MS_OUT="$HOME/geniac_math/math_scale/math_scale_20261021_v1"
cat "$MS_OUT/suite_report.json"
cat "$MS_OUT/runtime_summary.json"
```

提出完了時に `SUBMISSION_READY R8_6k` 等を表示。各条件の提出は、後続条件が実行中でも利用できる。

| 条件 | 提出JSON（`$MS_OUT` 以下） |
| --- | --- |
| R8-6k | `R8_6k/submission/submission.json` |
| R8-10k | `R8_10k/submission/submission.json` |
| R32-6k | `R32_6k/submission/submission.json` |
| R32-10k | `R32_10k/submission/submission.json` |

同じ提出フォルダーに `format_check.json`、`hashes.json`、`manifest.json`、`complete.json`、推論時間を保存。公式232 ID・件数・形式を検査する。SHA256はJSON、モデル、tokenizer、設定、コードを対象にする。公式推論message builder・公式チェッカーを再利用し、単一モデル、greedy、n=1、生成回答無編集。既存Curriculumと同じcontext8192/max_tokens2048（公式olympiad既定4096からの変更）・seed42・chunk32・FP16・VRAM .8。

公式入力は提出phaseだけで取得・参照する。過去の提出回答は読まない。問題・生回答を含み得るログは `private_logs/` に置き、consoleへ転送しない。学習や公開評価に公式入力を渡さない。

各条件の `report.json` は固定PRM200＋LIMO200評価、抽出失敗、反復率、平均生成長、上限到達率、既存300k/3,000問および先行条件との差・paired bootstrap95%区間を保存する。`question_changes.json` は問題別正誤変化。既存grader・max_tokens2048を維持し、再サンプリングしない。過去の評価はモデル・split・設定・grader・packageの一致を検証して再利用。公開評価と本番スコアは区別し、本番スコアは未提出のためnull。

`runtime_summary.json` は起動からのwall時間、各phaseの全試行時間、学習allocated VRAMと全phase device VRAMを区別。`suite_report.json` は各条件のpending/running/submission_ready/complete/skipped状態・提出先・実数・評価を逐次更新する。

## ローカル検証

保存済み公開token poolの確認値（リモート実績ではない）：未使用PRM7,536＋LIMO383、累計10,000問を重複なしで確保可能、教師2,753,159 token、PRM:LIMO=80.0036:19.9964。R8追加438更新（188＋250）、R32計626更新（376＋250）。端数batchは過去の1,040/3,000問境界も維持する。[計画値](local_validation.json)

```bash
.venv-local/bin/python -m pytest -q remote/math_scale/tests remote/math_curriculum/tests remote/math_training/tests/test_submission.py remote/math_training/tests/test_losses.py
bash -n remote/math_scale/launch.sh
```

実GPU結果は実行後の `suite_report.json` と `runtime_summary.json` を正とする。

本番R8-6kスコア0.19133の報告後、元の10時間枠の残りを使う[追加キュー](FOLLOWUP.md)を実装。元4条件の終了後も、学習→提出JSON生成・形式検査→公開評価を固定順で繰り返す。
