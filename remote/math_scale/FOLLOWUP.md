# 元の10時間枠を使う追加キュー

R8-6kの本番スコアは **0.19133**（2026-10-21、ユーザー報告）。既存3,000問の0.20384から−0.01251。原因や統計的有意性は未確定で、公開評価と混同しない。[出典記録](followup_evidence.json)

固定公開400問は105問正解から106問正解へ変化（PRM98→97、LIMO7→9）。差は+0.25ポイント、paired bootstrap 95%区間は−2.0〜+2.5ポイントで、明確な改善は確認できない。本番と公開評価の差だけから過学習などの原因を断定しない。

**元suiteの4条件を変更せず完了させ、その直後にこの固定キューを自動実行する。期限は元の起動23:10:02から10時間後、2026-10-22 09:10:02 JST。** 追加で10時間は設けない。停止時間・待機・準備・学習・推論・評価も同じ枠に含める。

## 実験順

各行は一つの連続学習計画。行内の各checkpointで **学習 → optimizer付きcheckpoint保存 → merge → 232回答JSON → 公式形式検査 → 公開評価 → 次の学習** を繰り返す。提出が失敗したら以後を停止する。公開評価の成績によるキュー・データ・LR変更はない。

| 優先順 | 系統ID | 開始点 | Stage 2 rank / alpha | LR上限 | 提出を作る累計件数 |
| --- | --- | --- | --- | --- | --- |
| 1 | `C3_R8_L2e6` | 既存3,000問 | 8 / 16 | 2e-6 | 4,000 → 6,000 → 10,000 |
| 2 | `C3_R8_L1e6` | 同上 | 8 / 16 | 1e-6 | 同上 |
| 3 | `C3_R8_L5e7` | 同上 | 8 / 16 | 5e-7 | 同上 |
| 4 | `F_R16_L5e5` | 既存GSM merge親・新規Stage 2 | 16 / 32 | 最初1,040問5e-5、以降5e-6 | 3,000 → 6,000 → 10,000 |
| 5 | `F_R32_L2e5` | 同上 | 32 / 64 | 最初1,040問2e-5、以降2e-6 | 同上 |
| 6 | `F_R8_L1e5` | 同上 | 8 / 16 | 最初1,040問1e-5、以降1e-6 | 同上 |
| 7 | `C1_R8_L2e6` | 既存300k＝1,040問 | 8 / 16 | 2e-6 | 3,000 → 6,000 → 10,000 |
| 8 | `C1_R8_L1e6` | 同上 | 8 / 16 | 1e-6 | 同上 |
| 9 | `F_R16_L2e5` | GSM merge親・新規Stage 2 | 16 / 32 | 最初1,040問2e-5、以降2e-6 | 同上 |
| 予備1 | `TAIL_R8_L15e7` | 既存3,000問 | 8 / 16 | 1.5e-6 | 3,500 |
| 予備2 | `TAIL_R8_L75e8` | 同上 | 8 / 16 | 7.5e-7 | 3,500 |
| 予備3 | `TAIL_R8_L3e6` | 同上 | 8 / 16 | 3e-6 | 3,500 |

最大30個の**追加**提出物を用意した待ち行列であり、30個すべての完了を約束する計画ではない。過去実測では元4条件が約1.5〜2時間、追加1行は概ね50〜70分、予備は15分前後が目安。キューは残り枠より十分長くし、各checkpoint直前の実測token throughput・直近の提出/評価時間で更新する。学習30%、推論25%の余裕と保存時間を確保し、完成可能なcheckpointを順に実施する。時間不足ならその行の残りをskipし、後ろの短い独立条件が入るかを確認する。使い切るための空待ちや不要な再学習は行わず、最終提出の安全余裕は残す。

## 比較の狙いと制約

まず強い3,000問adapterを保持しながら、更新を小さくした場合と4,000問で止めた場合を調べる。その後に中間rank16、rank32の低LR、新規rank8の低LRを比較する。元300kからの継続は、3,000問までの追加区間自体のLRも下げる対照。予備は小さな500問継続の強度比較。

同一固定10,000問集合・順序と元のPRM/LIMO tokenizer結果を再利用。各モデル内で重複問題なし、回答＋EOS NLL、max8192、実効batch16・microbatch1、seed42、教師トークン80:20目標。4,000問など端数境界のみ小batchとし、その位置は開始前に固定する。Stage 1はすべて共有、再学習しない。各系統内は全体cosine計画を最初に凍結し、提出推論後もoptimizer/scheduler/scaler/RNGを復元する。

rank・LR・開始checkpoint・optimizer履歴・cosine horizonに差がある探索であり、全条件がrankだけの比較ではない。多数の固定公開400問比較は探索的で、95%区間は問題再標本化のみ、未補正多重比較。公式スコアはこのキューの実行中の自動選択・教材選別に使用しない。公式入力・過去回答は学習/公開評価へ渡さない。

## 起動と再開

ローカルでテスト・push後、リモートで次を実行する。元suiteのsnapshot・プロセスは変更しない。

```bash
cd "$HOME/geniac-remote"
git pull --ff-only origin main
nohup bash remote/math_scale/queue_followup.sh \
  "$HOME/geniac_math/math_scale/math_scale_20261021_v1" \
  math_scale_followup_20261021_v1 \
  > "$HOME/geniac_math/math_scale_followup_20261021_v1.console.log" 2>&1 < /dev/null &
echo "PID=$!"
```

キュー独自lockで二重起動を防ぐ。待機中はGPU lockを取得せず、元suiteの全4条件completeと既存成果物hashを確認してから共有lock2個を取得する。元suiteが失敗/予算停止していれば追加学習も開始しない。venv/cacheは再利用、installなし。元の絶対期限をコピーし、外側timeoutと各phaseのSIGTERM/強制終了ガードを共有する。

同じコマンド・同じIDで再開可能。完了phaseはhash確認後skipし、未完了の学習は同じ状態を復元する。期限は更新されない。追加キューだけ止める場合はそのcoordinatorへSIGTERM（元suiteは止まらない）。

```bash
tail -f "$HOME/geniac_math/math_scale_followup_20261021_v1.console.log"
MF_OUT="$HOME/geniac_math/math_scale/math_scale_followup_20261021_v1"
cat "$MF_OUT/suite_report.json"
cat "$MF_OUT/runtime_summary.json"
```

`FOLLOWUP_QUEUED` は待機中、`PREDECESSOR_COMPLETE_LOCKS_ACQUIRED` が開始、`SUBMISSION_READY C3_R8_L2e6_4000` 等が提出物完成。出力例は

```text
$MF_OUT/C3_R8_L2e6_4000/submission/submission.json
$MF_OUT/C3_R8_L2e6_4000/submission/format_check.json
$MF_OUT/C3_R8_L2e6_4000/submission/hashes.json
$MF_OUT/C3_R8_L2e6_4000/checkpoint/
$MF_OUT/C3_R8_L2e6_4000/model_manifest/
$MF_OUT/C3_R8_L2e6_4000/report.json
```

空き容量は起動時点で68GB（元suiteは当時R8-6k評価中）。追加実験は各自の**一時mergeコピーだけ**、提出と評価の全成果物hash確認後に削除する。元の全モデル・元4提出・旧レポートには触れない。新しい各adapter、optimizer付きcheckpoint、親モデル参照、モデル全ファイルhash、JSONと公開評価は保持するのでmergeを再構築できる。`model_manifest/rebuild.json` に再構築手順を保存。merge解放済みの実験を通常再開するときは再mergeせず完成receiptを確認する。

ローカル検証は固定順・非重複・件数、期限継承、待機中のGPU未使用、提出失敗ゲート、LR適用、出自不変、追加mergeのみの解放、既存処理の回帰を対象とする。
