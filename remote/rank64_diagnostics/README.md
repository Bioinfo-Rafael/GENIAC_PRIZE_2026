# D/R12：保存済み公開データだけで行う診断

新規学習・モデル読み込み・再推論・データ取得を一切行わないCPU処理。公式提出データも使用しない。完了済みR12と、その `source.json` が指すDを読み、別ディレクトリに診断結果を保存する。

## 受領済み診断結果（2026-10-21）

リモートで実行したarchiveを解析済み。**D→R12はdev27改善・32悪化（差−1.25ポイント、95%区間−5.25〜＋2.50）**。反復が新たに発生した悪化は7件だけで、残る25件は両方反復19件・両方非反復6件だった。科目ではAlgebra−3、Intermediate Algebra−2、Counting & Probability−2、arithmetic−1、Prealgebra＋2、Precalculus＋1、Geometry/Number Theoryは差0。

D/R12の全375/750更新のlossは有限で、最初の25更新の0.70180/0.70895から最後の25更新の0.56065/0.55843へ低下。教師token配分は下記ローカル集計と一致し、EOSは全例にある。checkpointは実際に725/750しか残っていなかった。

本番はD **0.16999**、R12 **0.12394**。Dの提出でも反復121/232・上限到達131/232、R12では145/232・156/232へ増加。推論設定・主要ライブラリ版は両者で一致。

[統合分析と全loss窓](../../research/20261021_d_r12_diagnostic_review.md)、[変化59件の一覧](../../research/20261021_d_r12_dev_changes.md)、[集計JSON](../../research/20261021_d_r12_diagnostic_metrics.json)。新たな学習・再推論はしていない。公開dev本文がarchiveに含まれないため、59件の数学的な誤答理由の分類は未実施。以下のコマンドは集計の再現用であり、今回の結果確認のために再実行する必要はない。

## 調べること

1. **dev400のD→R12**：同一ID・評価契約を検査し、誤答→正解、正解→誤答、維持を集計。GSM/PRM、保存済みsubject（Algebra等）、問題文の文字数帯ごとに分け、生成長・反復・non_stopの変化も出す。paired bootstrap 95%区間も再計算する（科目別は探索的比較で、多重比較補正はしない。実行間変動も含めない）。元の採点結果を使い、再採点・再生成しない。subjectは元データのラベルであり、細かな誤答原因を自動で断定しない。
2. **checkpoint残存状況**：125/250/375/500/750と実際に残るdirectoryを照合。750は保存済みdev結果を利用できる。元実装は25更新ごとに保存する一方、直近2個以外を削除するため、通常は725/750しか残らない。欠落checkpointを再学習で復元しない。存在してもこのスクリプトは再推論せず、追加評価候補として報告する。
3. **教師token・lossログ**：6,000/12,000例の長さ、共通6,000例と追加6,000例、問題文・説明・最終boxed・boxed後の本文・EOS・書式・領域境界のtoken数を集計。decode→encodeが全token一致することを検査し、最初の予測不能tokenを除く実際のtargetだけ数える。固定16例ごとの正規化に対応する平均weightも出す。保存loss・LR・gradient norm・秒数を更新番号で照合し、25更新窓に集計する。累積教師token・共通/追加例数も125/250/375/500/750で記録する。

「最終回答」は最後の括弧が閉じた `\boxed{...}` の範囲という機械的定義。文章全体の最終回答や数式の正答性を推定していない。boxed後に単位や説明があること自体は誤りではない。領域境界を跨ぐtokenはboundaryに分け、二重計上しない。

**token割合はlossの係数配分であり、観測された部分別NLLや勾配ノルムの比率ではない。** 当時のログは各更新の総lossしか保存していないため、歴史的な問題文/EOS別lossは復元できない。部分別NLLを測るには、保存モデルを使った別途forward計算が必要だが、今回は低優先として実行しない。lossの時系列も毎回違うbatchでの値であり、同一検証集合上の改善曲線ではない。

## ローカルで既に確認できた結果

同じ選択コードで作成済みの公開教材キャッシュを実測した。Dの6,000例はR12に全て含まれ、共通例のtoken列も一致した。受領したリモート集計でも同じ値と更新ログとの照合を確認した。

| 項目 | D 6,000例 | R12 12,000例 |
|---|---:|---:|
| 平均系列長 | 266.43 | 269.62 |
| 中央値 / p90 | 197 / 513 | 197 / 534 |
| 教師target数 | 1,592,551 | 3,223,460 |
| 問題文のtarget割合 | 25.532% | 25.822% |
| 説明（最終boxedより前） | 64.577% | 64.368% |
| 最終boxed | 2.398% | 2.379% |
| boxed後の本文 | 3.290% | 3.276% |
| EOS | 0.377% | 0.372% |
| 書式＋境界 | 3.827% | 3.782% |

全例にEOSが1個あり、EOS後のtokenは0、最終boxedも全例で検出された。追加6,000例の平均長は272.82。末尾EOSの脱落や教師系列長の急増は認めない。配分はD/R12でほぼ同じなので、全文NLLだけで両者の差を説明はできない。ただし全文NLLと別lossを比較したわけではなく、仮説2を否定する結果でもない。

R12の375更新時点は6,000例を消費しているが、元のD教材は3,009例、追加教材は2,991例。累積targetは1,605,454。**この時点のモデルが残っていたとしても、Dと同じ6,000例を学習したモデルではない。** 件数、教材の構成、更新順、750更新用schedulerの進行が異なる。

## リモート実行（CPUのみ）

```bash
(
  set -e
  cd "$HOME/geniac-remote"
  git pull --ff-only origin main
  if git sparse-checkout list >/dev/null 2>&1; then
    git sparse-checkout set remote/math_training remote/stage1_rank_ablation remote/rank64_12k remote/rank64_diagnostics
  fi
  R12_RUN="$HOME/geniac_math/rank64_12k/rank64_12k_20261021_183439"
  test -f "$R12_RUN/evaluation/R12/generations/b0_eval.json"
  nohup bash remote/rank64_diagnostics/launch.sh "$R12_RUN" \
    >> "$HOME/geniac_math/rank64_diagnostics.console.log" 2>&1 < /dev/null &
  echo "PID=$!"
)
```

既存 `.venv-rank-ablation` を使用。新しいpackageは入れない。GPUを無効化しofflineで実行するため、D提出の推論と並行してもGPUを使わない。同じ出力先の二重実行をlockで拒否する。ソース実験への書き込みは行わない。入力hashとコードを固定し、同じコマンドでCPU集計を再実行できる。入力が変わった場合は停止する。

```bash
tail -f "$HOME/geniac_math/rank64_diagnostics.console.log"
DIAG="$HOME/geniac_math/diagnostics/rank64_12k_20261021_183439_saved_v1"
cat "$DIAG/README.md"
cat "$DIAG/dev_transitions.json"
cat "$DIAG/checkpoints.json"
```

- `report.json` / `README.md`：全体の集計・限界。
- `dev_transitions.json`：全体・dataset・subject・問題長別の正誤変化と区間。
- `dev_question_changes.csv`：問題IDごとの変化・生成長・終了・反復。問題文・回答本文は含めない。
- `teacher_summary.json`、`{D,R12}_teacher_tokens.csv`：token配分と分布。
- `{D,R12}_updates.csv`：各更新のloss・LR・gradient norm・長さ・領域weight。
- `{D,R12}_loss_windows.csv`：25更新ごとのloss・長さ・時間。
- `checkpoints.json`：残るcheckpointと希望地点の可否。
- `identity.json`：解析した公開入力とログのhash。

同じ親ディレクトリに **`rank64_12k_20261021_183439_saved_v1.tar.gz`** を生成する。診断JSON/CSV/READMEだけを含み、問題文・回答本文・モデル・公式提出データ・private logを含めない。この小さいarchiveをローカルへ取得すれば、正誤が変わった分野とloss推移の具体的な解釈を続けられる。

## 検証

合成データで領域別targetの総数、最後のboxed、token列不一致の拒否、正誤変化の双方向集計、checkpoint欠落、実更新ログとの照合を検査した。実際の公開教材6,000/12,000例でもtoken列のround-tripと集計を検証した。合成の正誤ラベル＋実際の公開教師キャッシュを使う一連の集計・archive作成も検証した。この合成正誤の値は実験成績として記録しない。GPU計算はしていない。
