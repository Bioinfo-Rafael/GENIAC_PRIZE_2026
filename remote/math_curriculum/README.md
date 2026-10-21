# GSM8K → PRM800K/LIMO：数学だけの2段階SFT

新しく学習するのは **1条件だけ**。Tulu Stage 1、run03等の既存adapter、過去の提出回答は読み込まない。固定revisionのBaseからGSM2,000例を学習してmergeし、新しい数学LoRAを学習する。現在は公開評価を省略し、最終checkpointの公式提出JSON生成まで自動実行する。Omnicampusへの送信はしない。

ローカルの公開データ処理・CPUテストを実施済み。その後、GPU学習ログ・提出JSON・本番スコアを受領した。公開評価は省略した。以下は再現・再開用の起動手順。


## 最新結果：数学Curriculum累計3,000問で本番0.20384（2026-10-21）

**訂正後の本番スコア0.20384（ユーザー報告）が、旧300kの0.19317から+0.01067で最高更新。** 固定PRM/LIMO 400問は102→105正解（25.50→26.25%）、PRM94→98/200、LIMO8→7/200。9改善・6悪化、正答率差のpaired bootstrap 95%区間は−1.25〜+2.75ポイントで0を含み、公開評価の改善はまだ明確とは言えない。本番スコアと公開正答率は別指標。

GSM Stage 1と300k LoRAを引き継ぎ、LR5e-6で未使用1,960問・123更新を追加して累計3,000問を完了。PRM2,960＋LIMO40問、教師844,566トークン、教師比79.83:20.17。追加学習更新367.81秒（6分8秒）、ロード・保存込み学習phase400.94秒（6分41秒）、評価合計807.67秒、提出生成phase254.71秒、記録済み全phase合計1,533.93秒（25分34秒）。意図的な評価停止・再開を含むが、停止中の手動待機等は含まない。

提出232回答は反復39→42、平均生成長641.16→648.12 tokenで、反復減少がスコア上昇の説明にはならない。推論設定・主要ライブラリ版は一致。形式検査はエラー・違反・警告すべて0。元JSONは無編集。

[詳細分析・評価・学習量・時間](../../research/20261021_math_curriculum_3000_analysis.md)／[受領評価report](../../research/20261021_math_curriculum_3000_report.json)／[受領runtime](../../research/20261021_math_curriculum_3000_runtime.json)／[提出回答集計](../../research/20261021_math_curriculum_3000_submission_metrics.json)／[実行・再開手順](README.md)。新runは `math_curriculum_20261021_3000_v1`。以下の最高・未確認という旧記録は当時の状態。

## 追加実験：Stage 2累計3,000問へ継続（2026-10-21）

既存の `math_curriculum_20261021_v1` の300k adapterを引き継ぐ。**Stage 1の再学習・再mergeは行わない。** 既存runのモデル・checkpoint・提出JSONには書き込まず、新run `math_curriculum_20261021_3000_v1` に保存する。今回の経路は、下記の初回実験用「評価省略」経路と異なり、**継続後モデルの提出JSONを先に生成し、その後に既存300kと継続後の2モデルを固定PRM/LIMO評価セットで比較**する。評価結果にかかわらず継続後モデルを提出用に出力するという依頼に従い、別モデルは学習しない。

### 起動・途中再開

リモートで以下を実行する。最後のスクリプトが同じrunの旧coordinatorだけをSIGTERMで停止し、終了を待ってから `nohup` でバックグラウンド再開し、PIDとログ先を表示する。学習・merge完了済みなら再実行せず、提出JSONを優先する。既に新coordinatorが実行中ならそのまま継続する。既存venv・HFキャッシュを使用し、pip installは行わない。同じコマンド・同じrun IDで途中再開できる。別runの実行中は既存共有lockで拒否される。

```bash
(
  set -e
  cd "$HOME/geniac-remote"
  git pull --ff-only origin main
  if git sparse-checkout list >/dev/null 2>&1; then
    { git sparse-checkout list; printf '%s\n' remote/math_curriculum remote/math_training remote/public_eval_diagnostics remote/rank64_12k; } |
      sort -u | git sparse-checkout set --stdin
  fi
  bash remote/math_curriculum/start_continuation.sh \
    math_curriculum_20261021_3000_v1 math_curriculum_20261021_v1
)
```

```bash
tail -f "$HOME/geniac_math/math_curriculum_20261021_3000_v1.console.log"
```

停止する場合は、この新runのPIDを確認して `kill -TERM PID` を使う。更新境界で保存する。再開では新runのoptimizer/scheduler/RNGを復元し、完了済みphaseをスキップする。既存300kのoptimizerやschedulerは引き継がない。

### 引き継ぎ・学習レシピ

- 元の `stage2/pairs.json` と `stage2/schedule.json`、`training/rft/complete.json` の署名・完了更新数から使用済み問題を復元する。候補poolの件数や65×16という推測だけで既存件数を決めない。
- 300k milestoneとfinal adapterのhash一致、merge済みモデルのlineage、固定データ・tokenizer・モデルのSHA256を検証する。以後の再開でも出自の変更を検出する。必要な元ファイルがない場合は、再学習で代用せず停止する。
- 元の未使用tokenize済みpoolを使うため、再取得・再tokenize・再splitは不要。元の派生問題グループ単位の分離を保持し、使用済みIDと追加IDを保存する。PRM/LIMO以外は追加しない。
- 既存のGSM mergeモデルを親にして**学習済みrank8/alpha16のLoRA重みを読み込む**。新規重みからの学習は行わず、追加用AdamWとcosine scheduler（warmup5%）を新しく作る。LR上限5e-6、回答＋EOSのみNLL、context8192、seed42、microbatch1、実効batch16。Unsloth/NF4/FP16等は既存設定を継承する。
- 累計3,000問を厳密に上限とし、最後の更新だけ16問未満を許す。既存1,040問なら追加1,960問＝122更新×16＋最後8問で123更新。重複によるpaddingや3,008問への超過は行わない。
- 元の教師トークンも含めた累計80:20に近づくsourceを、seed42の固定順とトークン長だけで選ぶ。LIMO不足時は繰り返さずPRMで補い、全pool不足なら実際の件数・不足数・比率を報告する。完全な解説を保持する。

### ローカル検証で得た計画値と時間の目安

以下は保存済み公開データに対するCPU検証であり、**リモートの学習完了実績ではない**。起動時に実際の元runから再計算する。[検証JSON](continuation_validation.json)

| 指標 | 既存300k | 追加分 | 累計 |
| --- | ---: | ---: | ---: |
| ユニーク問題数 | 1,040 | 1,960 | 3,000 |
| PRM / LIMO問題数 | 1,025 / 15 | 1,935 / 25 | 2,960 / 40 |
| 教師トークン | 298,233 | 546,333 | 844,566 |
| PRM / LIMO教師トークン比 | 79.93 / 20.07% | 79.78 / 20.22% | 79.83 / 20.17% |

旧Stage 2の成功更新198.80秒から教師トークン比例で見積もると、追加更新は約6分。ただし長さ構成・コンパイル等で変わり、ロード・保存・merge・評価・提出推論を含まない。全工程は暫定15〜45分程度を目安とし、実測の保証はしない。最初の数更新の速度・ETAを優先する。速度測定だけの別学習はしない。

### 評価・保存先

`data/eval.json` を元runからそのまま固定で再利用する。PRM200＋LIMO200（元splitで不足している場合は元の実件数）、greedy・1問1回答・最大2048 token、同じ `evaluate.py` / 最後のboxed / math-verifyを使用する。旧GSM/PRM dev400や本番回答は読まない。既存300k評価が完了し、評価セット・モデルhash・設定・grader・ライブラリ版が一致する場合は新runへコピーして再利用し、推論し直さない。一致しなければ同じ新環境で評価する。

```bash
MC_OUT="$HOME/geniac_math/math_curriculum/math_curriculum_20261021_3000_v1"
cat "$MC_OUT/status.json"
cat "$MC_OUT/data/report.json"
cat "$MC_OUT/report.json"
cat "$MC_OUT/runtime_summary.json"
cat "$MC_OUT/submission/format_check.json"
```

| 新run以下 | 内容 |
| --- | --- |
| `continuation.json` | 元run・300k adapter・モデル/データ/設定のhashと継続方針 |
| `data/used_stage2_ids.json`, `data/additional_stage2_ids.json` | 既存使用済みID・追加ID（重複なし） |
| `data/report.json` | 既存・追加・累計の予定問題数、source別教師トークン、比率、不足理由 |
| `stage2/training/rft/complete.json` | 追加学習の完了更新数・実教師トークン。候補pool数と区別 |
| `stage2/training/rft/` | 新run専用のoptimizer付きcheckpoint・最終adapter・更新ログ |
| `models/cumulative3000/` | GSM merge親＋継続更新済みStage 2をmergeした単一モデル |
| `evaluations/{300000,cumulative3000}/` | 同一条件の回答・PRM/LIMO/全体正答率・分母・生成長・反復・上限到達等 |
| `report.json`, `question_changes.json` | 実現した累計件数・2モデルの比較・問題別正誤変化・paired bootstrap 95%区間 |
| `runtime_summary.json`, `phase_times.jsonl` | 追加学習・評価・提出の時間、ピークVRAM。Stage 1時間は加算しない |
| `README.md` | 完了後に自動生成する学習実績・評価結果・時間のまとめ |
| `submission/submission.json` | 継続後モデルの未編集提出JSON |
| `submission/{format_check,hashes,manifest}.json` | 公式形式検査・SHA256・モデル/tokenizer/設定・出自 |

元runの `models/stage1` と `models/300000` は読み取り参照し、新runではStage 2最終mergeだけを行う。新run名・`cumulative3000` は目標名で、不足時は `report.json` の実件数を正とする。公式提出処理は `math_training/submission.py` と既存wrapperを使用し、単一モデル・greedy・1問1回答、出力無編集、Omnicampus送信なし。提出対象は評価結果と無関係に継続後モデルへ固定し、公式入力は提出処理だけに渡す。提出JSON生成後に公開評価を行い、本番入力・生成回答を学習・公開評価・モデル選択へ流用しない。

ローカルでは継続専用・既存Curriculum・共通loss/提出/scheduleの計51テストを通過。使用済みID除外、端数batch、教材不足、初期LoRA/再開LoRAの読み分け、評価cache一致条件、失敗後のphase再開、元run不変、継続モデルからの無編集JSON生成を検証した。シェル構文・Python構文・差分形式も検査済み。

実行順変更は `control_continuation_first/run.py` へ別保存し、既存runの `code/` snapshot・学習署名・checkpointは変更しない。提出JSON完成時には `SUBMISSION_READY_BEFORE_EVALUATION` と表示する。その後の評価中でも `submission/submission.json` をダウンロードできる。

実装は既存runnerへの継続分岐、既存GPU loaderへの初期adapter指定、件数ベースの追加schedule作成が中心。共通学習・checkpoint・grader・公式提出エンジンは変更していない。GPU実行と結果の受領は完了し、冒頭の「最新結果」に記録した。以下の初回300k記録とローカル計画値は当時の状態を示す。

## 実行結果：本番スコア0.19317（2026-10-21）

**ユーザー報告の0.19317で、記録上の最高run03（0.18758）を+0.00559更新。** 受領した提出JSON232回答の公式形式チェックはerrors/violations/warningsすべて0。公開400問評価はユーザー指示で省略し、30万予算側を直接提出した。

run03→今回で平均生成長789.37→641.16 token、反復55→39件、上限到達56→46件。math_compで出力挙動が改善した一方、olympiadは20/36件が2048 tokenに到達。回答中の検算では改善・悪化が両方あり、正答率の改善原因やtask別得点は未確認。

添付ログでStage 1 125更新、Stage 2 65更新の完了を確認。成功更新の時間は約168秒＋199秒（ロード・保存等は別）。提出推論時間の実測ファイルは未受領。[詳細監査・設定差・検算例・集計JSON/CSV](../../research/20261021_math_curriculum_submission_audit.md)を参照。下記の実行前の記述より本節の受領結果を優先する。

## 更新：評価を省略して提出JSONを先に生成

現在の既定動作は、PRM200＋LIMO200の**評価推論を実行せず**、Stage 2を最後まで学習して30万予算側のcheckpointから提出JSONを作る。10万側adapterは保存するが、そのmerge・評価のための中断は省略する。正答率比較・paired bootstrapは実施せず、`report.json` に `evaluation_status: skipped` を記録する。既に選択まで完了しているrunでは、その確定済み選択・提出を保持する。

評価用400問の分離と学習からの除外は維持する。これによりデータ・学習条件・optimizer状態が変わらず、既存runを再学習せず継続できる。以下の旧「評価と選択」の説明は `run.py --evaluate-checkpoints` を明示した場合のみ適用する。

既に旧コマンドで起動した場合も、pull後に次のコマンドで切り替えられる。同じrunの旧coordinatorだけへSIGTERMを送り、checkpoint保存・終了を待ってからバックグラウンド再開する。別のGPUジョブは停止しない。新方式が既に動いていればそのまま継続する。

```bash
cd "$HOME/geniac-remote"
git pull --ff-only origin main
bash remote/math_curriculum/start_submission_first.sh math_curriculum_20261021_v1
```

新しい制御処理は `control_submission_first/run.py` に別保存する。既存の `code/`、学習署名、checkpoint、完了phaseのreceiptは変更しない。提出先とログは下記のまま。学習開始前でも同じコマンドで起動できる。

## pull・バックグラウンド起動

既存の `~/geniac-remote`、`~/geniac_math`、HF認証・キャッシュを利用する。学習は実績のある `.venv-rank-ablation`（Unsloth入り）、公開評価・提出推論は `.venv-eval`、公式データ取得は `.venv-submission-data`。自動再インストールや速度候補の探索はしない。

```bash
(
  set -e
  cd "$HOME/geniac-remote"
  git pull --ff-only origin main
  if git sparse-checkout list >/dev/null 2>&1; then
    { git sparse-checkout list; printf '%s\n' remote/math_curriculum remote/math_training remote/public_eval_diagnostics remote/rank64_12k; } |
      sort -u | git sparse-checkout set --stdin
  fi
  MC_ID=math_curriculum_20261021_v1
  MC_LOG="$HOME/geniac_math/$MC_ID.console.log"
  mkdir -p "$HOME/geniac_math"
  nohup bash remote/math_curriculum/launch.sh "$MC_ID" >> "$MC_LOG" 2>&1 < /dev/null &
  echo "PID=$! LOG=$MC_LOG"
)
```

同じコマンド・同じIDで再開。新しい実験としてやり直す場合だけIDを変える。既存実験の実行中は共有lock/GPU検査で停止するため、終了後に再実行する。空きディスク40GiB以上が必要。venvの場所が異なる場合は `GENIAC_TRAIN_PYTHON`、`GENIAC_EVAL_PYTHON`、`GENIAC_DATA_PYTHON` を起動前に指定する。作業先は `GENIAC_WORKSPACE` で変更できる。

```bash
tail -f "$HOME/geniac_math/math_curriculum_20261021_v1.console.log"
```

コンソールにphase、学習更新数、経過秒・残り推定秒、教師tokens/sec、ピークVRAMを表示する。phaseログは30秒ごとにも進捗を表示する。最初の1〜3更新の推定はコンパイル・warmupの影響を含む暫定値で、学習が進むと更新される。測定更新も本学習に含め、速度測定用の再学習はしない。

## 条件と処理順

| 項目 | Stage 1 | Stage 2 |
|---|---|---|
| 初期モデル | `Qwen/Qwen3-4B-Base` | 今回のGSM Stage 1をFP16 mergeしたモデル |
| Base revision | `906bfd4b4dc7f14ee4320094d8b41684abff8539` | 同左 |
| データ | GSM8K 2,000問・1 epoch | PRM800K/LIMO、教師token目標80:20、GSMなし |
| LoRA | 新規rank8 / alpha16 | 新規rank8 / alpha16 |
| LR | 1e-4 | 5e-5 |
| 最大系列長 | 2048 | 8192 |
| 実効batch | 16（microbatch2、OOM時1） | 16（microbatch1） |
| loss | run01を参考にpromptを含む全非padding NLL | promptをmaskした回答＋EOSのみのNLL |
| 終了 | 125更新 | 最大300,000教師token以下の完全な更新境界 |

共通：Unsloth、NF4 double quantization、FP16、gradient checkpointing、seed42、dropout0、q/k/v/o/gate/up/down projection。optimizer/schedulerは既存 `worker.train` のAdamW（weight_decay .01）、cosine、warmup5%、gradient clipping1を再利用する。run01の完全再現とはせず、指定されたLR・rank・epoch・batch・系列長を満たす新実験とする。GSMの`####`からのboxed整形は既存 `gsm_legacy_target`。固定chat templateと既存語彙を使い、PADには予約済み`<|fim_pad|>`を指定する。語彙追加・モデル拡張はしない。

1. 許可データを固定revision/SHA256で取得し、全ソースをまとめて重複・派生問題のグループを作る。
2. 固定seed42のランダム順でLIMO200、PRM200を先に固定し、その全グループをStage 1/2の学習から除外する。
3. GSM Stage 1の学習・保存・FP16 merge。
4. Stage 2を10万予算の境界まで学習し、保存adapterをFP16 mergeして固定評価を実行。
5. **結果による早期打ち切りはせず**、同じoptimizer/scheduler/RNGを復元して30万予算まで継続。
6. 30万側を保存・merge・評価。両者の全体正答数が多い方を採用。同点は10万側。反復・生成長で選択を変更しない。
7. 採用済みのmergeモデルを再利用して公式JSONを生成・形式検査する。

Stage 1後のmergeは量子化重みへの直接mergeを避け、公式BaseをFP16で再読込して実施する。Stage 2は保存した親を使用する。評価の間は学習GPUプロセスを終了し、再開時に学習状態を復元する。別の比較モデルは学習・推論しない。採用後の再merge・再評価は不要。

### 完全な解説とトークン上限

LIMO/PRMは元の解説全文＋EOSを保持し、系列長を超える例は丸ごと除外する。途中切断・回答途中のlabel maskは行わない。教師tokenはshift後の非mask labelsで数える。完全な解説と16例の更新単位のため、100,000/300,000ちょうどとは限らない。各予算のPRM80%・LIMO20%の上限を両方守り、収まる完全な16例groupのみを追加する。選択は固定順・長さだけで行い、各問題は最大1回使用する。

実績が予算の95%未満、LIMO比率が20%から1ポイント超の差になる場合は理由付きで停止する。これは性能に基づく打ち切りではなく入力条件の不成立。保存名 `100000` / `300000` は予算名で、**実際の教師token数は必ず `data/report.json` / `stage2/schedule.json` を参照**する。

ローカル公開データ検証では、評価PRM200＋LIMO200、GSM2,000を確保。Stage 1は125更新・322,649非mask token。Stage 2は23更新時点で99,678 token（PRM79,822＋LIMO19,856）、65更新時点で298,233 token（PRM238,379＋LIMO59,854）。最終比率79.93:20.07。LIMOの学習候補183件を8192超で丸ごと除外した。旧splitの除外集合が異なる環境では数値も変わるため、リモートの実績を正本とする。

### 重複対策と新しい評価

元問題の正規化SHA256、空白・大小文字・Unicodeの正規化、数値を伏せた構造、同一source ID、5-gram Jaccard 0.85以上を使い、推移的に結合したグループ単位で分離する。PRM内の派生例とPRM/LIMO間の重複をまとめ、同一グループを評価内でも1問に制限する。LIMOを先に割り当ててPRM側からの重複占有を避ける。意味や正解難易度による人手・LLM選別は行わない。機械的基準では検出できない意味上の言い換えやBase事前学習との重複までは保証できない。

既存 `inputs/split.json` または `inputs/dev_eval/split_v1.json` があれば旧dev/finalを除外集合としてのみ読み、新評価には再利用しない。ローカル検証では `local_eval/split_v1.json` を使用。存在の有無・hashと除外原票を保存する。既存dev400の回答・成績は読み込まない。評価が独立200問に満たないLIMOは取得できた全件と不足・除外件数を記録し、PRM200と合わせた実際の分母で評価する。PRM200不足や固定後のprompt超過は黙って問題を削除せず停止する。

正解はPRM/LIMOの元`answer`フィールド。**既存 `grading.last_boxed` の最後のboxed/fboxのみ**を抽出し、`grading.parse_math`とmath-verifyで同値性を検査する。既存のmarker/numeric-only fallbackは今回使わず、全checkpointで固定する。greedy・1問1回答・最大生成2048、context8192。gold解析不能・採点エラーも全体分母に残し、正解には算入しない。

PRM/LIMO/全体の正解数・分母・正答率、抽出失敗、採点不能、平均生成長、上限到達率を保存する。反復率は補助指標。10万→30万の問題別正誤変化と、paired bootstrap 2,000回/seed42の95%区間を記録。評価セットはcheckpoint選択にも使うため、独立した最終テストスコアとは称さない。信頼区間に学習のばらつき・選択バイアスは含まれない。

## 再開と結果確認

コードを初回起動時にrun内へsnapshotし、再開も同じコードを使う。ライブラリ版・コード・設定・完了成果物のhash変更は拒否する。10更新ごとと各milestoneでadapter・optimizer・scheduler・scaler・RNGを保存。SIGTERMは更新境界で保存し、SIGKILL/電源断では直近保存状態から再実行する。10万側adapterは専用milestoneに保存し、通常の古いcheckpoint削除で失わない。評価・提出は保存済み回答から再開する。完了phaseはreceipt/hash検査後にスキップし、既存実験のファイルを変更しない。

```bash
MC_OUT="$HOME/geniac_math/math_curriculum/math_curriculum_20261021_v1"
cat "$MC_OUT/status.json"
cat "$MC_OUT/data/report.json"
cat "$MC_OUT/report.json"
cat "$MC_OUT/selection.json"
cat "$MC_OUT/submission/format_check.json"
```

| `MC_OUT`以下 | 内容 |
|---|---|
| `submission/submission.json` | 未編集の提出JSON |
| `submission/format_check.json` | 公式チェッカー結果。errors/violationsが空であること |
| `submission/hashes.json`, `submission/manifest.json` | JSON・モデル重み・tokenizer・設定のSHA256と出自 |
| `models/stage1`, `models/{100000,300000}` | Base→今回のGSM→各Stage 2の順にmergeしたモデルとlineage |
| `stage{1,2}/training/rft/` | checkpoint、milestone adapter、更新ログ |
| `data/eval.json`, `data/report.json`, `stage2/schedule.json` | 固定評価・除外件数・実教師token配分 |
| `evaluations/{100000,300000}/` | 公開評価の回答、採点、データセット別集計、生成時間 |
| `report.json`, `selection.json`, `question_changes.json` | 正答率比較・paired CI・採用理由・問題別変化 |
| `phase_times.jsonl`, `runtime_summary.json` | 各phaseの秒数、Stage別学習秒数、評価秒数、提出生成だけの秒数、ピークVRAM |
| `logs/`, `private_logs/`, `receipts/` | 実行ログ・提出専用ログ・完了phaseのハッシュ |

提出処理は `math_training/submission.py`、`rank64_12k/official_prompt.py`、同梱公式config/checkerをimportする。配布コードで変更可能な推論パラメータを使い、context8192・FP16・temperature0・全task最大2048・chunk32・seed42。単一モデル、1問1回答、best-of-N/self-consistency/検索/rerankingなし。回答の空白も編集しない。元の既定値完全再現ではなく許可パラメータの設定変更。公式データの取得はモデル選択確定後のみで、学習・選択へ渡さない。提出のstdout/stderrは権限を限定した`private_logs`へ保存する。

## 再利用・時間の見込み・検証

確認した既存資材：ルートREADME/実験結果、run01設定、`stage1_rank_ablation/engine.py`、`rank64_12k`一式、`math_training/{worker,published_data,c_utils,grading,submission}.py`、`e1_limo`、配布推論設定・checker。

再利用：固定データ取得と解説tokenize、GSM整形、スケジュール検査、学習/checkpoint/AMP/OOM再試行、vLLM評価、最後のboxed抽出・math-verify、paired bootstrap、公式message生成・JSON保存・形式チェッカー。新規部分は重複グループ分割、上限厳守の80:20 schedule、新規LoRA loader、fused native NLLへの局所接続、phase制御。旧コードは変更しない。Unslothのnative masked CEを使い、8192×語彙の全logitsをPython側へ保持しない。

既存Ada短系列SFTは6,000例で約12分（R12 README/Stage 1比較結果）。今回のGSMは2,000例・最大407 token、数学は約30万教師tokenだが最大8174長のため、単純な件数比例による断定はできない。暫定目安は学習合計5〜20分、merge/ロード5〜15分、公開評価2回＋提出推論10〜40分、合計20〜75分＋初回取得。**未実測の目安**であり、最初の更新のETAとphase実測を優先する。時間を理由に30万側学習や必要評価を省略しない。

```bash
.venv-local/bin/python -m pytest -q remote/math_curriculum/tests
.venv-local/bin/python -m pytest -q remote/math_training/tests/test_submission.py remote/math_training/tests/test_losses.py remote/math_training/tests/test_c_suite.py
bash -n remote/math_curriculum/launch.sh
```

公開データの実取得・分離・完全長tokenize・予算計算（[集計](local_validation.json)）、新規14テスト＋既存21テストで、予算上限/batch16、派生問題の分離、最後のboxed固定、採点不能分母、選択、失敗後再開・完了phaseスキップを検証。合成入力による公式形式チェッカー、採用モデル指定、回答無加工、提出再開も検証。GPUのUnsloth実行、48GBでの速度・VRAM、実正答率、公式JSONの実生成はリモート起動後に確認する。
