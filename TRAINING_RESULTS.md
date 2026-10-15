# 学習条件と提出結果

更新日：2026-10-14

提出スコアはユーザーから報告された結果。学習条件はリポジトリの実装・記録、学習時間は既存の実行記録に基づく。公式問題や提出JSONの本文は参照していない。

## 結果一覧

| 条件・結果 | run01：初回 | run02：教材・解説を変更 | run03：学習件数を増加 |
|---|---|---|---|
| RUN_ID | `t4_20261013_run01` | `t4_20261013_run02_prmfull` | `t4_20261014_run03_prmfull6k` |
| Baseモデル | Qwen/Qwen3-4B-Base | 同左 | 同左 |
| Stage 1 | Tuluの指示追従データ2,000件を学習 | run01のStage 1を再利用 | run01のStage 1を再利用 |
| Stage 2の教材 | GSM8K 2,000件 | PRM800K 1,500件＋GSM8K 500件 | PRM800K 4,500件＋GSM8K 1,500件 |
| Stage 2合計件数 | 2,000件 | 2,000件 | 6,000件 |
| 解説の扱い | 配布コードの`c1`圧縮 | 解説全文 | 解説全文 |
| Stage 2学習回数 | 1 epoch | 1 epoch | 1 epoch |
| Stage 2学習率 | `5e-5` | `5e-5` | `5e-5` |
| Stage 2学習時間 | 約18.5分 | 約23.5分 | 約68.8分 |
| 提出スコア | **0.14095** | **0.174014** | **0.18758** |
| 前回からの改善 | — | +0.033064 | +0.013566 |

学習時間には環境構築・モデル取得・マージ・推論を含まない。初回のStage 1学習は別途約23.5分。

最高スコアはrun03の **0.18758**。基準0.2011との差は **0.01352**、目標0.22との差は **0.03242**。いずれも未到達。

## 共通設定と実験の関係

| 項目 | 設定 |
|---|---|
| 学習環境 | Colab Tesla T4 |
| 学習方法 | 4bit QLoRA |
| LoRA | rank 8、alpha 16、dropout 0 |
| 学習精度 | FP16 |
| 最大系列長 | 2,048トークン |
| バッチ | 1件、勾配累積16、実効バッチ16件 |
| Stage 1 | `tulu3_persona_if` 2,000件、1 epoch、学習率`1e-4` |

run02・run03は、前のStage 2からの継続学習ではない。どちらも同じ「Base＋run01のStage 1」を出発点として、新しいStage 2 LoRAを学習した。

回答部分だけにlossを限定する変更は未導入。run02・run03は問題文を含む非padding部分を学習対象とする（先頭の因果境界ラベルがマスクされる場合を除く）。

run02・run03では固定した評価用800問を学習候補から除外し、重複を除去。全文が2,048トークンを超える例を除外し、採用した解説は途中で切らず、EOSを付加した。

## 確認済みの学習集計

| 項目 | run02 | run03 |
|---|---|---|
| 学習完了 | `complete=true` | `complete=true` |
| optimizer steps | 125 | 375 |
| 学習トークン数 | 528,362 | 1,598,551 |
| 学習時間（秒） | 1407.1929 | 4127.5154 |
| 平均train loss | 0.651883 | 0.608221 |

教材が異なるため、train lossの大小だけで汎化性能や過学習は判断できない。

## 結果の解釈

- run01→run02：PRM800K導入・解説全文の利用などをまとめて変更し、提出スコアが改善した。個々の変更の寄与は分離できない。
- run02→run03：教材比率と主要設定を維持し、学習件数を3倍にしてさらに改善した。1 epochのままなので更新回数も125→375に増え、学習時間は約3倍になった。
- データ量を増やす方針には改善の実績がある。ただし、さらに増やせば0.22を超えると保証する結果ではない。
- 固定devでのrun02/run03比較は未実施。ユーザーの希望により提出用推論へ進んだ。

## 実行コードと保存先

- run01学習：[notebook_all_t4.ipynb](colab_t4/notebook_all_t4.ipynb)
- run02学習：[run02_auto_t4.ipynb](colab_t4/experiments/run02_auto_t4.ipynb)。検査エラー修正後の再開には[resume_run02_t4.ipynb](colab_t4/experiments/resume_run02_t4.ipynb)を使用。
- run03学習：[run03_auto_t4.ipynb](colab_t4/experiments/run03_auto_t4.ipynb)
- 提出用推論：[run02_inference_t4.ipynb](colab_t4/experiments/run02_inference_t4.ipynb)。run03ではRUN_IDとStage 2の指定をrun03へ変更して使用。

Driveのワークスペースは `MyDrive/geniac_workspace/`。共通Stage 1は `adapters/t4_20261013_run01_stage1/`。各Stage 2は `adapters/<RUN_ID>_stage2/`、提出ファイルの設定上の保存先は `predictions/<RUN_ID>.json`。

run02・run03の集計は `experiments/<RUN_ID>/training_summary.json`。README内に残る「未実行」「候補」などの過去時点の説明より、本ファイルの結果一覧を最新状況として参照する。

## 次の実験案と論文調査（2026-10-14）

以下は提案であり、実装・実行済みの結果ではない。

### 第一候補：12,000件、1 epoch

run03と同じBase＋Stage 1から新しいStage 2を学習し、PRM800K 9,000件＋GSM8K 3,000件に増やす。学習率5e-5、rank8/alpha16、実効batch16、maxlen2048、全文解説、全非padding lossは維持する。run03のStage 2への追加学習ではない。

根拠の中心はこの実験で得たrun02→run03の改善。論文がこの具体的な設定で0.22達成を示したわけではない。新しい問題を増やした効果と更新回数増加の効果は未分離。

`local_eval/run03_preflight/summary.json`では、固定heldout・重複・系列長の除外後の候補はPRM11,388件、GSM7,273件。提案の件数はこの範囲内。12,000件版の実際のトークン数・重複除外・EOS検査は実装時に再確認する。

更新回数は約750。学習時間はrun03の68.8分を単純に2倍して約138分、余裕を見て2〜3時間を見込む（未実測、環境構築・マージ・公式推論は別）。GPUメモリの主要設定は変えない。推論時間が学習件数に比例して増えるわけではないが、学習後の生成長には依存する。

### 後続候補

1. 6,000件を2 epochs：同じ教材をもう1周見せる効果を調べる。Stage 1から新規学習し、run03と同じデータ・主要設定を使用する。1 epoch完了時のcosine学習率はほぼゼロなので、完了checkpointの単純再開と、最初から2 epochsのスケジュールで学習する実験は区別する。時間目安は約138分。12,000件×1 epochとの比較は、同程度の更新回数で新規問題と反復の効果を考える材料になる。
2. 解説・回答部分だけのloss：採用候補だが無条件に改善する修正ではない。問題文を条件として読む処理は維持し、その部分のlabelsだけを-100にする。token境界とEOSが正しく学習対象になるかを検査する必要がある。
3. LoRA rank16/alpha32：表現できる更新の規模を増やす候補。ただし現在rank8が性能の制約だと示す証拠はなく、優先度は下げる。

### 論文から得られる示唆と適用限界

- [Qwen2.5-Math Technical Report](https://arxiv.org/html/2409.12122v1)：1.5B/7B/72Bの数学モデルを扱い、SFTでは3 epochs、系列長4096を使用。複数epochは妥当な実験候補。ただし専用事前学習・データ・学習方式が異なり、今回の4B QLoRAへ数値をそのまま移せる証拠ではない。
- [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)：4Bを含む小型モデルに大型教師からの蒸留を使用。今回のルールでは別の生成モデルを教師として使えないため、この中心的手法をそのまま再現することはできない。
- [LIMO](https://arxiv.org/html/2502.03387v3)：主な比較対象はQwen2.5-32B-Instruct。少数の良質な推論例の有効性を示すが、強力な教師モデルによる解答生成・人手の確認を含む。今回、その教師・教材・人手選別を直接導入できない。「少数例なら必ず十分」とも解釈しない。
- [s1](https://arxiv.org/abs/2501.19393)：Qwen2.5-32B-Instructに1,000例でSFTし、推論中のbudget forcingを利用。データの質・難度・多様性という視点は参考になるが、指定外教材や推論ループ改変は今回の条件に合わない。
- [Instruction Fine-Tuning: Does Prompt Loss Matter?](https://aclanthology.org/2024.emnlp-main.1267/)：prompt lossの重みの有効性がデータと評価タスクで変わることを報告。回答のみlossを常に優位とする根拠にはならない。今回の数学4Bモデルでの直接検証ではない。

ローカルの概要・ルールDOCXの6.1〜6.4を再確認。指定外生成モデルによる教師・データ処理、指定外教材、データの人手加工・選別は禁止。推論コードの変更範囲も限定され、best-of-N/self-consistencyは禁止。自分がルールに沿って学習したモデルでの合成データ生成やGRPO等は選択肢だが、生成・検証・学習の計算量と実装負担が増えるため、次の1回としては優先しない。

系列長を2048→4096へ延ばす案も現時点では低優先。現在の前処理集計で長さ超過により除外されたPRM例は8件であり、今回の教材で最大長が主な制約だとは確認できない。全体スコアしかないため、推論の長さ打切りや回答抽出の失敗が主因かどうかも未確認。

まず12,000件×1 epochを1候補として実装し、データと学習完了を検査したうえで、既存の提出用推論を使う。公式提出は正常完了・形式検査後に行う。候補を一度に何本も投入せず、結果を見て次を決める。0.2011/0.22到達を保証するスコア予測は置かない。

## run04実行手順（コード作成済み、GPU実行は未実施）

RUN_ID：`t4_20261014_run04_prmfull12k`。日付をまたいでも変更不要。

### 学習

1. [run04_auto_t4.ipynb](colab_t4/experiments/run04_auto_t4.ipynb) をVS Codeで開く。
2. 新しいColab T4サーバーに接続。run03の学習に使用したGoogle Driveアカウントを使う。
3. 「すべて実行」を押し、Drive認証を行う。ZIPのアップロード・設定編集・途中の手動再起動は不要。
4. `RUN04_TRAINING_COMPLETE` が出れば学習完了。目安は学習だけで約2〜3時間、環境準備などは別。

中断した場合は同じノートブックを最初から実行する。保存済みcheckpointから再開し、完了済みなら重みを検証してスキップする。同時に複数起動しない。

### 提出用推論

1. 学習とは別の新しいColab T4で [run04_inference_t4.ipynb](colab_t4/experiments/run04_inference_t4.ipynb) を開く。**すべて実行は使わない。**
2. `[prepare-inputs]` → `[setup-install]` を実行し、カーネルを再起動。
3. `[config]` → `[hf-login]` → `[adapter-prepare]` → `[base-load]` → `[stage1-merge]` → `[stage2-merge]` → `[merged-save]` を順番に実行。トークンは非表示の入力欄へ入力する。
4. カーネルを再起動。ここではサーバーを削除しない（`/content`に保存したマージ済みモデルを使う）。
5. `[config]` → `[hf-login]` → `[adapter-prepare]` を再実行し、次に `[eval-data]` 以降を順番に実行。Base読み込み・マージは繰り返さない。
6. `[check-submission]` の成功を確認。Driveの `geniac_workspace/predictions/t4_20261014_run04_prmfull12k.json` をダウンロードしてOmnicampusに提出する。adapter-finalizeも実行する。HFへのモデル提出は別途必要。

MyDrive直下に以前置いた `inference_config.json` と `math_sft.tar.gz` を最初のセルで自動コピーする。run04のモデル指定は設定済み。既存run01〜run03のノートブック・アダプターは上書きしない。

VS Codeのカーネル再起動後も旧プロセスがGPUを保持することがある。Engine初期化エラー時は `!nvidia-smi` とログで状態を確認する。過去の固定PIDをkillするセルは今回のノートブックには含めていない。

生成元は [build_run04.py](colab_t4/experiments/build_run04.py)。実行にはIPythonが必要。学習版はrun03のコードから件数・識別子・ワーカー保存先・完了表示を変更。推論版は配布原本から作成し、configと依頼済みのgetpass認証以外の既存コードセルが一致することを検査する。新しいノートブックには過去の実行出力やwidget metadataをコピーしない。

## run04：アカウント変更後の公開Drive復旧

ユーザーは461/750までのログを報告。その後、アカウント変更時のDriveマウントが `Credentials propagation unsuccessful` で失敗したため、公開ファイルを直接取得する [resume_run04_public_t4.ipynb](colab_t4/experiments/resume_run04_public_t4.ipynb) を作成。

公開フォルダ一覧で確認できたcheckpointは50/100/150。150のtrainer_stateはglobal_step=150、max_steps=750。450が保存済みだとは確認できず、今回の復旧版は150を取得する。公開側に450が見えない原因は未確認。

新しいT4でこの復旧版を「すべて実行」する。Drive認証なしでStage 1、checkpoint-150、元のtraining_signatureをダウンロードし、元の条件一致検査を維持して再開する。記録された主要ライブラリを固定するが、GPUでの再開動作は未検証。条件不一致なら停止する。

保存先は `/content/run04_recovery/geniac_workspace`（一時領域）。完了後 `/content/run04_recovery_export.zip` を作るので、VS CodeのColabファイル一覧からダウンロードし、ZIP内のgeniac_workspaceをDriveへ配置する。公開共有は読み取り用であり、元Driveへ自動で書き戻す仕組みではない。ランタイム削除前に保存必須。元のプロセスが停止していることを確認して使用する。

## run04学習完了・Drive配置（2026-10-15）

ユーザーがダウンロードした `/Users/cls-lab/Downloads/run04_recovery_export.zip` を検査。training_summaryはcomplete=true、global_step=750、epoch=1.0。最終adapter_model.safetensorsのSHA-256が記録と一致することを確認した。再開セッションの実行時間は6991.8819秒（約116.5分）。公式推論・スコアはまだ未確認。

PCのGoogle Drive同期フォルダ経由で、元のgeniac_workspaceへ最終Stage 2とtraining_summary.jsonを配置した。既存Stage 1と過去runは変更していない。Stage 2の共有フォルダ：https://drive.google.com/drive/folders/19C31pIEakOJ5qg8s0tI57PSErnx6jKvY

### run04公開取得版の提出用推論

元Driveのマウント認証が再び失敗したため、ユーザー指定で `run04_inference_public_t4.ipynb` を追加。配布ファイル2個を原本のバイト列で同梱し、Stage 1/2の12ファイルを公開Driveから取得してSHA-256を検査する。Driveマウントを除去し、保存先を `/content/geniac_public_workspace` に変更。問題取得・推論・JSON生成・形式検査のコードは既存版と一致することを検査済み。HF認証は引き続き必要。

VS Codeで新しいT4サーバー `infer04` を作成。公開取得セルを実行し、配布物2個の準備と12ファイルのハッシュ一致が実際に成功した。現在の出力先はColabの一時領域なので、推論完了後の提出JSONはサーバー終了前にダウンロードする。

### run04公開取得版の実行状況（2026-10-15）

Colab T4 `infer04` で配布物・アダプター取得、依存関係インストール、HF認証、Base読込、Stage 1→Stage 2統合を実行。操作復帰時にStage 2を重複実行したため、カーネル再起動・Base再読込からやり直し、各Stageを一度ずつ統合した。最終統合モデルは8.06GB、保存時間11.7分。保存後に再起動し、GPU使用量0.0GBを確認。評価データ232問取得、vLLM起動2.7分、chat_template一致を確認した。

推論セルを開始し、chunk 1（58問）が実行中であることを確認。その後、PC操作ツールが `cgWindowNotFound` を返し、VS Code・Chrome双方の画面を取得できなくなったため、その時点以後の推論完了は未確認。カーネルは停止していない。提出も未実施。復帰時は再実行せず、まず既存セルの状態を確認する。

提出先は https://edu.omnicamp.us/courses/191/assignments/1092/ 。開始前の画面は提出3回／上限15回、直近スコア0.18758。ユーザーから今回の推論完了後の提出まで明示的に依頼済み。

## run04結果・run05短時間追加学習（2026-10-15）

ユーザー報告：run04は0.18308。run03の0.18758より0.00450低下。

run05_fast_t4.ipynbを作成。run03 Stage2の重みを引き継ぎ、元の検証済み6,000件キャッシュから固定seedでPRM288/GSM96件を抽出。問題文をlossから除き、回答とEOSのみ学習。rank8/alpha16、lr1e-5、実効batch16、maxlen2048。最大24更新、または学習開始から300秒経過を更新終了時に検知して停止する。コンパイル等を含む厳密な5分上限ではない。

推論出力上限は1024/1024/2048。温度0、seed42、232問は維持。元の評価取得・vLLM構築・推論・JSON保存・形式検査セルを文字列としてそのまま実行。親kernelはGPUライブラリを読み込まず、各段階を別プロセスで実行し再起動操作を省略。途中結果・最終JSON・ログ・学習アダプターをDriveへ保存。提出はユーザーが行う。

目標46〜65分だがGPU実測前で保証なし。Stage1復元と最終統合の保存、ダウンロードで超過し得る。ローカル検査：構文、回答開始位置不一致の拒否、回答EOSとpaddingのlossマスク、公式5セルの一致、vLLM子プロセス用mainガードを確認。GPU上の学習・推論は未実行。

Drive同期フォルダ geniac_workspace/run05_fast/ にノートブックとコードをコピー済み。起動はブラウザColabの新しいT4から「すべて実行」。元Driveの認証とHF token入力が必要。


## run06：演算記号スコアによる2,400件追加学習（2026-10-15、準備済み）

ユーザー指定の機械的選別。run04の検証済み12,000件のsolutionについて、等号/加減算1、乗算2、除算/分数/累乗3、根4、積分5の出現回数を合計。PRM上位1,800＋GSM上位600を選択し、同点はseed42のハッシュ順。本文は無編集。採用数2,400、重複なし、固定評価800問との重複なしを検査。図のasyコード等はスコアからのみ除外。

run06_calc2400_t4.ipynbはブラウザColab T4用の自己完結ノートブック。run04 Stage2の同じLoRAを継続し、response+EOS loss、lr1e-5、150更新、実効batch16、maxlen2048、rank8/alpha16。50更新ごとに保存・再開。推論5セルと出力上限はrun04と一致。別RUN_IDへ保存し、推論JSONのメタデータに追加学習条件を記録。

選別をローカルで実行済み。平均長462.18トークン（元269.62、1.71倍）、最大1974、合計1,109,233トークン。保存済みtokenizerとchat templateで2,400件の回答境界/EOS/長さを検査。記号計数、再現性、heldout拒否、lossマスク、推論コード保存の7テスト通過。GPU学習・推論は未実行、スコア未確認。時間概算は学習45〜60分、全体1.5〜2.5時間。採用教材中の積分記号は1回で、積分分野の十分な学習は期待できると未確認。

実行手順はcolab_t4/experiments/run06_実行手順.txt。Downloadsにもノートブック・手順・関連ファイルZIPを配置。

### run06 VS Code Colab版

ユーザー指定によりrun06_calc2400_vscode_t4.ipynbを追加。Driveマウントを除き、既存公開Stage1/run04 Stage2をハッシュ照合して取得。元12,000件キャッシュはgzip/base64で原本同梱。計算コードは保存先以外でDrive版と一致検査済み。結果は/content/run06_public/geniac_workspace、バックアップは/content/run06_export.zip。学習終了・推論終了時にcheckpointを含むZIPを作る。サーバー削除前のPC保存が必要。GPU未実行。

run06 VS Code版末尾にanywidgetによるZIP転送セルを追加。公式wikiの回避策に基づき1MiB単位で転送、完了時に自動保存を試行し保存ボタンも表示。PC保存完了はkernelから検証しない。ローカルで分割転送・再転送・構文を検査、VS Code UIの実動作は未確認。

### run06推論の保存先欠落を修正

保存済みrun06_status_check.ipynbの出力で、最初の58問を19.7分で生成後、predictionsディレクトリがなくpartial.jsonのopenでFileNotFoundErrorとなったことを確認。13:35 JSTのログ。14:07 JSTのGPU表示ではEngineCoreがメモリを保持しGPU使用率0%。最終JSON未作成、学習後ZIPは存在。推論開始前にpredictions.mkdir(parents=True, exist_ok=True)を追加し、既存ノートブックの出力を保持して修正。学習の再実行は不要。最初の58問は段階保存前のため再生成が必要。修正後GPU再実行は未確認。

### run06 学習済み重みの退避・推論専用再開

ユーザー希望によりrun06_save_trained_adapter.ipynbとrun06_resume_inference_vscode_t4.ipynbを作成。前者は150更新完了と重みハッシュを検査し、LoRA/tokenizer/完了記録をrun06_trained_adapter.zipへ保存・ダウンロード。後者はPC保存ZIPを/contentへアップロード後に検証復元し、Base+公開Stage1+保存run06を統合して推論のみ行う。trainフェーズは拒否。保存先mkdir欠落も修正。元58問は未保存のため推論は再生成。ローカルでバックアップ→復元の往復、ノートブックと埋め込みコードの構文を検査済み。実GPU再実行は未確認。
