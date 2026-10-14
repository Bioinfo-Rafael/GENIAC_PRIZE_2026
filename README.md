# GENIAC PRIZE 2026 Colab T4 実行記録と手順

## Linux / A6000でrun02を学習する準備（未実行）

Colab接続の代替として `remote/run02.py` を追加した。Linux/CUDA上で、入力ZIPの検証・配置 → Base＋既存Stage 1の統合 → 新規Stage 2の学習 → adapter保存を別プロセスで順に実行する。Driveの認証・notebookのカーネル選択・手動再起動は不要。PRM 1,500＋GSM 500、1 epoch、FP16/4bit LoRAなどの条件はColab版run02と同じ。Stage 1は学習し直さずrun01を再利用する。

2026年10月14日、[Bioinfo-Rafael/GENIAC_PRIZE_2026](https://github.com/Bioinfo-Rafael/GENIAC_PRIZE_2026) を非公開で作成し、初回commit `3a2b9ce` をpushした。VS CodeのRemote Tunnel `rafael` への接続は確認したが、PC操作ツールのタイムアウトにより接続先GPUの確認、clone、GPU学習はまだ未実施。A6000での動作・所要時間・改善スコアは未検証。`remote/requirements-train.txt` は導入用リストであり、検証済みの固定環境ではない。実際に導入された版は `logs/packages.txt` に保存する。既存checkpointから再開する場合、ライブラリ更新を避けて同じvenvを使う。

コードをcloneしたLinuxマシンでPython 3.11または3.12のvenvを作り、別途転送した `run02_colab_inputs.zip` を指定する。GitHubにはこのZIP、adapter、教材、実行済みnotebook出力、認証情報を含めない。初回は `git clone <確定したリポジトリURL>`、以後のコード更新時は `git pull --ff-only` を使う。

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -u remote/run02.py --install \
  --bundle ~/run02_colab_inputs.zip \
  --workspace ~/geniac_workspace
```

SSHが切れても継続したい場合は、上記をサーバー上の `tmux` セッション内で実行する。複数GPUのマシンでは利用可能なGPUを確認して `CUDA_VISIBLE_DEVICES` を指定する。使用中の他ジョブを停止しない。

再開は同じvenv・workspaceで、`--install` を外して同じコマンドを使う。50 stepごとのcheckpointから再開し、完了済みなら重みのハッシュを確認してスキップする。同一workspaceへの二重起動はロックで防ぐ。`RUN02_TRAINING_COMPLETE` が完了の目印。出力は `~/geniac_workspace/adapters/cuda_20261013_run02_prmfull_stage2/`、集計は `~/geniac_workspace/experiments/cuda_20261013_run02_prmfull/training_summary.json`。

このコマンドは学習と保存まで。開発用評価・公式推論・提出は含まない。公式推論には指定コードとT4での時間条件が引き続きある。現在の概要・ルールは「Colabとローカル」を認めているが、今回のリモートサーバーがその範囲に含まれるかは未確認。

公式推論用に [run02_inference_t4.ipynb](colab_t4/experiments/run02_inference_t4.ipynb) を準備済み。配布原本のconfigセルにあるRUN_ID・Stage 1/2の保存先名・recipe説明だけを変更した。その後、ユーザー指定によりVS Code対応として `[hf-login]` のトークン取得を `userdata.get` から `getpass` の非表示入力へ変更した。この認証セルは配布原本との差分となる。モデル推論・問題取得・提出形式の処理は変更していない。実行出力は空。実行前に、完成したrun02 Stage 2と既存Stage 1を指定のDrive保存先へ転送する必要がある。公式推論・提出はまだ未実行。2026年10月14日、ユーザーからColab版run02で `RUN02_TRAINING_COMPLETE` が表示されたとの報告を受領。attention_mask検査を修正した再開用notebookで学習を完了した流れ。推論notebookのRUN_IDとStage 2保存先を `t4_20261013_run02_prmfull` に修正した。学習集計の内容・改善スコアはまだ未確認。

`remote/run02_worker.py` は通常のPythonコード。元の `train_stage2.ipynb` を更新した場合は `python3 remote/build_worker.py` で再生成する。元の実行済みnotebookは変更しない。

2026年10月13日に Downloads 内の直近5分の ZIP 2つをこのディレクトリへ展開した。原本は直下、設定済みの実行用コピーは `colab_t4/` にある。ZIP 自体は Downloads に残している。

課題は Qwen3-4B-Base の数学性能を追加学習で高め、指定の推論コードが生成する JSON を Omnicampus に提出すること。提供された「概要・ルール」2026/10/02版では、合格条件は期間中の最高スコア **0.2011以上**。提出は失敗分も含め累計15回まで。JSON の締切は **2026年10月15日12:00**、開発モデルの Hugging Face アップロード期限は **同日17:00**。時刻は提供資料の表記に従う。Omnicampus 使用方法の資料は今回の ZIP に含まれていない。

**訂正（2026年10月13日）:** 以前ここに記載した「6.1 は Colab の実行操作をエージェントに委任することを禁止している」という断定は誤り。同梱の「概要・ルール」原文には、その禁止文は確認できない。6.4 は推論コードの編集をモデル情報・vLLM の推論パラメータ等に制限し、6.5 は推論用データを外部サービスやコーディングエージェントに読ませることを禁止している。現在の文書では学習・推論・データ処理にColabとローカルを認めている（下記追記）。エージェントは大会の推論用データや回答プレビューを閲覧しない。

## 今回実際に実行したもの（2026年10月13日確認）

性能改善の次の実験は、[数学SFTの論文調査と実装計画](research/20261013_math_sft_plan.md) を参照。現行設定の由来、Colab内の開発用評価、教材・解説圧縮・lossの比較順を整理している。開発用の [evaluate_dev.ipynb](colab_t4/experiments/evaluate_dev.ipynb) を作成済み。新しい学習コードとColab上での評価・改善スコアの確認はまだ行っていない。

**今回の流れは「統合版で学習 → 別の推論用ノートブックでmerge・推論 → Omnicampusに提出」。ファイル名の番号順に全部を実行したわけではない。**

RUN_IDは `t4_20261013_run01`。提出スコアはユーザー報告で **0.14095**。合格ラインの0.2011には未到達。
以下は保存済みノートブックのコード・実行カウントと学習ログ、およびユーザーの提出報告から整理したもの。推論用データ・回答本文は閲覧していない。

### ファイルごとの役割と実行状況

| ファイル | 役割 | 今回の実行状況 |
| --- | --- | --- |
| [colab_t4/notebook_all_t4.ipynb](colab_t4/notebook_all_t4.ipynb) | 学習から推論までのコードを内包する統合版 | **学習に使用**。保存時の `PHASE="train"` と、Stage 1・2の学習完了・adapter保存ログを確認。統合版の推論工程を使った記録は今回の保存ファイルにはない |
| [colab_t4/notebook2_inference_t4.ipynb](colab_t4/notebook2_inference_t4.ipynb) | 学習済みadapterを読み込み、merge・推論・JSON生成を行う | **学習後に使用**。モデル読込、Stage 1・2のmerge、推論、JSON保存、形式検査、adapter-finalizeの実行履歴あり |
| [colab_t4/notebook1_sft_t4.ipynb](colab_t4/notebook1_sft_t4.ipynb) | 学習専用版。Stage 1・2のコードを通常のセルで読める | **このファイル自体の実行履歴なし**。ただし `stage1-adapter`・`stage2-sft` と同一のコードが統合版に埋め込まれ、そこで実行された |
| [notebook1_sft.ipynb](notebook1_sft.ipynb) | 運営配布の学習用原本 | 実行履歴なし。学習設定は未指定のまま |
| [notebook2_inference.ipynb](notebook2_inference.ipynb) | 運営配布の推論用原本 | 実行履歴なし。今回使ったのは `colab_t4/` 内のコピー |

「実行履歴なし」は、保存ファイルに実行カウント・出力がないという意味。過去の別セッションで一度も実行していないことまで証明するものではない。

### 今回の実行の流れ

```text
1. colab_t4/notebook_all_t4.ipynb：PHASE="train"
   元のQwen3-4B-Baseをロード
     → Stage 1：Tulu 2,000件で学習（23.5分）
     → adapter AをDriveに保存
     → Stage 1の変更を組み込んだモデルをColab内に保存
     → そのモデルを読み直す
     → Stage 2：GSM8K 2,000件で学習（18.5分）
     → adapter BをDriveに保存、adapter検証

2. colab_t4/notebook2_inference_t4.ipynb に切り替え
   推論環境の準備・設定・adapterの読込準備
     → [base-load] 元のQwen3-4B-Baseをロード
     → [stage1-merge] 保存済みadapter Aを組み込む
     → [stage2-merge] 保存済みadapter Bを組み込む
     → [merged-save] 組み込み済みモデルを保存
     → [llm-build] vLLMで組み込み済みモデルをロード
     → [inference] 大会の問題への回答を生成
     → [predictions-save] 提出用JSONを保存
     → [check-submission] JSONの形式検査
     → [adapter-finalize] adapterのchat templateを確認・整備

3. OmnicampusにJSONを提出
   ユーザー報告スコア：0.14095
```

上図は確認できる工程のつながり。環境切替・再起動・途中の再実行を含む厳密な操作時系列は、保存履歴だけでは復元できない。セルの実行番号は再起動するとリセットされ、再実行で前後するため、番号だけで全工程を並べ替えない。再実行時の環境切替・再起動手順は後述を参照。

**推論用ノートブックの `stage1-merge` / `stage2-merge` は学習ではない。学習は統合版で完了しており、ここでは保存された変更部分をモデルに適用している。**

### コードを読む場所

統合版はコードを圧縮文字列として内包し、`[run-phase]` の `_u_execute()` が `_u_payload[group][label]` を取り出して `exec()` で実行する。そのため、学習の中身を読むには `notebook1_sft_t4.ipynb` の対応セルが分かりやすい。

| 知りたい処理 | 読むファイル・セル | 主なコード |
| --- | --- | --- |
| データセット・件数・学習率など | 学習専用版の `[config]` | `STAGE1_DATASET`、`STAGE2_DATASET`、`N_STAGE1`、`N_STAGE2` |
| Stage 1のモデル作成・重み読込 | 学習専用版の `[stage1-adapter]` | `FastLanguageModel.from_pretrained(model_name=BASE_MODEL, ...)` |
| Stage 1の学習 | 同じ `[stage1-adapter]` | `get_peft_model(...)` → `SFTTrainer(...)` → `trainer.train()` |
| Stage 2のモデル作成・重み読込 | 学習専用版の `[stage2-sft]` | `FastLanguageModel.from_pretrained(model_name=STAGE1_MERGED_LOCAL, ...)` |
| Stage 2の学習 | 同じ `[stage2-sft]` | 新しいLoRAを追加 → `trainer.train()` |
| 推論前の元モデル読込 | 推論用コピーの `[base-load]` | `AutoModelForCausalLM.from_pretrained(...)` |
| 学習済み変更部分の読込と組込 | 推論用コピーの `[stage1-merge]` / `[stage2-merge]` | `PeftModel.from_pretrained(...)` → `merge_and_unload()` |
| 回答生成用モデルの読込 | 推論用コピーの `[llm-build]` | `llm = LLM(model=INFERENCE_MODEL, ...)` |

学習専用版を編集しても、既存の統合版に埋め込まれたコードは自動では更新されない。次の実験では実行するファイルと設定を揃える。`build_unified.py` は統合版を生成するためのスクリプトで、既存の実行済み統合版を上書きするため、履歴を保存せずに再生成しない。

### 保存先と未確認事項

- Stage 1 adapter：`MyDrive/geniac_workspace/adapters/t4_20261013_run01_stage1/`
- Stage 2 adapter：`MyDrive/geniac_workspace/adapters/t4_20261013_run01_stage2/`
- 提出JSONの設定上の保存先：`MyDrive/geniac_workspace/predictions/t4_20261013_run01.json`
- 学習時間は合計約42分。環境構築、ダウンロード、merge、推論の時間は含まない。
- 推論全体の所要時間と提出累計回数は、この確認では未確定。
- Hugging Faceへのadapterアップロード完了は未確認。Driveへの保存や `[adapter-finalize]` の実行とは別の工程。
- 統合版末尾の `%run -i /content/export_colab_run.py` セルに実行履歴はない。エクスポート完了は未確認。

## run02の実行手順（操作を減らした自動起動版・推奨）

使用するのは [run02_auto_t4.ipynb](colab_t4/experiments/run02_auto_t4.ipynb) と [run02_colab_inputs.zip](colab_t4/experiments/run02_colab_inputs.zip)（約129MB）。以下は学習開始からStage 2保存までの手順。**この自動版ではPHASEの編集も、途中のセッション再起動も不要。** 下の手動版 `train_stage2.ipynb` と同時には実行しない。

1. 別アカウントに変える場合だけ、VS Codeで `⌘⇧P` → `Colab: Sign Out`。後の接続時に開くGoogle認証画面で使用するアカウントを選ぶ。
2. VS Codeで `colab_t4/experiments/run02_auto_t4.ipynb` を開く。
3. ノートブック右上の「カーネルの選択」→「Colab」→「New Colab Server」から **T4 GPUの新しいサーバー** を選ぶ。必要なら「別のカーネルを選択」からColabへ進む。ローカルPythonや既存のvLLM推論サーバーを選ばない。
4. VS Code左側のExplorerで同じディレクトリの `run02_colab_inputs.zip` を右クリック → **Upload to Colab**。複数サーバーが表示されたら今作ったT4を指定。アップロード完了通知を待つ。保存先は `/content/run02_colab_inputs.zip`。ZIPは自分で展開しない。
5. ノートブック上部の **「すべて実行」** を1回押す。設定セルは編集しない。
6. Driveの認証を求められたら、保存先にするGoogleアカウントで認証する。ブラウザからVS Codeへ戻る案内が出たら戻る。以後は自動で進む。

上記の接続・アカウント選択・アップロード操作は [Colab拡張の公式User Guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide) に基づく。Google認証とGPU確保はユーザー側の操作・サービス状態に依存するため完全には省略できない。

自動で行う工程：ZIPの整合性確認 → Driveへの配置 → 学習環境の導入 → 公式Baseの取得 → 保存済みStage 1の統合 → Stage 2の追加学習 → adapter・集計のDrive保存。mergeとtrainを別Pythonプロセスで実行し、モデル用GPUメモリを工程間で解放する。親のnotebook kernelには学習ライブラリを読み込まないため、パッケージ導入後の手動再起動を不要にしている。

ZIPにはrun01のStage 1/2、元の学習キャッシュ2つ、固定したdev/finalと除外ハッシュを同梱。新しいGoogleアカウントでも既存フォルダの手動コピーは不要。既存ファイルと内容が違えば停止し、上書きしない。約8GBのBaseモデルはZIPに含まれず、Colabがダウンロードする。

### 完了を確認する

最後に **`RUN02_TRAINING_COMPLETE`** が出れば学習完了。

- 新Stage 2：`MyDrive/geniac_workspace/adapters/t4_20261013_run02_prmfull_stage2/`
- 共有用の学習集計：`MyDrive/geniac_workspace/experiments/t4_20261013_run02_prmfull/training_summary.json`
- ログ：同じrunフォルダの `logs/install.log`、`logs/merge.log`、`logs/train.log`

学習完了はスコア0.22達成を意味しない。ここまでで生成評価・公式提出・HF公開は行われない。まず学習集計を確認し、次に保存されたrun02で開発評価または公式推論へ進む。

### 止まった場合

- ZIPがない：アップロード完了を確認し、「すべて実行」をやり直す。
- T4に接続できない：GPUが確保できるまで学習は開始できない。CPUでは続行しない。
- 認証で止まった：Drive認証を完了する。拡張内で認証が進まない場合はWeb版Colabで同じnotebookをアップロードし、T4とZIPを用意して実行する。
- 学習が中断：同じサーバーなら「すべて実行」をやり直す。条件が一致すれば直近のcheckpointから再開。サーバーが消えた場合は新しいT4への接続とZIPアップロードもやり直す。
- 最初の50 optimizer step以前にはcheckpointがないので、その場合はStage 2を開始から実行する。
- それ以外のエラーはメッセージかログを共有する。エラー後のセルだけを実行して進めない。

Colab上のGPU実行はまだ未検証。ローカルではnotebook形式・コードの構文、ZIP展開と上書き防止、子プロセスの失敗検出、既存の教材・評価テストを確認した。キャンセルされたMac評価は再開していない。

## 次の学習候補：run02（PRM中心、解説全文）

[train_stage2.ipynb](colab_t4/experiments/train_stage2.ipynb) を作成した。Colab T4の新しい学習ランタイムで、この1冊をアップロードする。

1. `PHASE="setup"` → すべて実行 → セッション再起動。
2. `PHASE="merge"` → すべて実行 → セッション再起動（ランタイムを削除しない）。
3. `PHASE="train"` → すべて実行。

Stage 1は `t4_20261013_run01_stage1` をそのまま再利用し、Stage 2を新規学習する。元のStage 2からの継続学習ではない。

| 項目 | run02 |
| --- | --- |
| Stage 2教材 | PRM800K 1,500件＋GSM8K 500件 |
| 解説 | 全文。EOS込み2048以内の例を機械的に採用 |
| 変更しない主要設定 | 1 epoch、学習率5e-5、rank 8 / alpha 16、実効batch 16 |
| loss | 非padding全体。回答のみlossへの変更は今回は行わない |
| 学習から除外 | 固定dev/final計800問 |
| 途中保存 | 50 optimizer stepごと、直近2checkpoint。条件一致時のみ再開 |
| 新Stage 2保存先 | `MyDrive/geniac_workspace/adapters/t4_20261013_run02_prmfull_stage2/` |
| 学習集計 | `MyDrive/geniac_workspace/experiments/t4_20261013_run02_prmfull/training_summary.json` |

教材の取得・除外・トークン化はローカルで実行済み。2,000件、合計528,362トークン、平均264.181、最大1,923。PRM候補のうち全文が2048を超える8件を除外し、採用例の解説は途中で切らない。GSMの元設定は平均約163トークンだったため、学習時間はrun01の18.5分と同じとは仮定しない。

**ColabでのGPU学習と生成評価は未実行。0.22超えを確認した設定ではない。** 教材を変える最初の候補として用意した。学習済みrun01は上書きしない。実際のlabelsとEOS保持を学習前に検査する。

学習後の推論は `STAGE1_SLUG="t4_20261013_run01_stage1"`、`STAGE2_SLUG="t4_20261013_run02_prmfull_stage2"`、`RUN_ID="t4_20261013_run02_prmfull"` を指定する。開発評価で使う場合は今回の学習キャッシュも `TRAIN_CACHE_SPECS` に追加し、同じ固定問題・同じvLLM環境でrun01と比べる。

実装の正本は [stage2_recipe.py](colab_t4/experiments/stage2_recipe.py) と [build_train_stage2.py](colab_t4/experiments/build_train_stage2.py)。除外ハッシュ・固定revisionは [run02_reference.json](colab_t4/experiments/run02_reference.json) に保存。元の実行済みnotebookは変更していない。

## Macでの開発評価（2026年10月13日追記）

現在の同梱「概要・ルール」5、6.1〜6.3にはColabとローカルを認める記載がある。以前の「Colab限定」という案内は現在のファイルには当てはまらない。提出用推論コードの制約と、開発用評価の実行場所は区別する。

Downloadsの `drive-download-20261013T135027Z-1-001.zip` から、学習キャッシュ2つ（各2,000件）だけを `data_cache/` へ展開した。Stage 1・2のadapterは `Downloads/drive-download-20261013T094957Z-1-001/` を使う。

MacのApple GPU向け入口は [evaluate_local.py](colab_t4/experiments/evaluate_local.py)。プロジェクト直下で次を実行する。

```sh
.venv-local/bin/python -u colab_t4/experiments/evaluate_local.py
```

`local_eval/` に固定split、将来の学習から除外するハッシュ、モデルrevision、結果を保存する。`local_eval/runs/run01_quick_mps_v1/summary.json` が最終集計。各問終了時に `predictions_private.json` を保存し、同じ条件で再開可能。問題や回答本文はエージェントへ渡さず、集計だけを用いる。

Colab版と同じ公式Base＋adapter A/BをFP16で統合するが、生成は **Transformers/MPS**。vLLMと数値的に同一とは限らず、速度も別物なので、Macでの候補比較として扱う。公式スコア・Colab版の値と直接混ぜない。元の `evaluate_dev.ipynb` は引き続きColab用。実行の完了状況はログとsummaryの有無で確認する。

2026年10月13日のローカル事前診断：GSM8Kの学習2,000件中、c1で解説本文の末尾が削られるのは30件（1.5%）。全文でも2048トークン超は0件。保存済みadapterのchat templateは推論用設定と一致した。これは保存済みtokenizerによる再計算で、過去のSFT labels/lossや生成正答率を測定した結果ではない。

## 次に実行するもの：開発用評価

[evaluate_dev.ipynb](colab_t4/experiments/evaluate_dev.ipynb) を **新しいColab T4** で開く。この1冊だけで動く構成で、補助ファイルのアップロードは不要。実行済みの学習・推論ノートブックは変更していない。

1. `PHASE="setup"` で「すべて実行」し、セッションを再起動する。
2. `PHASE="merge"` で「すべて実行」し、セッションを再起動する。ランタイムは削除しない。
3. `PHASE="evaluate"` で「すべて実行」する。初回は `EVAL_SIZE="quick"` の128問。
4. Driveの `geniac_workspace/dev_eval/runs/run01_quick_v1/summary.json` を確認・共有する。問題・生成回答本文の共有は不要。

既存run01のadapter A/BをFP16で統合し、学習済み問題を除いたPRM800K math_splits/trainとGSM8K main/trainを使う。学習データのキャッシュがない場合は停止する。別の学習履歴がある場合は設定の `TRAIN_CACHE_SPECS` に追加する。指定外の学習履歴や意味的重複は自動検出できない。

正答率、答案抽出失敗、出力長による打切り、生成トークン数・時間、c1による解説削減率、全文の2048トークン超過率を保存する。数式の機械採点には限界があり、正解側の解析失敗も別途集計する。**この開発正答率は公式スコアと別の指標。提出回数は消費しない。**

quickは固定dev400問の部分集合。候補比較を400問に進める場合は `EVAL_SIZE="dev"` と別の `EVAL_RUN_ID` を設定する。最終確認用400問も別途確保し、このノートブックでは採点しない。次回学習では `dev_eval/heldout_problem_hashes.json` の800問を除外する必要がある。**既存学習ノートブックにはまだ除外処理がないため、次の学習コード作成時に追加する。**

途中回答はchunkごとにDriveへ保存。同じID・モデル・問題・生成条件・ライブラリ版なら再開し、条件の異なる結果は混ぜない。Colabでの実際の環境構築・merge・生成は未検証。ローカルではコードのみを検証し、16件のテスト（採点、重複除外、再開、notebook形式）が通過した。実データやモデルはローカルで取得・実行していない。

開発者向け：読みやすい補助関数の正本は [dev_eval.py](colab_t4/experiments/dev_eval.py)。[build_evaluate_dev.py](colab_t4/experiments/build_evaluate_dev.py) が補助関数と配布テンプレートを通常セルへ埋め込み、自己完結したnotebookを生成する。補助関数を変更したら生成スクリプトを実行する。これは**未実行の開発notebook**を上書きするため、実行ログ付きコピーは別名で保存する。

## 以下の手順の読み方

以下は初期に用意した実行手順。今回の実行履歴そのものは上記を参照する。「1 学習」は学習専用版を使う場合の代替手順であり、今回それを直接実行したという意味ではない。

## 最初に用意するもの

### 統合ノートブック

`colab_t4/notebook_all_t4.ipynb` に学習から推論・JSON形式検査までをまとめた。
補助ファイルは同梱済み。VS Codeではこのファイルを開き、カーネル選択から **Colab → T4** に接続する。
各工程は先頭の `PHASE` を指定して「すべて実行」で進める。

1. `setup_train` → カーネル再起動。
2. `train` → 学習済みadapterをDriveへ保存。新しいT4ランタイムへ切り替える。
3. `setup_inference` → カーネル再起動。
4. `merge` → **カーネルのみ再起動**。ランタイムを削除しない。
5. `inference` → JSON保存、形式検査、adapterのtemplate確認まで実行。

1冊に統合しているが、環境切替・再起動をまたぐ無人連続実行ではない。
学習済みadapterが存在する場合は上書きせず停止する。途中Stageからの学習復旧は既存の学習ノートブックで行う。
推論の詳細出力はColab内の `/content/geniac_private_inference.log` に留め、エージェントは開かない。
HFへのモデルアップロードは後述の手順を使用する。

VS Code拡張の[公式既知問題](https://github.com/googlecolab/colab-vscode/wiki/Known-Issues-and-Workarounds)では
`drive.mount()` は対応済み、`userdata.get()` は未対応とされている。
後者で認証が止まる場合、トークンをチャットへ貼らず、Web版Colabで同じノートブックを開いてSecretsを利用する。
推論コードの認証処理も同梱原本のままなので、この制限を黙って回避する書換えは加えていない。

統合版の作成時の静的検証は `colab_t4/unified_validation.json` に記録。その後、統合版で学習し、別の推論用コピーで推論した履歴を確認済み（冒頭参照）。

### アカウントと認証

1. Google アカウント、Hugging Face アカウント、Omnicampus アカウント。
2. Google Drive の空き容量。配布ルールでは10GB以上を推奨。
3. Hugging Face にログインし、以下の利用条件に本人が同意する。**評価データの内容は閲覧しない**。
   https://huggingface.co/datasets/geniac-prize-physical-ai-competition-2026/compe_screening_eval
4. https://huggingface.co/settings/tokens で read token を作成する。
5. Colab 左側の鍵アイコンから Secrets に `HFH_READ` を登録し、各ノートブックからのアクセスを有効にする。トークンをコードやチャットに貼らない。

## 1 学習

1. https://colab.research.google.com/ の「ファイル → ノートブックをアップロード」で `colab_t4/notebook1_sft_t4.ipynb` を開く。
2. 「ランタイム → ランタイムのタイプを変更」で GPU を T4 にする。
3. 左のファイルペインから、`colab_t4/math_sft.tar.gz` と `colab_t4/inference_config.json` を `/content/` へアップロードする。tar は自分で展開しなくてよい。
4. `[setup-install]` セルを実行する。エラーが出た場合は先に進めない。
5. 「ランタイム → セッションを再起動する」を実施する。
6. `[setup-verify]` から下をセル順に実行する。Drive のマウントと認証を行い、最後の `[adapter-verify]` まで進む。
7. 2つの adapter が Drive の `MyDrive/geniac_workspace/adapters/` に保存されたことを確認する。`ADAPTER_STORAGE="drive"` なので、この時点では Hugging Face へはアップロードしない。

最初から「すべてのセルを実行」は使わない。途中に手動の再起動がある。

### 初回設定

| 項目 | 設定 |
| --- | --- |
| RUN_ID | `t4_20261013_run01` |
| Stage 1 | `tulu3_persona_if` 2,000件、1 epoch、学習率 `1e-4` |
| Stage 2 | `gsm8k` 2,000件、1 epoch、学習率 `5e-5`、配布コードの c1 圧縮 |
| QLoRA | 4bit、rank 8、alpha 16、dropout 0 |
| 学習精度 | fp16 |
| 系列長 | 2048 |
| バッチ | 1、勾配累積16、実効バッチ16 |
| 保存先名 | `t4_20261013_run01_stage1` / `t4_20261013_run01_stage2` |

この設定で学習した記録があり、今回の提出スコアはユーザー報告で0.14095。合格スコアには未到達。実測の学習時間はStage 1が23.5分、Stage 2が18.5分。GPU割当・依存ライブラリの導入・モデル取得・mergeにも別途時間が必要。

## 2 推論と JSON 作成

1. 学習用セッションを終了し、**新しい T4 ランタイム**で `colab_t4/notebook2_inference_t4.ipynb` を開く。学習用の Unsloth 環境と推論用 vLLM 環境を混ぜない。
2. 新しい `/content/` に `math_sft.tar.gz` と `inference_config.json` を再アップロードし、Secrets の `HFH_READ` へのアクセスを有効にする。
3. `[setup-install]` を実行し、ノートブックの案内どおり **セッションを再起動**する。
4. `[config]` → `[hf-login]` → `[adapter-prepare]` → `[base-load]` → `[stage1-merge]` → `[stage2-merge]` → `[merged-save]` の順で実行する。
5. merge 保存が完了したら、もう一度 **セッションを再起動**する。このとき「切断してランタイムを削除」は使わない。通常の再起動なら `/content/` の merge 済みモデルが残る。
6. 再起動後は `[config]` → `[hf-login]` → `[adapter-prepare]` → `[eval-data]` → `[llm-build]` → `[inference]` → `[predictions-save]` → `[check-submission]` → `[adapter-finalize]` の順で実行する。`[base-load]`〜`[merged-save]` は再実行しない。
7. 形式チェックの成功を確認して、Drive から `geniac_workspace/predictions/t4_20261013_run01.json` をダウンロードする。中身を編集せず Omnicampus に提出する。

推論の所要時間は配布資料で約100分とされている。ルール上は無料版T4で150分以内に最後まで実行できる必要がある。今回の推論全体の実測時間は、この確認では未確定。

推論は配布設定のまま、float16・系列長4096・GPU使用率0.85・temperature 0・58問単位の途中保存。初期作成時は RUN_ID、adapter 名、recipe 表記を設定した。その後の実行済みコピーには `[llm-build]` のCUDAライブラリ読込・Colab互換処理などの追加もあるため、「設定セルだけを変更した状態」ではない。作成時の静的検証だけでは現在のコピーの変更範囲を保証しない。推論用コードの編集制限との適合は別途確認が必要。

## 3 開発モデルを Hugging Face に保存

推論最後の `[adapter-finalize]` 完了後に、2つの adapter をアップロードする。

1. 学習用ノートブックを T4 ランタイムで開く。新規環境なら `/content/` に2つの補助ファイルを置き、`[setup-install]` → 再起動 → `[setup-verify]` を実施する。
2. Secrets に write token を `HFH_WRITE` として登録し、このノートブックからのアクセスを有効にする。
3. `[config]` の `ADAPTER_STORAGE` だけを `"both"` に変更する。RUN_ID と保存先名はそのまま。
4. `[config]` → `[workspace-init]` → `[hf-login]` → `[adapter-push]` を実行する。学習セルは実行しない。既存 adapter の警告は保存済みファイルがあるためで、push のために学習をやり直す必要はない。
5. 2つの private リポジトリが作られ、`adapter_config.json` と重みがあることを確認する。`colab_t4/MODEL_CARD_TEMPLATE.md` を参考に出典とロード順をそれぞれの README へ記載する。学習済みであること・実際の設定を確認してから記載する。
6. 期限後はモデルを更新せず、結果発表まで保存する。運営から公開を求められたら対応する。

## エラーや中断が起きた場合

- 学習中の OOM は、他の GPU セッションを終了したうえで発生セルとライブラリの版を確認する。初回コピーは batch 1 に設定済み。追加で系列長を減らす場合は別実験として識別する。
- 推論の初期化でメモリ不足なら、merge 保存後の再起動を確認する。なお失敗する場合は、編集可能な `ENFORCE_EAGER=True` を試せる。速度は変わるため150分条件を再確認する。
- 依存ライブラリのインストールや import に失敗した場合、問題文・出力回答・トークンを含まないエラーログだけを共有する。特に推論用の編集禁止セルを自己判断で変更しない。
- 通常のセッション再起動ではローカルファイルが残るが、ランタイム削除では消える。推論ランタイムを失った場合は補助ファイルを再アップロードし、環境構築と adapter merge からやり直す。同じモデルと設定、同じ RUN_ID なら配布コードが Drive の途中結果を使う。
- 学習は各 Stage の最後に adapter を保存し、途中の自動 checkpoint はない。途中で切れた Stage の学習進捗は復元できない。保存済み adapter の上書きに注意する。
- 学習条件やモデルを変える場合は、両ノートブックの RUN_ID と Stage 1 / 2 の保存先名を新しく揃える。異なるモデルや推論設定の途中結果を混ぜない。

## 原本の確認と検証範囲

ZIP の CRC と展開ファイルのバイト一致は全件成功。展開原本に変更はない。一方、**同梱の SHA256SUMS.txt とは以下3点が不一致**だった。

- `notebook1_sft.ipynb`
- `notebook2_inference.ipynb`
- `math_sft.tar.gz`

`check_submission.py` と `inference_config.json` は一致。原因は未確認。配布更新後のハッシュ未更新の可能性はあるが断定できない。実行前に配布元の最新版・ハッシュを照合する。詳細は `integrity_report.json`。元のハッシュファイルは書き換えていない。

初期準備時はローカルでPythonセルの構文、学習・推論設定の一致、当時の推論コードの編集範囲、展開の完全性を確認した。その後のColab実行履歴とユーザー報告は冒頭に追記した。今回のREADME更新ではコード・実行履歴を確認し、モデルの実行や推論用データ・回答本文の閲覧は行っていない。

学習用コピーでは install 失敗時の停止、Unsloth の先行 import、`SFTTrainer(processing_class=...)`、`use_exact_model_name=True` を追加した。取得元は `Qwen/Qwen3-4B-Base` のまま、4bit化は Colab で行う。依存パッケージは配布と同じ未固定インストールなので、環境変化によるエラーは実機で確認する必要がある。

参照した API: [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)、[Unsloth model loader](https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py)。大会条件はこのフォルダの配布文書を参照し、9/28の説明スライドより後の「概要・ルール」を優先した。


## run02の結果確認とrun03候補（2026年10月14日）

ユーザー報告のrun02公式スコアは **0.174014**（run01 0.14095から+0.033064）。共有Driveの `experiments/t4_20261013_run02_prmfull/training_summary.json` と `logs/train.log` を取得して確認した。`complete=true`、1 epoch、125 optimizer steps、学習時間1407.1929秒（約23.5分）、平均train loss 0.651883。表示されたlossは序盤0.9451、最終区間0.6551。学習lossだけでは汎化性能・過学習の有無は判定できない。以前のattention_mask検査エラーの履歴はログに残るが、最終集計は完了を示す。公式問題・提出JSON本文は閲覧していない。

次の候補は [run03_auto_t4.ipynb](colab_t4/experiments/run03_auto_t4.ipynb)。PRM4,500＋GSM1,500の合計6,000件、1 epoch。Stage 1はrun01を再利用し、Stage 2は新規LoRA。run02のStage 2は継ぎ足さない。学習率・LoRA・batch・最大長・全非padding loss・教材比率を維持する。固定評価800問は引き続き除外。

新しいColab T4で「すべて実行」を1回押し、run02と同じGoogleアカウントでDriveを認証する。ZIPは不要。完了は `RUN03_TRAINING_COMPLETE`、保存先は `geniac_workspace/adapters/t4_20261014_run03_prmfull6k_stage2/`。run03のGPU学習は未実行。

ローカルの機械的事前処理は完了：6,000件、1,598,551トークン、平均266.425、最大1,923、全件EOSあり。run02の528,362トークンの約3.03倍。学習時間の粗い見積もりは約71分＋環境構築・取得・mergeであり、実測ではない。学習後はrun02と同じ固定devで比較し、次の公式提出を判断する。0.22超えの保証はない。
