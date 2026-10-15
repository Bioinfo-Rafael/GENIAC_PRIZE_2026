# RTX 6000 Ada：自動正誤ペアによるRFT / DPO比較

`remote/math_training/`だけで動くLinux用Python実装。共通Stage 1、run03 Stage 2、教材6,000件、固定dev/final分割は、ユーザー指定の共有Driveから必要な12ファイルだけ自動取得する。Gitに教材・重み・生成答案は置かない。ダウンロード内容は全ファイルSHA-256を検証し、run03重みは保存済み学習集計のハッシュとも照合済み。

7問やfix7は入力しない。既存教材を固定seed・3:1の比率で最大512問抽出し、各問4生成して正誤を採点する。正常終了して判定できた正解/誤答を1ペアずつ作り、同じrun03からRFTとDPOを別々に学習する。RFTはペアの正解のみ、DPOは正負両方を使う。DPOの固定参照はrun03そのもの。途中終了・解析不能を計算上の負例にしない。

## 初回の取得

Python 3.10〜3.12（推奨3.11）、Linux、NVIDIA GPU約48GB、環境構築後60GiB以上の空き容量が必要。ホストのPython・既存venvは変更せず、`~/geniac_math`内に学習用とvLLM評価用のvenvを作る。ドライバ570向けにtorch2.8.0のcu128を指定する。

リポジトリ全体の作業ファイルは不要。新しく実行専用checkoutを作る場合：

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/Bioinfo-Rafael/GENIAC_PRIZE_2026.git ~/geniac-remote
cd ~/geniac-remote
git sparse-checkout init --no-cone
git sparse-checkout set '/remote/math_training/'
git checkout main
```

既存cloneを使う場合は、そのclone内で次を一度実行する。未commitの編集がある場合は先に保存する。

```bash
git status --short
git sparse-checkout init --no-cone
git sparse-checkout set '/remote/math_training/'
git pull --ff-only origin main
```

`git pull`自体はコミット単位。sparse-checkoutは作業ツリーの配置を限定し、partial cloneの`--filter=blob:none`は不要なファイル本文の初回取得を抑える。既存の通常cloneへsparse-checkoutを設定しても、過去に取得したGitオブジェクトは消えない。

## 編集後のpullとバックグラウンド実行

ローカルでpushした後、リモートで実行する：

```bash
cd ~/geniac-remote
git pull --ff-only origin main
mkdir -p ~/geniac_math
GENIAC_RUN_ID="ada_$(TZ=Asia/Tokyo date +%Y%m%d_%H%M%S)"
nohup bash remote/math_training/launch.sh \
  --workspace "$HOME/geniac_math" \
  --run-id "$GENIAC_RUN_ID" \
  --budget-minutes 180 \
  > "$HOME/geniac_math/$GENIAC_RUN_ID.console.log" 2>&1 < /dev/null &
echo "PID=$! RUN=$GENIAC_RUN_ID"
```

`python3`が3.13以降なら、起動前に`export GENIAC_BOOTSTRAP_PYTHON=python3.11`とする。既存venvをactivateする必要はない。初回は依存パッケージ・Base重みの取得を含み、これも180分の予算に入る。次回はvenv・HFキャッシュ・入力を再利用する。依存導入だけで時間を使い切った場合、console logに残り、GPU学習は始まらない。同じコマンドを後で実行して準備を継続できる。

Driveへのアクセス制限・日次ダウンロード上限などが生じた場合はエラーで止まる。異なる重みへの代替や、ハッシュが違うファイルの採用は行わない。

## ログ・状態・結果

上のコマンドと同じシェルでは：

```bash
tail -f "$HOME/geniac_math/$GENIAC_RUN_ID.console.log"
```

別のシェルでは`GENIAC_RUN_ID`を表示されたrun名に設定する。

```bash
GENIAC_RUN_ID=ada_実際の日時
cat "$HOME/geniac_math/runs/$GENIAC_RUN_ID/status.json"
cat "$HOME/geniac_math/runs/$GENIAC_RUN_ID/report.json"
ls "$HOME/geniac_math/runs/$GENIAC_RUN_ID/logs"
nvidia-smi
```

console logには現在のphaseと残り時間を出す。詳細な生成進捗・lossは`runs/<run>/logs/`にある。例えば学習中は`tail -f ~/geniac_math/runs/<run>/logs/train_dpo.log`。

`status.json`は終了時の状態、`pid.json`は実行中のPIDを保存する。`report.json`は正答率、PRM/GSM別、判定不能、反復・打切り、生成長、基準モデルとの正誤変化・ペアbootstrap区間。公式提出スコアではない。

| status | 意味 |
| --- | --- |
| `quick_comparison_complete` | 3モデルの固定quick128比較まで完了。採用確定ではない |
| `dev_comparison_complete` | 3モデルの固定dev400比較まで完了。final400は未実施 |
| `insufficient_pairs` | 有効ペアが64未満で学習を開始しなかった。`pair_summary.json`を参照 |
| `insufficient_time_for_paired_training` | 両枝と評価の完了時間を確保できず、データ・参照計算・試走まで保存 |
| `budget_exhausted` / `interrupted` | 完了phaseと学習checkpointを保存。再開可能 |
| `failed` | `error.json`と該当phaseのログを確認 |

比較の初期値は`configs/ada_dpo.json`。microbatch2、実効batch16、lr5e-6、2 epochs、beta0.1。生成の試走から問題数を128/256/512のいずれかへ固定し、その後の再開では変更しない。2枝の学習試走（各2更新、重みは破棄）から所要時間を見積もる。コードは時間不足時にepochや設定を黙って変更せず、保存して停止する。

## 停止・再開

保存を伴う停止：

```bash
GENIAC_RUN_ID=ada_実際の日時
python3 -c 'import json,os,signal,sys; os.kill(json.load(open(sys.argv[1]))["pid"], signal.SIGTERM)' \
  "$HOME/geniac_math/runs/$GENIAC_RUN_ID/pid.json"
```

PIDが現在の対象プロセスか確認してから使う。`kill -9`は避ける。学習は更新境界でcheckpointを保存する。通常時も5更新ごとに保存する。

**同じコード・同じ環境の再開**は、元run名に`--resume`を付ける。手動での再開1回ごとに新しい最大180分枠になる。

```bash
cd ~/geniac-remote
GENIAC_RUN_ID=ada_実際の日時
nohup bash remote/math_training/launch.sh \
  --workspace "$HOME/geniac_math" --run-id "$GENIAC_RUN_ID" \
  --budget-minutes 180 --resume \
  >> "$HOME/geniac_math/$GENIAC_RUN_ID.console.log" 2>&1 < /dev/null &
```

すでにpullでコードが変わった場合、同じrunへの混在再開は拒否する。新しい実験は新しいrun名で始める。以前のコードで継続したい場合は、保存されたコードsnapshotを明示して実行できる：

```bash
GENIAC_RUN_ID=ada_実際の日時
nohup "$HOME/geniac_math/.venv-train/bin/python" -u \
  "$HOME/geniac_math/runs/$GENIAC_RUN_ID/code/run.py" \
  --workspace "$HOME/geniac_math" --run-id "$GENIAC_RUN_ID" --resume \
  --eval-python "$HOME/geniac_math/.venv-eval/bin/python" --budget-minutes 180 \
  >> "$HOME/geniac_math/$GENIAC_RUN_ID.console.log" 2>&1 < /dev/null &
```

再開はoptimizer/scheduler/GradScaler/RNGを含む。コード・設定・環境パッケージ・入力・ペア・参照計算の一致を検査する。学習済みbranchは再学習しない。GPUワークスペースの排他ロックがあるため、同じworkspaceへ二重実行しない。

## 出力の場所と確認範囲

- `inputs/`：Driveからの検証済み入力。
- `runs/<run>/training/{rft,dpo}/final/`：継続学習後のStage 2 LoRA＋tokenizer。Base＋共通Stage 1に適用する。run03 Stage 2をさらに重ねて適用しない。
- `runs/<run>/models/{b0,rft,dpo}/`：比較用の統合モデル。
- `generations/`：回答本文・生トークンID・finish/stop reason・採点。本文は編集しない。
- `context.json`、`packages_*.json`、`code/`：設定・入力hash・環境・実行コード。

初回実装の範囲はRFT/DPOのquick/dev比較まで。ペア不足時は自動で別方式を開始せず、診断を保存する。DFT/GRPO/Self-Explore、未使用final400での採用判定、公式推論・提出は別の後続作業。7問をその場で評価して設定を変える処理も含めない。

ローカルでは実入力のSHA・非重複・tokenizer、lossの値と勾配、padding/勾配累積、小さなQwen3＋LoRAの実際のbackward、保存/再開をテストする。**RTX 6000 Ada上のCUDA・bitsandbytes・vLLMの通し実行は未検証**。依存は直接依存を固定しており、実際の全導入版をrunごとに記録する。初回のGPU実行でメモリ・速度・互換性を確認する。

開発用テスト：

```bash
python -m pytest -q remote/math_training/tests
python -m compileall -q remote/math_training
bash -n remote/math_training/launch.sh
```
