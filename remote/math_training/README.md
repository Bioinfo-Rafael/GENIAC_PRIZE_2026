# RTX 6000 Ada：自動正誤ペアによるRFT / DPO比較

`remote/math_training/`だけで動くLinux用Python実装。共通Stage 1、run03 Stage 2、教材6,000件、固定dev/final分割は、ユーザー指定の共有Driveから必要な12ファイルだけ自動取得する。Gitに教材・重み・生成答案は置かない。ダウンロード内容は全ファイルSHA-256を検証し、run03重みは保存済み学習集計のハッシュとも照合済み。

7問やfix7は入力しない。既存教材を固定seed・3:1の比率で最大512問抽出し、各問4生成して正誤を採点する。正常終了して判定できた正解/誤答を1ペアずつ作り、同じrun03からRFTとDPOを別々に学習する。RFTはペアの正解のみ、DPOは正負両方を使う。DPOの固定参照はrun03そのもの。途中終了・解析不能を計算上の負例にしない。

## 8時間の比較実験を実行する

`--suite`で後続の比較実験を実行できる。元の180分pilotは`--suite`なしで引き続き実行できる。suiteは2つの対照＋5方式、別seedによる再確認、時間に収まる場合の6000問への拡張、未使用final400での最終確認を1ジョブで順次処理する。新方式のGPU通し実行はまだ未確認。ローカルでは損失・キャッシュ検証・模擬GPUによる全体制御をテストしている。

今回完了したrunを利用するコマンド：

```bash
cd ~/geniac-remote
git pull --ff-only origin main
mkdir -p "$HOME/geniac_math"
GENIAC_SUITE_ID="suite8h_$(TZ=Asia/Tokyo date +%Y%m%d_%H%M%S)"
nohup bash remote/math_training/launch.sh --suite \
  --workspace "$HOME/geniac_math" \
  --run-id "$GENIAC_SUITE_ID" \
  --source-run "$HOME/geniac_math/runs/ada_20261016_221122" \
  --budget-minutes 480 \
  > "$HOME/geniac_math/$GENIAC_SUITE_ID.console.log" 2>&1 < /dev/null &
echo "PID=$! SUITE=$GENIAC_SUITE_ID"
```

前回runの重み・生成答案・ペア・環境記録・code snapshotを使うので、完了まで元runを移動/削除しない。前回runは変更しない。旧pilot（a3b82b4）の生成コード、モデル設定、split、入力manifest、採点コード、tokenizer、導入パッケージの一致を検証する。不一致は理由を表示して停止し、未検証キャッシュを黙って使わない。`--source-run`を省略した新しいsuiteではb0の生成から実行する。

```bash
tail -f "$HOME/geniac_math/$GENIAC_SUITE_ID.console.log"
# 終了後：
cat "$HOME/geniac_math/suites/$GENIAC_SUITE_ID/suite_report.json"
```

`tail -f`はCtrl+Cで閉じても本体は継続する。別のシェルでは`GENIAC_SUITE_ID`に表示されたsuite名を設定する。

比較条件：

| ID | 方式 | 生成問題数 | 学習データ | LR / epoch |
| --- | --- | ---: | --- | --- |
| c1_rft_512 | RFT | 512（既存） | 元の混在183問の正解 | 5e-6 / 1 |
| c2_rft_low_lr_512 | RFT | 512（既存） | 同じ183件 | 1e-6 / 1 |
| e1_rft_mixed | RFT | 2048 | 正誤混在問題の正解 | 5e-6 / 1 |
| e2_rft_positive | RFT | 2048 | 正解のみ問題も含む、1正解/問 | 5e-6 / 1 |
| e3_dft_positive | DFT | 2048 | e2と同じ | 5e-6 / 1 |
| e4_dpo | DPO | 2048 | 正誤混在問題の1ペア/問 | 5e-6 / 1 |
| e5_dpo_nll | DPO＋正解NLL（係数0.1） | 2048 | e4と同じ | 5e-6 / 1 |

正解例は固定seedで選び、人が問題や答案を選ばない。反復フィルタ・最短答案優先も追加していない。DPOのbetaは0.1。各方式は同じrun03から独立開始し、生成poolは共通。学習中のモデルでは生成しない。試走は最長側の例でmicrobatch 2/4/8を比較し、effective batch16は固定。OOMになったサイズは記録して除外する。半分/終了時のアダプターも保存するが、今回の自動採用候補は1 epoch終了時に固定する。

最初に終了ラベル・EOS設定・保存再読込時のlogit一致を点検する。無更新の再統合モデルとb0の全テンソル・dtype・推論設定・tokenizerの一致をCPUで検証し、不一致なら停止する（`audit/integrity.json`）。dev先頭32問の生成差は実行時の揺れとして記録し、重みと設定が同一なら続行する。実際に旧suiteでは398テンソルと設定が同一でも4/32問の生成が変わった。epoch等の学習専用設定差は推論条件の差とは扱わない。テンプレート末尾の余分なtokenはauditに記録し、この実験では変更しない。生成差がある場合は`inference_variability`と`uncertainty_scope`をreportに保存する。paired bootstrapは問題の再標本化であり、推論実行間の揺れは含まないため、小さな改善を確証と扱わない。

時間配分は点検30分・対照45分・2048問生成75分・主比較120分・追加確認90分・final60分・予備60分。これは上限枠で、待機して8時間を埋める処理はない。生成・評価・初期準備も480分に含める。最終評価用の60分を残して追加実験を停止する。6000問への拡張は実測スループットで時間内と見込めた場合のみ。計画にあった「改善モデルによる再生成」は今回は選ばず、データ拡張を実装した。

devで+2ポイント、反復/非stopの悪化が各+3ポイント以内、GSMの悪化が4問以内の候補から最良を選び、同じデータ・shuffle seed43で再学習して再確認する。両seedで基準を満たした場合に候補を固定する。条件を満たす候補がなければb0を維持する。これは探索ルールであり有意差の保証ではない。

候補のdev選択を`selection.json`へ固定してからfinal400を評価する。finalを見て別の候補へ選び直す処理はない。finalの差の95%区間が0をまたぐ、または品質基準を満たさない場合、`recommendation`はb0維持になる。7問/fix7の監査と公式提出はこのジョブに含まない。

主な成果物は`~/geniac_math/suites/<suite名>/`：

- `suite_report.json`：候補別dev成績・正誤変化・区間、再現確認、final比較、推奨アダプター。
- `status.json`：`suite_complete`、`suite_complete_no_improvement`、`time_limited`、`interrupted`、`failed`。
- `suite_report.json`の`incomplete_trials`：時間切れや失敗の候補。suite完了は全候補の成功を意味しない。
- `logs/`：全phaseの詳細ログ。consoleにもphaseと候補ごとの結果を表示。
- `audit/`：終端ラベル・保存再読込・無更新生成の比較。
- `trials/<候補>/training/{rft,dpo}/final/`：候補アダプター。DFTはrft、DPO+NLLはdpoの下。
- `trials/<候補>/training/{rft,dpo}/train.jsonl`：loss、gradient norm、log probabilityやpreference margin。
- `trials/<候補>/training/{rft,dpo}/overflow.jsonl`：FP16勾配オーバーフローの再試行記録。重み・scheduler・更新回数を進めずloss scaleを半減し、同じバッチを再計算する。8回連続しても回復しない場合はcheckpointを保存して候補を停止する。
- `pools/`：生成文と自動選別結果。答案本文は編集しない。

候補の比較用統合モデルはdev評価後に削除してディスクを節約する。アダプター・checkpoint・生成結果を保持し、最終候補だけfinal前に再統合する。b0・Stage 1の元runのモデルは削除しない。`recommended_adapter`は共通Stage 1に適用するStage 2アダプターであり、run03と重ねて適用しない。

中断からの再開は同じコード・環境で上の起動コマンドに`--resume`を追加し、元のsuite名と同じ`--source-run`を指定する。手動再開は新しい最大480分枠となる。コード変更後の混在再開は拒否するので、保存snapshotの`code/launch.sh --suite`を使うか、新しいsuite名にする。再開時にもfinal前の選択は固定されたまま。

### FP16エラーで失敗した候補だけを新しいsuiteで再実行

`suite8h_20261017_143739`ではc1/c2/e1/e3が完了し、e2/e4/e5がFP16非有限勾配で停止した。`--source-suite`はこの修正前（bde69f9）のsuiteを検証し、生成済み2048問と完了済み候補のアダプター・dev結果を再利用する。失敗した候補は修正済みのloss scalingでrun03から学習し直す。旧optimizerやコード署名を書き換えて接続する処理ではない。元suiteと元pilotを残して実行する。

```bash
cd ~/geniac-remote
git pull --ff-only origin main
GENIAC_SUITE_ID="retry8h_$(TZ=Asia/Tokyo date +%Y%m%d_%H%M%S)"
nohup bash remote/math_training/launch.sh --suite \
  --workspace "$HOME/geniac_math" --run-id "$GENIAC_SUITE_ID" \
  --source-run "$HOME/geniac_math/runs/ada_20261016_221122" \
  --source-suite "$HOME/geniac_math/suites/suite8h_20261017_143739" \
  --budget-minutes 480 \
  > "$HOME/geniac_math/$GENIAC_SUITE_ID.console.log" 2>&1 < /dev/null &
echo "PID=$! SUITE=$GENIAC_SUITE_ID"
tail -f "$HOME/geniac_math/$GENIAC_SUITE_ID.console.log"
```

完了済み条件はreportの`imported_from`、引継ぎ元とハッシュは`recovery_manifest.json`に記録する。修正版が改善する保証はなく、再試行上限でも非有限勾配が続く候補は失敗として残す。旧suiteのfinalではb0だけが評価済みで、候補のfinalを用いた再選択はしていない。

## 初回の取得

Python 3.10〜3.12（推奨3.11）、Linux、NVIDIA GPU約48GB、環境構築後60GiB以上の空き容量が必要。起動スクリプトは対応するPythonを自動検出する。ホストにない場合は[uvの公式インストーラー](https://docs.astral.sh/uv/getting-started/installation/)を使い、作業領域内にPython 3.11を準備する。ホストのPython・既存venvは変更せず、`~/geniac_math`内に学習用とvLLM評価用のvenvを作る。ドライバ570向けにtorch2.8.0のcu128を指定する。

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

`python3`が3.13以降でも、`python3.11`などがあれば自動で選ぶ。なければuvがPython 3.11を取得する。明示指定したい場合だけ`export GENIAC_BOOTSTRAP_PYTHON=/path/to/python3.11`を使う。既存venvをactivateする必要はない。初回はPython・依存パッケージ・Base重みの取得を含み、これも180分の予算に入る。次回はvenv・HFキャッシュ・入力を再利用する。依存導入だけで時間を使い切った場合、console logに残り、GPU学習は始まらない。同じコマンドを後で実行して準備を継続できる。

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

元の`run.py`の範囲はRFT/DPOのquick/dev比較まで。ペア不足時は自動で別方式を開始せず、診断を保存する。後続の`suite.py`はDFTとfinal400評価にも対応する。GRPO/Self-Explore、公式推論・提出は別の後続作業。7問をその場で評価して設定を変える処理も含めない。

ローカルでは実入力のSHA・非重複・tokenizer、lossの値と勾配、padding/勾配累積、小さなQwen3＋LoRAの実際のbackward、保存/再開をテストする。ユーザー提供の`ada_20261016_221122`ログではRTX 6000 Ada環境でdev400比較まで完了した。依存は直接依存を固定しており、実際の全導入版をrunごとに記録する。

開発用テスト：

```bash
python -m pytest -q remote/math_training/tests
python -m compileall -q remote/math_training
bash -n remote/math_training/launch.sh
```

## 計画の記録：合計8時間の比較実験（2026-10-17）

予算はRTX 6000 Ada 48GB 1枚での生成・学習・評価を合計した480分。実装時間は含めない。以下は元の計画を記録したもの。実装済みの範囲と実行手順は冒頭の「8時間の比較実験を実行する」を参照。元のrun.pyは180分pilotとして維持し、480分の実験はsuite.pyで実行する。

### 根拠となる今回の成績

ユーザーから受領した`ada_20261016_221122`のreportと学習ログ。公式スコアではない。リモート成果物本体はローカルへ取得していない。

| モデル | dev400正解数 | 正答率 | 反復検出数 | 正常stop以外 | 平均出力token |
| --- | ---: | ---: | ---: | ---: | ---: |
| b0 (run03) | 253 | 63.25% | 121 | 123 | 753.90 |
| RFT | 255 | 63.75% | 196 | 202 | 1128.46 |
| DPO | 253 | 63.25% | 123 | 127 | 772.43 |

RFTは14問改善・12問悪化、差のpaired bootstrap 95%区間は[-2.0,+3.0]ポイント。DPOは3問改善・3問悪化、[-1.25,+1.25]ポイント。正答率の明確な改善は確認できていない。

512問×4生成から有効正解1193件・誤答741件。正解のみ210問、正誤混在183問、誤答のみ118問、使用不能1問。両枝とも混在183問から1組ずつ取り、2 epoch、24更新。RFTには正解のみ210問が未使用。chosenの反復は3/183、中央値61単語、90%点132単語で、反復答案の大量混入という仮説は支持されない。単語数をreportのtoken数と直接比較しない。DPOの終盤lossは約0.60だが、これだけで未学習・収束・飽和とは判断できない。

### 0. 比較前の点検（上限30分）

- tokenizerのEOS、im_end、generation_configのstop IDs、教師答案末尾とcompletion maskを実トークンで確認する。テンプレートが終端と改行を付けた後にEOSを追加する経路があり、生成時の終端と整合するか確認する。現段階でバグとは断定しない。
- 同一32問でb0の学習前後の量子化/統合経路、最終回答より後の反復、EOS確率を確認。学習前の再保存・統合だけでも結果が変わる場合は原因修正を優先する。
- DPOはchosen/rejectedそれぞれの平均token log probability、参照からの差、preference margin・正解率、gradient norm、実更新回数を記録。loss低下だけで判断しない。
- 採点不能は負例にしない。採点タイムアウトは記録して継続。終了・採点・token化の不具合があれば、共通修正後にb0から比較し直し、旧結果との単純比較は避ける。

### 1. 学習量の対照（上限45分）

保存済み183ペアと生成結果をハッシュ付きで再利用し、run03から独立に実行する。元runのsnapshotを書き換えない。

| ID | 学習例 | 方法 | peak LR | epoch | 問い |
| --- | --- | --- | --- | --- | --- |
| C1 | 元のchosen183件 | RFT | 5e-6 | 1 | 2 epochより前なら反復増加を避けられるか |
| C2 | 同じ183件 | RFT | 1e-6 | 1 | 更新幅を小さくすると正解を維持できるか |

0.5/1 epochのcheckpointを保存。quick128は異常検出のみ、採用判定はdev400で行う。C1と元2 epochはcosine scheduleの長さも異なるため、epoch数だけの厳密な因果比較とは呼ばない。

### 2. 共通生成データの拡張（上限75分）

元の6000教材から固定seed・PRM:GSM=3:1で2048問を選び、run03から各4回答生成する。7問/fix7、dev/finalを入力・選別に使わない。既存512問を含む固定順序にし、モデル・問題・生成設定・tokenizer・採点コードの一致を検証して再利用する。生成文の手修正はしない。

二つのデータを同じ生成結果から作る。Mは正誤混在問題の1ペア/問。Pは少なくとも一つ有効な正解のある全問題から、固定seedで1正解/問（正解のみ問題も含む）。MのchosenとPの対応答案は同じにする。183/512と393/512の比率が続けば、M約730組、P約1570件が見込まれるが、実数を記録する。答案の最短選択や反復フィルタは主実験に追加しない。

### 3. 五つの方式を比較（上限120分）

すべてrun03から独立に開始。初期比較はLR=5e-6、1 epoch、effective batch=16、同じ生成・採点条件。DFT等でlossの数値スケールは異なるため、lossの大小で方式を順位づけしない。

| ID | 方法 | データ | 確かめること |
| --- | --- | --- | --- |
| E1 | RFT | Mのchosen | C1に対し問題数を増やす効果 |
| E2 | RFT | P | 正解のみ問題も使う効果（標本数・分布・更新数が変わる実用比較） |
| E3 | DFT | E2と同じP | token確率で重み付けする目的関数の効果 |
| E4 | DPO | M | 元183組より大きいペア集合での効果、beta=0.1 |
| E5 | DPO + 正解NLL | E4と同じM | 正解を誤答より優先するだけでなく、正解自体の確率を維持・増加させる効果 |

DFTはtokenごとの`-stop_gradient(p)*log(p)`をcompletionに適用。E5はDPOのsequence-sum定義を維持し、chosenのtoken平均NLLを係数0.1で加える探索設定。論文そのままの再現とは称さない。係数・LRの変更は追加実験として記録する。

### 4. 有望な方式の追加確認（上限90分）

dev400で改善し、反復・stop・GSMの大きな悪化がない上位方式を最大2つに絞る。まず最良方式を別の学習shuffle seedで再学習して再現性を見る（生成データは共通）。残りは、実測時間に応じて次のどちらか一つへ配分する。

- データ拡張：2048→最大6000問へ固定順序を延長し、同じ方式をrun03から再学習。改善がデータ量に応じて続くか確認。
- 生成更新：最良モデルで同じ2048問を再生成し、新しい正解/誤答から短く追加学習。固定データを繰り返す場合との差を調べる。

全方式が悪化なら拡張を行わず、反復・停止・忘却の診断へ戻す。初回8時間枠に新規GRPO基盤の構築と多数の係数探索は入れない。

### 5. 最終確認（上限60分）と予備（60分）

30+45+75+120+90+60+60=480分。前回の約30数分の通し実行を参考にした予算配分で、所要時間の保証ではない。実測で見積もりを更新し、最終評価60分を確保できない追加実験は始めない。データ拡張を優先度順に省き、比較条件を途中で黙って変えない。

- dev400での実用的な目標はb0より+2ポイント（8問）以上、反復率・non-stop率の悪化は各+3ポイント以内。GSMで5問以上の悪化は自動採用せず精査する。これらは探索上の基準で、統計的有意性を意味しない。
- 最終候補をdevだけで1つ固定して、未使用final400でb0との比較を一度行う。finalを見て係数・checkpoint・候補を再選択しない。
- paired bootstrap区間、正解→誤答/誤答→正解、PRM/GSM別、採点不能・反復・終了率を報告。多候補探索後のdev差を確証扱いしない。finalで区間が0をまたぐ場合も「改善確定」としない。
- 最終答案はモデルの単一生成で評価。7問は設定選択から隔離し、最終候補を固定した後の参考監査だけに使う。dev/finalと公式提出スコアは区別する。

### 実装時の変更点

共通生成キャッシュの検証付き共有、M/Pの分離、DFTとDPO+NLL、epoch別checkpoint、追加診断、final split評価、480分を管理する実験スケジューラーが必要。既存runの再開署名を回避せず、各branchの親モデル・データhash・設定を記録する。GPUは1ジョブずつ使用。microbatchは2→4→8を実測し、effective batch16を維持して速い設定を採る。精度/量子化設定の変更は別の要因になるため、主比較と混ぜない。

### 文献と適用範囲

- [数学推論のRFTと推論経路の多様性](https://arxiv.org/abs/2308.01825)：正解生成の再学習を支持。今回のモデルでの改善保証ではない。
- [Dynamic Fine-Tuning](https://arxiv.org/abs/2508.05629)：token確率による重み付けを提案。今回のQwen3 run03へ適用するのは新しい実験。
- [Iterative Reasoning Preference Optimization](https://arxiv.org/abs/2404.19733)：数学推論でDPOに正解NLLを加える方法と反復生成を検討。E5と生成更新の動機。
- [ReST-EM](https://arxiv.org/abs/2312.06585)：生成・正解検証・再学習を反復する根拠。
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)：次段階のオンラインRL候補。今回は既存基盤で比較できる方式を先に検証する。
