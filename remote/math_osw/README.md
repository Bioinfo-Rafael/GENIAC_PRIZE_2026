# OSW型1時間パイロット

既存3k adapterから固定順序の最大128問を1巡する。各問題で現在のpolicyから2回答を生成し、正解数0/1/2に対して元の教師解説のlossに1/.5/0を掛ける。生成回答を教師ラベルにはしない。従来のtoken平均を維持し、PRM/LIMO教師全文＋EOS、rank8/alpha16、LR上限1e-6、元の438更新cosineを使う。

**CPUテスト済み。今回のCUDA/Unsloth/vLLM通し実行は未実施。0.30達成を保証する実験ではなく、小規模パイロット。** 短いprefixなので438更新のwarmup中で終わる点も意図した条件。学習できた件数で候補名を付け、未更新の親モデルを新モデルと称して出力しない。

## 起動

既存GPU・venv・固定キャッシュを使用する。追加インストールや外部教師は不要。GPUが使用中なら他ジョブを止めず終了する。14GiB以上の空き容量が必要。

```bash
cd /home/suzuki/geniac-remote
OSW_RUN="math_osw_$(date +%Y%m%d_%H%M%S)"
nohup bash remote/math_osw/launch.sh "$OSW_RUN" \
  > "/home/suzuki/geniac_math/${OSW_RUN}.console.log" 2>&1 < /dev/null &
echo "PID=$! RUN=$OSW_RUN"
tail -f "/home/suzuki/geniac_math/${OSW_RUN}.console.log"
```

コードをtar.gzで受け取った場合は、既存repoを上書きせず専用ディレクトリへ展開し、その中の`remote/math_osw/launch.sh`を指定する。

既定の前実験パスは`/home/suzuki/geniac_math/math_scale/math_scale_20261021_v1`。別の場合はlaunch.shの第2引数で指定。出力先は`~/geniac_math/math_osw/<run ID>/`。

## 時間・成果物

- 起動から絶対60分。再開時も元の期限を使い、時計を延長しない。
- 初期チェックと公式入力キャッシュ確認は5分以内。公式入力は分離したfetch/submit処理だけが読む。
- 学習は起動34分を締切とし、次のbatchを始める前に過去所要時間を確認。生成中の締切到達は途中回答を採点せず未完了batchを廃棄する。正常に終わったbatchの全stateを保存する。
- 保存・merge後、公式232回答を元の設定で単発生成。形式チェック、SHA256、提出JSONを含むZIPを**公開評価より先に**確保する。
- 公開400問は残り11分超の場合のみ実施。未評価・失敗でも既に完成した提出JSONは保持する。
- 最終ログに`SUBMISSION_READY`と`ZIP_READY`。`suite_report.json`に同じパス、hash、実時間、評価状態を記録する。
- 不具合・極端な低速・キャッシュ不足で提出JSONが完成しない場合は失敗を記録する。60分停止を、60分以内の成功保証とはしない。

出力の例（32問学習できた場合）：

```text
math_osw/<run ID>/
  suite_report.json
  OSW_C3_R8_L1e6_q3032.zip
  OSW_C3_R8_L1e6_q3032/submission/submission.json
  OSW_C3_R8_L1e6_q3032/submission/format_check.json
  chain/stage2/training/rft/train.jsonl
```

提出サイトへはユーザーがJSONを提出する。ZIP中の`OSW_C3_R8_L1e6_q*.json`が提出ファイルで、provenance配下は記録用。

## 検証

### 実行中のジョブを20分に短縮

Linux上の`cap_running.py --root <既存runディレクトリ>`を別プロセスで実行すると、既存の起動時刻から20分を上限にする。時計や学習コードのsnapshotは書き換えない。最初の非ゼロLR更新がcheckpointへ保存された時点、または起動7分で学習へSIGTERMを送り、完了済みbatchからmerge・提出推論へ進める。提出ZIPが用意できたらcoordinatorへ終了要求を送り、任意の公開評価を省略・中断する。20分の10秒前に対象runだけへ終了要求を送り、5秒後も残る対象phaseのprocess groupを終了させる。既に20分経過した場合は直ちに停止処理する。

`cap20_request.json`に短縮要求を記録する。従来の`budget.json`は60分のままなので、この追加制約も併読すること。CPU制御テスト済み、Linux/GPU実機は未検証。OSの停止不能なI/O待ちなどを除く停止上限であり、20分以内の提出JSON完成や精度改善は保証しない。有効な更新が保存されなければ、新モデルの出力を拒否する既存動作を維持する。

```bash
.venv-local/bin/python -m unittest discover -s remote/math_osw/tests -v
bash -n remote/math_osw/launch.sh
```

実際のtorch backward・checkpoint保存を使ったCPUテストで重み付け、生成を教師にしないこと、全重み0時のoptimizer/weight decay停止、途中batch破棄を確認する。ほかに固定順序・評価との重複・子process期限停止・receipt改変・ZIP原本一致を検査する。GPU・学習効果の検証とは区別する。
