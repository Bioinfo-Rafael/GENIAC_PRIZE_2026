# 保存済み公開devの抽出診断と独立公開評価セット

新規学習前の最初の2項目だけをCPUで実行する。①A・D・R12の保存済み公開dev400回答の抽出方法比較、②数学SFTと既存dev/finalから分離した公開評価400問の固定。新規学習、モデルロード、生成、教師NLLのforward、loss比較、提出データへのアクセスは行わない。

既存実験ディレクトリは読み取り専用として扱い、結果は別の `diagnostics/…_public_eval_v1/` に保存する。処理中に出力するのは件数・メタデータだけ。既存の `math_training/grading.py` の正解ラベル処理・数式同値判定と `core.comparison` のpaired bootstrapを再利用する。

## 実行・再開

以前の古いGitでも動くよう、sparse checkoutは `add` ではなく既存リストを保持した `set --stdin` を使用する。リモートの既存 `.venv-rank-ablation` を利用し、パッケージの再インストールはしない。`math-verify` は保存済み評価と同じバージョン、`pyarrow` が必要。

```bash
(
  set -e
  cd "$HOME/geniac-remote"
  git pull --ff-only origin main
  if git sparse-checkout list >/dev/null 2>&1; then
    { git sparse-checkout list; printf '%s\n' remote/math_training remote/public_eval_diagnostics; } |
      sort -u | git sparse-checkout set --stdin
  fi

  GENIAC_R12="$HOME/geniac_math/rank64_12k/rank64_12k_20261021_183439"
  test -f "$GENIAC_R12/evaluation/R12/generations/b0_eval.json"
  test -f remote/public_eval_diagnostics/launch.sh
  GENIAC_LOG="$HOME/geniac_math/$(basename "$GENIAC_R12")_public_eval.console.log"
  nohup bash remote/public_eval_diagnostics/launch.sh "$GENIAC_R12" \
    >> "$GENIAC_LOG" 2>&1 < /dev/null &
  echo "PID=$! LOG=$GENIAC_LOG"
)
```

同じコマンドで再開できる。公開データはハッシュ確認付きキャッシュから再利用し、抽出診断は25問ごとの保存地点から再開する。コード・入力・採点関連パッケージを固定し、途中で変わった場合は停止する。同じ処理の二重起動はロックで拒否。出力にコードのスナップショットを保存するので、再開時もそのコードを使う。

別の公開評価セットも過去に利用していた場合は、初回起動時に明示的に追加できる（`problem` を含む公開問題行のJSON/JSONL。コンペの推論問題ファイルは指定しない）。今回の既定除外には公開train全体が含まれるので、過去の公開train由来の数学SFT・評価問題はその範囲で除外済み。

```bash
bash remote/public_eval_diagnostics/launch.sh "$GENIAC_R12" \
  --exclude-public-rows /absolute/path/to/other_public_eval.json
```

除外条件を変更したい場合は既存の固定評価セットを上書きせず、`run.py --r12 … --output NEW_DIRECTORY --cache … --exclude-public-rows …` で別の保存先に作成する。

## 結果の確認

```bash
GENIAC_DIAG="$HOME/geniac_math/diagnostics/rank64_12k_20261021_183439_public_eval_v1"
tail -f "$HOME/geniac_math/rank64_12k_20261021_183439_public_eval.console.log"
# 完了後（tailはCtrl-Cで抜ける）
cat "$GENIAC_DIAG/README.md"
cat "$GENIAC_DIAG/complete.json"
```

| ファイル | 内容 |
|---|---|
| `README.md` | 抽出方法別正答率、解釈上の注意、独立評価セット概要 |
| `extraction_report.json` | 全体・dataset・分野・難易度・出力長別の全指標、固定した各抽出規則でのA/D/R12比較とpaired bootstrap 95% CI |
| `extraction_summary.csv` | 上記の層別集計。すべて層全体を分母とする |
| `extraction_questions.csv` | 公開devの問題ID、選択した式、正誤、抽出失敗、長さ、終了理由。問題本文は複写しない |
| `extraction_{A,D,R12}_rows.json` | 再開用の問題別診断 |
| `independent_dev400.json` | 新しい公開評価問題・正解・分野・公式難易度。既存workerが扱えるid/dataset/problem/answerを含む |
| `independent_report.json` | ソースハッシュ、選択seed、層別配分、重複除外理由、固定セットのハッシュ、正解ラベル解析不能件数 |
| `identity.json` / `code/` | 入力とコード・採点パッケージの固定記録 |
| `complete.json` | 正常完了とCPU処理時間。生成・学習を行っていないことの記録 |

ダウンロード用に診断結果だけをまとめる場合：

```bash
tar -czf "$HOME/geniac_math/public_eval_diagnostics.tar.gz" \
  -C "$GENIAC_DIAG" README.md complete.json identity.json \
  extraction_report.json extraction_summary.csv extraction_questions.csv \
  independent_report.json independent_dev400.json
```

## 1. 抽出規則と解釈

- **Current**: 既存graderをそのまま使用。boxedがあれば最後のboxedを優先し、なければ既存のanswer marker・数値単独の規則。
- **First box / Last box**: 最初 / 最後の `\boxed{…}` または `\fbox{…}` の開始位置を選び、対応する閉じ括弧までを抽出。選んだboxedが壊れていたら失敗とし、別のboxedへ戻らない。
- **Explicit final**: 最後の明示的な `final answer is/:/=` または行頭 `####` の同一行の内容。そこにboxedがあればその行の最後のboxed。一般的な本文中の「answer」や任意の数字にはfallbackしない。改行をまたぐ答え・単位付き文章などは解析不能となり得る。

式の選択には正解を使わない。選択後の数式はすべて同じ既存graderで判定する。**正解に応じてfirst/lastのどちらかを選ぶoracle性能は計算せず、提出用JSONや推論コードも変更しない。** First/Last比較は公開devの回答挙動の診断。

全体の分母は各モデル400問。未抽出・解析不能も不正解として残す。`missing_selection` は規則に該当する式がない数、`extraction_failure` は選択した式が解析不能だった場合も含む。後者は本文に答えが存在することや、計算が正しいことの証明ではない。正解ラベルの解析不能・grader例外も別に記録する。

`first_correct_last_wrong` は最後のboxedが欠損・壊れている場合も含む。そのうち両方が解析できた例は `first_correct_last_wrong_both_extractable` に分ける。逆方向の変化も記録する。問題別CSVで該当IDを確認できる。

難易度はPRM元データの公式level 1–5を問題IDで結合する。元devの正解ラベルは変更しない。GSMに公式難易度はないため `not_provided` とし、長さで難易度を捏造しない。生成長は256以下、257–512、513–1024、1025–2048、2048超の排他的区間。長さ別の問題構成はモデルで変わるため因果比較ではない。

保存済みの正誤とCurrent再採点の食い違いも報告する。A・D・R12のdevハッシュ、推論条件、graderコード・math-verify版が一致しなければ停止し、再推論はしない。bootstrapは問題の再標本化であり、実行の非決定性や多重比較を補正しない。

## 2. 独立公開評価セットの作り方

ソースは既存の数学学習と同じ公開データ系列のみ。PRMのOpenAI math_splits/test 500問、GSM8K main/test 1,319問を候補とする。公開ソースは以下のリビジョンとSHA256をコードで固定する。

- [OpenAI PRM800K](https://github.com/openai/prm800k/tree/7ecc794703b2877f63226f2477a49b34f9b25163): `7ecc794703b2877f63226f2477a49b34f9b25163` の `prm800k/math_splits/{train,test}.jsonl`。
- [OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866/main): `740312add88f781978c0658806c59bc2815b9866` のmain train/test parquet。

PRM train 12,000行＋GSM train 7,473行全体、既存の公開dev/final、明示された追加公開評価問題を除外元とする。D/R12の実際の数学学習IDがこのtrain除外集合に含まれることも検査する。

重複判定は①既存と同じNFKC・空白正規化ID、②大文字小文字・空白だけを無視した一致、③20 token以上の文で単語/記号5-gram Jaccardが0.85以上。数字や数式演算子は保持する。候補内の重複も同じ規則で取り除く。近似一致は保守的な除外であり、意味的重複を完全に検出するものではない。

残ったPRMからsubject×levelの構成比に比例して300問を最大剰余法で配分し、各層を固定ハッシュ順で抽出。GSMから100問を固定ハッシュ順で抽出。seedは20261021。問題や回答の良し悪し・モデルの正誤を選択に使わず、gold解析不能でも選択後に除外しない。400問を確保できなければ停止する。

「独立」とは**既知の数学SFT・既存公開dev/finalからの分離**。Stage 1の全教材やベースモデルの事前学習との重複は未確認。同じ数学データ系列内の評価であり、本番の分布を代表すると保証するものではない。新セットの推論は別段階で行い、それまでは学習やモデル選択に利用しない。将来の推論時にはモデルのtokenizer/contextに対する入力長の検査も必要。

## ローカル検証（2026-10-21）

- 8件のCPUテストが成功。抽出失敗、最初の正答→最後の誤答、固定分母、難易度層化、完全/近似重複、既存nested split、評価条件不一致の拒否を確認。
- 固定公開ソースの実データで、PRMの近似重複5問を除外し495問、GSMは1,319問が候補として残った。300+100問を確保し、既存graderで全400問のgoldを解析できた。
- この公開セットの行ハッシュ（`core.digest`）: `7eafe2c0c46678af1483ea724948ccd394914771eb8362d4990ef4e91cf87d70`。追加除外リストを指定した場合は変わり得る。
- 合成回答400問×3モデルによる全パイプラインと再開の結合テストが成功。これはA/D/R12の実スコアの測定ではない。実回答の集計はリモートで実施する。

```bash
python -m unittest discover -s remote/public_eval_diagnostics/tests -v
bash -n remote/public_eval_diagnostics/launch.sh
```
