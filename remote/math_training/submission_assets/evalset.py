"""Load the question set the submission is scored on.

The questions come from a dataset the organisers publish, which holds the question text
and its identifiers and nothing else: no reference answers. Participants therefore never
download the answers, which is what the earlier arrangement could not avoid -- it read the
upstream benchmark, whose rows carry a ground_truth column.

The field whitelist below stays in place as a second line of defence, so that a source
which did carry answers would still not pass them to the caller, and the expected count is
checked so a changed source is noticed rather than silently used. Looking at the evaluation
data is prohibited by the rules.
"""
from __future__ import annotations

from typing import Any, Iterable

# The organisers' question set. Access is gated with automatic approval, so reading it needs
# a Hugging Face token: accept the terms on the dataset page once, then set the token.
SOURCE = "geniac-prize-physical-ai-competition-2026/compe_screening_eval"
SPLIT = "test"
EXPECTED_COUNT = 232

# Everything the inference notebook needs. Anything else -- the reference answer above
# all -- is discarded before the caller sees a row.
KEEP_FIELDS = ("question_id", "task", "subtask", "turns")


def strip_grading_fields(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the fields inference needs, dropping the reference answers."""
    kept = []
    for row in rows:
        missing = [f for f in KEEP_FIELDS if f not in row]
        if missing:
            raise KeyError("question is missing %s" % ", ".join(missing))
        kept.append({f: row[f] for f in KEEP_FIELDS})
    return kept


def select(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the fields inference needs. The published set is already the scored set."""
    return strip_grading_fields(rows)


def load_eval_questions(
    rows: Iterable[dict[str, Any]] | None = None,
    *,
    expected: int | None = EXPECTED_COUNT,
) -> list[dict[str, Any]]:
    """Return the questions to run inference on.

    ``rows`` is only for tests; left out, the published set is downloaded.
    ``expected`` guards against the source changing underneath us.
    """
    if rows is None:
        from datasets import load_dataset

        try:
            rows = load_dataset(SOURCE, split=SPLIT)
        except Exception as exc:
            raise RuntimeError(
                "評価用データを取得できませんでした。\n"
                "  1. https://huggingface.co/datasets/" + SOURCE + " を開き、\n"
                "     ログインした状態で利用条件に同意してください (承認は自動)\n"
                "  2. Hugging Face の read token を Colab Secrets に HFH_READ として設定し、\n"
                "     hf-login セルを実行してください\n"
                "  元のエラー: " + type(exc).__name__ + ": " + str(exc)[:200]
            ) from exc
    questions = select(rows)
    if expected is not None and len(questions) != expected:
        raise RuntimeError(
            "評価用データが %d 問になりません (%d 問)。"
            "配布物が最新か確認してください" % (expected, len(questions))
        )
    return questions
