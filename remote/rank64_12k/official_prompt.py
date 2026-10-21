"""Unmodified message builder from distributed notebook2_inference [inference]."""
def _build_messages(question_text):
    """推論用 config の turns 定義に問題文を差し込んで messages を組む。"""
    msgs = []
    if _prompt_spec["system_prompt"]:
        msgs.append({"role": "system", "content": _prompt_spec["system_prompt"]})
    for turn in _prompt_spec["turns"]:
        msgs.append({
            "role": turn["role"],
            "content": turn["content"].replace(_placeholder, question_text),
        })
    return msgs
