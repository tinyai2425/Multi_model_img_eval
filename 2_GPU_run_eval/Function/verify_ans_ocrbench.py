import re

FINAL_ANSWER_PATTERN = re.compile(
    r"final\s*answer\s*[:：]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)

PREAMBLE_PATTERN = re.compile(
    r"^\s*(?:the\s+answer\s+is|answer|答案|结果)\s*[:：]?\s*",
    re.IGNORECASE,
)


def _strip_wrappers(s):
    s = s.strip()
    for a, b in (("$$", "$$"), ("$", "$")):
        while (
            len(s) >= len(a) + len(b)
            and s.startswith(a)
            and s.endswith(b)
        ):
            s = s[len(a):-len(b)].strip()
    pairs = [('"', '"'), ("'", "'"), ("`", "`"), ("*", "*"), ("“", "”"), ("‘", "’")]
    changed = True
    while changed and len(s) >= 2:
        changed = False
        for a, b in pairs:
            if s.startswith(a) and s.endswith(b):
                s = s[1:-1].strip()
                changed = True
                break
    return s


def _normalize_hme(s):
    if s is None:
        return ""
    s = _strip_wrappers(str(s).lower())
    s = re.sub(r"\s+", "", s)
    return s


def extract_answer(text):
    if not text:
        return ""

    matches = list(FINAL_ANSWER_PATTERN.finditer(text))
    if matches:
        ans = matches[-1].group(1).strip()
    else:
        non_empty = [s.strip() for s in text.split("\n") if s.strip()]
        ans = non_empty[-1] if non_empty else ""

    ans = PREAMBLE_PATTERN.sub("", ans).strip()
    ans = _strip_wrappers(ans)
    return ans


def _normalize_for_match(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def expect_to_list(expect):
    if expect is None:
        return []
    if isinstance(expect, (list, tuple)):
        return [str(x) for x in expect if x is not None]
    text = str(expect).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            import json
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (TypeError, ValueError):
            pass
    if " | " in text:
        return [p for p in text.split(" | ") if p]
    return [text] if text else []


def verify_answer(gt_list, response, question_type=None):
    gt_list = expect_to_list(gt_list) if not isinstance(gt_list, (list, tuple)) else [
        str(x) for x in gt_list if x is not None
    ]
    if not gt_list or response is None:
        return False

    is_hme = (
        question_type
        == "Handwritten Mathematical Expression Recognition"
    )

    pred = extract_answer(response)
    if is_hme:
        pred_norm = _normalize_hme(pred)
        resp_norm = _normalize_hme(response)
    else:
        pred_norm = _normalize_for_match(pred)
        resp_norm = _normalize_for_match(response)

    for gt in gt_list:
        gt_norm = _normalize_hme(gt) if is_hme else _normalize_for_match(gt)
        if not gt_norm:
            continue
        if gt_norm in resp_norm or gt_norm in pred_norm:
            return True
    return False
