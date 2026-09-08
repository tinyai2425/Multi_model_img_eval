import re

VALID_CHOICES = ("A", "B", "C", "D")

FINAL_ANSWER_PATTERN = re.compile(
    r"final\s*answer\s*[:：]?\s*\(?([ABCD])\)?",
    re.IGNORECASE,
)
SINGLE_LETTER_PATTERN = re.compile(r"\b([ABCD])\b")


def extract_choice(text):
    if not text:
        return None

    m = FINAL_ANSWER_PATTERN.search(text)
    if m:
        return m.group(1).upper()

    non_empty = [s for s in text.split("\n") if s.strip()]
    tail = "\n".join(non_empty[-3:]) if non_empty else ""
    m = SINGLE_LETTER_PATTERN.search(tail)
    if m:
        return m.group(1).upper()

    m = SINGLE_LETTER_PATTERN.search(text)
    if m:
        return m.group(1).upper()

    return None


def verify_answer(expect, answer):
    pred = extract_choice(answer)
    if pred is None:
        return False
    return pred == str(expect).strip().upper()
