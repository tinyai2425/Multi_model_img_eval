"""按 case['benchmark'] 分发给 MMStar / OCRBench 的评分函数。"""

import verify_ans_mmstar
import verify_ans_ocrbench


def benchmark_of(case):
    bench = case.get("benchmark")
    if bench:
        return str(bench)
    name = str(case.get("testCaseName") or "")
    if "OCRBench" in name:
        return "OCRBench"
    if "MMStar" in name:
        return "MMStar"
    return "Unknown"


def verify_case(case, response):
    bench = benchmark_of(case)
    if bench == "OCRBench":
        return verify_ans_ocrbench.verify_answer(
            case.get("expect"), response, case.get("question_type")
        )
    return verify_ans_mmstar.verify_answer(case.get("expect"), response)


def extract_prediction(case, response):
    if benchmark_of(case) == "OCRBench":
        return verify_ans_ocrbench.extract_answer(response)
    return verify_ans_mmstar.extract_choice(response)
