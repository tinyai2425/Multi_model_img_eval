import os
import sys
from contextlib import ExitStack

import pandas as pd

sys.path.append(os.path.abspath("Function"))
sys.path.append(os.path.abspath("../2_GPU_run_eval/Function"))

import API_collect
import dispatch
import extend_metrics
import extract_name
import file_utils
import report_split


USAGE = "Usage: python Eval_API_results.py <test_result_txt> [--show_detail]"


def parse_args():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)
    test_result_path = sys.argv[1]
    show_detail = "--show_detail" in sys.argv
    return test_result_path, show_detail


if __name__ == "__main__":
    test_result_path, show_detail = parse_args()
    if not os.path.isfile(test_result_path):
        raise FileNotFoundError(test_result_path)

    project_base, file_name = extract_name.parse_project_base_and_filename(
        test_result_path
    )
    api_output_dir = extract_name.make_dated_output_dir(
        project_base, "API", dataset="IMG"
    )

    API_collect.GET_answer = extend_metrics.GET_answer
    API_collect.calculate_repetition_rate = extend_metrics.calculate_repetition_rate
    API_collect.calculate_token_entropy = extend_metrics.calculate_token_entropy
    API_collect.verify_case = dispatch.verify_case
    API_collect.extract_prediction = dispatch.extract_prediction
    API_collect.iter_lines_safely = file_utils.iter_lines_safely

    print(f"[INFO] input txt    : {test_result_path}")
    print(f"[INFO] output dir   : {api_output_dir}")

    df_results = API_collect.parse_llm_api_results(test_result_path)
    if len(df_results) == 0:
        print("[ERROR] No test cases parsed; check the log format.")
        sys.exit(1)

    parquet_path = os.path.join(api_output_dir, "API-Results.parquet")
    df_results.to_parquet(parquet_path)
    print(f"[SAVE] Results saved to {parquet_path}")

    groups = report_split.split_df(df_results)
    benches = list(groups.keys())
    print(f"[INFO] benchmarks: {benches}")

    with ExitStack() as stack:
        writers = {}
        if show_detail:
            for bench in benches:
                if bench not in report_split.EVALUATORS:
                    continue
                os.makedirs(os.path.join(api_output_dir, bench), exist_ok=True)
                path = os.path.join(api_output_dir, bench, "API_Summary.xlsx")
                writers[bench] = stack.enter_context(pd.ExcelWriter(path))
        report_split.evaluate_groups(
            groups,
            api_output_dir,
            show_detail,
            sheet_prefix="api",
            mode="API",
            writers=writers,
            analysis_name="analysis.png",
        )

    if show_detail:
        report_split.write_averages(api_output_dir, benches, mode="API")
