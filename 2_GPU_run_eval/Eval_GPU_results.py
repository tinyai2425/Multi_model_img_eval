import os
import sys
from contextlib import ExitStack

import pandas as pd

sys.path.append(os.path.abspath("Function"))

import dispatch
import extend_metrics
import extract_name
import infer_case
import parallel_fetch_gpu
import report_split


USAGE = (
    "Usage: python Eval_GPU_results.py "
    "<gpu_jsonl_path> <ip> <port> <model_id> <version_flag> [--show_detail]"
)


def parse_args():
    if len(sys.argv) < 6:
        print(USAGE)
        sys.exit(1)
    jsonl_path = sys.argv[1]
    ip = sys.argv[2]
    port = int(sys.argv[3])
    model_id = sys.argv[4]
    version_flag = int(sys.argv[5])
    show_detail = "--show_detail" in sys.argv
    return jsonl_path, ip, port, model_id, version_flag, show_detail


def get_version_nums(version_flag):
    if version_flag < 1:
        raise ValueError("version_flag must be >= 1")
    return list(range(1, version_flag + 1))


if __name__ == "__main__":
    jsonl_path, ip, port, model_id, version_flag, show_detail = parse_args()
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(jsonl_path)

    project_base, file_name = extract_name.parse_project_base_and_filename(jsonl_path)
    version_nums = get_version_nums(version_flag)
    gpu_output_dir = extract_name.make_dated_output_dir(
        project_base, "GPU", dataset="IMG"
    )

    infer_case.verify_case = dispatch.verify_case
    infer_case.extract_prediction = dispatch.extract_prediction
    infer_case.extend_metrics = extend_metrics

    cases = infer_case.load_jsonl(jsonl_path)
    benches = sorted({dispatch.benchmark_of(c) for c in cases})
    for bench in benches:
        os.makedirs(os.path.join(gpu_output_dir, bench), exist_ok=True)

    print("Notice: Proxy settings should be disabled before proceeding.")
    print(f"[INFO] jsonl: {jsonl_path}")
    print(f"[INFO] benchmarks: {benches}")
    print(f"[INFO] vLLM endpoint: http://{ip}:{port}/v1, model={model_id}")
    print(f"[INFO] Output directory: {gpu_output_dir}")

    with ExitStack() as stack:
        writers = {}
        if show_detail:
            for bench in benches:
                if bench not in report_split.EVALUATORS:
                    continue
                path = os.path.join(gpu_output_dir, bench, "GPU_Summary.xlsx")
                writers[bench] = stack.enter_context(pd.ExcelWriter(path))

        for version_num in version_nums:
            print(f"\n[INFO] Processing version {version_num}...")
            ref_result_parquet_path = os.path.join(
                gpu_output_dir, f"GPU-Results-{version_num}.parquet"
            )
            df_results = parallel_fetch_gpu.parallel_fetch_reference_model(
                jsonl_path, ip, port, model_id
            )
            df_results.to_parquet(ref_result_parquet_path)
            print(f"[SAVE] Results saved to {ref_result_parquet_path}")

            groups = report_split.split_df(df_results)
            report_split.evaluate_groups(
                groups,
                gpu_output_dir,
                show_detail,
                sheet_prefix=f"v{version_num}",
                mode="GPU",
                writers=writers,
                analysis_name=f"analysis_v{version_num}.png",
            )
            print(f"[DONE] Evaluation for version {version_num}")

    if show_detail:
        report_split.write_averages(gpu_output_dir, benches, mode="GPU")
