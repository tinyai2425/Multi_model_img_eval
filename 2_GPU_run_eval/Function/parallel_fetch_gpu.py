import os
import time

import openai
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import infer_case


def parallel_fetch_reference_model(
    jsonl_path,
    server_ip,
    server_port,
    model_id,
    max_workers=None,
):
    """读取 1_Data_gen 产出的 GPU jsonl，并发请求 vLLM。"""
    client = openai.OpenAI(
        base_url=f"http://{server_ip}:{server_port}/v1",
        api_key="no-api-key-needed",
    )

    cases = infer_case.load_jsonl(jsonl_path)
    total = len(cases)
    if total == 0:
        raise ValueError(f"No samples found in {jsonl_path}")

    project_dir = os.path.dirname(os.path.abspath(jsonl_path))
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 4) * 2)

    start_time = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_seq = {}
        for seq, case in enumerate(cases):
            future = executor.submit(
                infer_case.measure_performance,
                client,
                model_id,
                case,
                project_dir,
            )
            future_to_seq[future] = seq

        for done_i, future in enumerate(as_completed(future_to_seq)):
            seq = future_to_seq[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"\n[ERROR] sample {seq} failed: {exc}")
                continue
            result["_seq"] = seq
            results.append(result)
            ttft = result.get("first_token_time") or 0.0
            ttc = result.get("total_time") or 0.0
            avg_elapsed = (time.time() - start_time) / (done_i + 1)
            remaining = avg_elapsed * (total - done_i - 1)
            print(
                f"\r{done_i + 1}/{total} samples processed "
                f"(TTFT:{ttft:.2f}s/TTC:{ttc:.2f}s), "
                f"{remaining:.2f}s remaining ...",
                end="",
                flush=True,
            )

    results.sort(key=lambda r: r.get("_seq", 0))
    for row in results:
        row.pop("_seq", None)
    df_results = pd.DataFrame(results)
    print(
        f"\r{len(results)} samples processed "
        f"({time.time() - start_time:.2f}s spent) in {jsonl_path}"
    )
    return df_results
