# 用法:
#   python Gen_img_cases.py <model_config.json> <MMSTAR_N> <OCRBENCH_N> \
#       [--enable_thinking true|false] \
#       [--mmstar_parquet PATH] [--ocrbench_parquet PATH]
#
# 产出（../../model-eval-storage/{model_name}/project-N/）:
#   project-N-IMG-GPU-{mm}-{ocr}.jsonl   MMStar+OCRBench 合成一份，给 vLLM
#   project-N-IMG-API-{mm}-{ocr}.jsonl   同上，鸿蒙只拷这一份（图片已是 base64）
#   MMStar_images/ / OCRBench_images/   仅 GPU 读图用
#   sampling_config.json
#
# N=0 表示跳过该数据集。同一 N 次抽样结果固定（无 RNG，按类别 round-robin）。

import base64
import json
import os
import shutil
import sys

import pandas as pd

sys.path.append(os.path.abspath("Function"))
import case_builder
import image_io
import prompts
import subset_select


USAGE = (
    "Usage: python Gen_img_cases.py <model_config.json> <MMSTAR_N> <OCRBENCH_N> "
    "[--enable_thinking true|false] "
    "[--mmstar_parquet PATH] [--ocrbench_parquet PATH]"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_MMSTAR = os.path.join(REPO_ROOT, "Data_set", "MMStar", "mmstar.parquet")
DEFAULT_OCRBENCH = os.path.join(
    REPO_ROOT, "Data_set", "OCRBench", "data", "test-00000-of-00001.parquet"
)
STORAGE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "model-eval-storage"))


def _flag_val(flag):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv):
        return None
    return sys.argv[i + 1]


def _parse_bool(value, default):
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"无法解析布尔值: {value}")


def get_unique_subdir(base_dir, prefix):
    counter = 1
    while os.path.exists(os.path.join(base_dir, f"{prefix}-{counter}")):
        counter += 1
    return f"{prefix}-{counter}"


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_jsonl(path, cases):
    with open(path, "w", encoding="utf-8") as f:
        for case in cases:
            json.dump(jsonable(case), f, ensure_ascii=False)
            f.write("\n")
    print(f"{len(cases)} cases saved to {path}")


def load_parquet(path, subset_n, by_cols):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 parquet: {path}")
    df = pd.read_parquet(path)
    if "index" not in df.columns:
        df = df.reset_index(drop=True)
        df["index"] = df.index.astype(int)
    if subset_n >= len(df):
        print(f"[INFO] requested {subset_n} >= dataset size {len(df)}, using all rows")
        return df.sort_values("index", kind="stable").reset_index(drop=True)
    selected = subset_select.stratified_subset(df, subset_n, by_cols=by_cols)
    print(
        f"[INFO] stratified subset {len(selected)}/{len(df)} by {by_cols} "
        f"(deterministic)"
    )
    return selected


def dump_row_image(row, images_dir, stem):
    img_bytes = image_io.coerce_image_bytes(row.get("image"))
    if not img_bytes:
        raise ValueError(f"empty image for index={row.get('index')}")
    mime = image_io.detect_mime(img_bytes)
    filename = f"{stem}{image_io.mime_to_ext(mime)}"
    abs_path = os.path.join(images_dir, filename)
    image_io.save_image(img_bytes, abs_path)
    return filename, mime, img_bytes


def _pair_cases(sampled, prompt, mime, img_bytes, test_case_name, expect, benchmark, extra_meta, image_rel):
    gpu = case_builder.build_gpu_case(
        sampled, prompt, mime, test_case_name, expect, benchmark, extra_meta, image_rel
    )
    b64 = base64.b64encode(img_bytes).decode("ascii")
    api = case_builder.build_api_case(
        sampled, prompt, mime, b64, test_case_name, expect, benchmark, extra_meta
    )
    return gpu, api


def generate_mmstar(df, sampled, project_name, project_path):
    images_dirname = "MMStar_images"
    images_dir = os.path.join(project_path, images_dirname)
    os.makedirs(images_dir, exist_ok=True)
    gpu_cases = []
    api_cases = []
    for row in df.to_dict(orient="records"):
        idx = int(case_builder.to_native(row["index"]))
        question = str(row["question"])
        expect = str(row["answer"]).strip().upper()
        category = case_builder.to_native(row.get("category"))
        l2_category = case_builder.to_native(row.get("l2_category"))
        stem = f"{idx:06d}"
        filename, mime, img_bytes = dump_row_image(row, images_dir, stem)
        prompt = prompts.MMSTAR_PROMPT_TEMPLATE.format(question=question)
        vertical = case_builder.sanitize_token(l2_category)
        test_case_name = f"{project_name}-MMStar-test-{vertical}-{idx}"
        extra_meta = {
            "index": idx,
            "question": question,
            "category": category,
            "l2_category": l2_category,
        }
        gpu, api = _pair_cases(
            sampled,
            prompt,
            mime,
            img_bytes,
            test_case_name,
            expect,
            "MMStar",
            extra_meta,
            f"{images_dirname}/{filename}",
        )
        gpu_cases.append(gpu)
        api_cases.append(api)
    return gpu_cases, api_cases


def generate_ocrbench(df, sampled, project_name, project_path):
    images_dirname = "OCRBench_images"
    images_dir = os.path.join(project_path, images_dirname)
    os.makedirs(images_dir, exist_ok=True)
    gpu_cases = []
    api_cases = []
    for row in df.to_dict(orient="records"):
        idx = int(case_builder.to_native(row["index"]))
        question = str(row["question"])
        expect = case_builder.ocrbench_expect_list(row.get("answer"))
        question_type = case_builder.to_native(row.get("question_type"))
        dataset = case_builder.to_native(row.get("dataset"))
        stem = f"{idx:06d}"
        filename, mime, img_bytes = dump_row_image(row, images_dir, stem)
        prompt = prompts.OCRBENCH_PROMPT_TEMPLATE.format(question=question)
        vertical = case_builder.sanitize_token(question_type)
        test_case_name = f"{project_name}-OCRBench-test-{vertical}-{idx}"
        extra_meta = {
            "index": idx,
            "question": question,
            "question_type": question_type,
            "dataset": dataset,
        }
        gpu, api = _pair_cases(
            sampled,
            prompt,
            mime,
            img_bytes,
            test_case_name,
            expect,
            "OCRBench",
            extra_meta,
            f"{images_dirname}/{filename}",
        )
        gpu_cases.append(gpu)
        api_cases.append(api)
    return gpu_cases, api_cases


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(USAGE)
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        mmstar_n = int(sys.argv[2])
        ocrbench_n = int(sys.argv[3])
    except ValueError:
        raise ValueError("MMSTAR_N 和 OCRBENCH_N 必须是整数（0 表示跳过）")

    if mmstar_n < 0 or ocrbench_n < 0:
        raise ValueError("用例个数不能为负数")
    if mmstar_n == 0 and ocrbench_n == 0:
        raise ValueError("MMSTAR_N 和 OCRBENCH_N 不能同时为 0")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    enable_thinking = _parse_bool(
        _flag_val("--enable_thinking"),
        bool(config.get("enable_thinking", False)),
    )
    mmstar_parquet = _flag_val("--mmstar_parquet") or DEFAULT_MMSTAR
    ocrbench_parquet = _flag_val("--ocrbench_parquet") or DEFAULT_OCRBENCH

    model_name = config.get("model_name")
    if not model_name:
        raise ValueError("model_config 缺少 model_name")

    project_base = os.path.join(STORAGE_ROOT, model_name)
    os.makedirs(project_base, exist_ok=True)
    project_name = get_unique_subdir(project_base, "project")
    project_path = os.path.join(project_base, project_name)
    os.makedirs(project_path, exist_ok=True)

    print(f"PROJECT_NAME {project_name}")
    print(f"Full path to project dir: {project_path}")
    print(f"[INFO] enable_thinking={enable_thinking}")
    sampled = case_builder.read_sampling(config, enable_thinking)

    gpu_all = []
    api_all = []
    n_mm = 0
    n_ocr = 0

    if mmstar_n > 0:
        df_mm = load_parquet(mmstar_parquet, mmstar_n, ["l2_category"])
        gpu_cases, api_cases = generate_mmstar(
            df_mm, sampled, project_name, project_path
        )
        n_mm = len(gpu_cases)
        gpu_all.extend(gpu_cases)
        api_all.extend(api_cases)
        print(f"[INFO] MMStar images (GPU only) -> {os.path.join(project_path, 'MMStar_images')}")

    if ocrbench_n > 0:
        df_ocr = load_parquet(ocrbench_parquet, ocrbench_n, ["question_type"])
        gpu_cases, api_cases = generate_ocrbench(
            df_ocr, sampled, project_name, project_path
        )
        n_ocr = len(gpu_cases)
        gpu_all.extend(gpu_cases)
        api_all.extend(api_cases)
        print(
            f"[INFO] OCRBench images (GPU only) -> {os.path.join(project_path, 'OCRBench_images')}"
        )

    write_jsonl(
        os.path.join(project_path, f"{project_name}-IMG-GPU-{n_mm}-{n_ocr}.jsonl"),
        gpu_all,
    )
    write_jsonl(
        os.path.join(project_path, f"{project_name}-IMG-API-{n_mm}-{n_ocr}.jsonl"),
        api_all,
    )

    archived_config = dict(config)
    archived_config["enable_thinking"] = enable_thinking
    sampling_path = os.path.join(project_path, "sampling_config.json")
    with open(sampling_path, "w", encoding="utf-8") as f:
        json.dump(archived_config, f, indent=2, ensure_ascii=False)
    shutil.copy2(config_path, os.path.join(project_path, os.path.basename(config_path)))
    print(f"sampling_config.json saved to {sampling_path}")
    print("[ALL DONE]")
