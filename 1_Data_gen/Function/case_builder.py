"""把 model_config 映射成 GPU / API 两条请求。config 里没有的采样字段不写进 jsonl。

Qwen3.5 官方采样（HuggingFace 模型卡）是：
  temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty
没有 frequency_penalty，因此默认 config 和生成请求都不带这个键。

字段位置：

GPU / vLLM（openai.ChatCompletions.create）
  顶层: model, messages, stream, stream_options, temperature, top_p,
        presence_penalty, seed, max_tokens
  extra_body: top_k, min_p, repetition_penalty,
              chat_template_kwargs.enable_thinking

鸿蒙 OpenAI-compatible API（全部顶层）
  model, messages, stream, seed, top_p, temperature,
  presence_penalty, enable_thinking

评分元数据（testCaseName / expect / ...）附在 jsonl 里；GPU 发送前剥掉。
"""

import re

IMAGE_PLACEHOLDER = "{base64_image}"

GPU_OPENAI_KEYS = (
    "model",
    "messages",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "presence_penalty",
    "seed",
    "max_tokens",
)

GPU_EXTRA_BODY_KEYS = (
    "top_k",
    "min_p",
    "repetition_penalty",
)

API_OPENAI_KEYS = (
    "model",
    "messages",
    "stream",
    "seed",
    "top_p",
    "temperature",
    "presence_penalty",
    "enable_thinking",
)


def sanitize_token(value):
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(value))
    return text.strip("_") or "unknown"


def to_native(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def ocrbench_expect_list(raw):
    if raw is None:
        return []
    try:
        return [str(x) for x in list(raw) if x is not None]
    except TypeError:
        return [str(raw)]


def _copy_present(dst, src, keys):
    for key in keys:
        if key in src and src[key] is not None:
            dst[key] = src[key]


def read_sampling(config, enable_thinking):
    """只读取 config 里实际出现的键，不给不存在的采样项填默认值。"""
    sampling = config.get("sampling") or {}
    sampled = {
        "model_name": config["model_name"],
        "enable_thinking": bool(enable_thinking),
    }
    if "max_tokens" in config:
        sampled["max_tokens"] = int(config["max_tokens"])
    if "seed" in config:
        sampled["seed"] = int(config["seed"])
    for key, value in sampling.items():
        if key.startswith("_"):
            continue
        sampled[key] = value
    return sampled


def build_messages(prompt, mime, image_payload):
    """image_payload: GPU 用占位符；API 用已经编好的 base64 字符串。"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{image_payload}"
                    },
                },
            ],
        }
    ]


def _eval_meta(test_case_name, expect, benchmark, extra_meta, image_file=None, mime=None):
    meta = {
        "testCaseName": test_case_name,
        "expect": expect,
        "benchmark": benchmark,
    }
    meta.update(extra_meta)
    if image_file:
        meta["image_file"] = image_file
    if mime:
        meta["image_mime"] = mime
    return meta


def build_gpu_case(sampled, prompt, mime, test_case_name, expect, benchmark, extra_meta, image_file):
    """vLLM：官方字段在顶层，扩展进 extra_body。图片仍用占位符，评测时读 image_file。"""
    case = {
        "model": sampled["model_name"],
        "messages": build_messages(prompt, mime, IMAGE_PLACEHOLDER),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    _copy_present(
        case, sampled, ("temperature", "top_p", "presence_penalty", "seed", "max_tokens")
    )
    extra = {}
    _copy_present(extra, sampled, GPU_EXTRA_BODY_KEYS)
    extra["chat_template_kwargs"] = {
        "enable_thinking": sampled["enable_thinking"]
    }
    case["extra_body"] = extra
    case.update(
        _eval_meta(
            test_case_name, expect, benchmark, extra_meta, image_file, mime
        )
    )
    return case


def build_api_case(sampled, prompt, mime, b64, test_case_name, expect, benchmark, extra_meta):
    """鸿蒙：OpenAI 顶层字段；图片 base64 已经编进 messages。"""
    case = {
        "model": sampled["model_name"],
        "messages": build_messages(prompt, mime, b64),
        "stream": True,
        "enable_thinking": sampled["enable_thinking"],
    }
    _copy_present(
        case, sampled, ("seed", "top_p", "temperature", "presence_penalty")
    )
    case.update(_eval_meta(test_case_name, expect, benchmark, extra_meta))
    return case
