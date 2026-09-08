"""Read a generated GPU jsonl case and call vLLM Chat Completions.

只把 GPU_OPENAI_KEYS + extra_body 传给 openai 客户端；评分元数据和 image_file
不会进请求。图片按 image_file 读入后替换 messages 里的 {base64_image}。
"""

import base64
import copy
import json
import os
import time

# 与 1_Data_gen/Function/case_builder.py 中 GPU_OPENAI_KEYS / IMAGE_PLACEHOLDER 对齐
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

verify_case = None
extract_prediction = None
extend_metrics = None


def load_jsonl(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _detect_mime(img_bytes, fallback="image/jpeg"):
    if not img_bytes or len(img_bytes) < 12:
        return fallback
    head = img_bytes[:12]
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head[:3] == b"GIF":
        return "image/gif"
    if head[:2] == b"\xff\xd8":
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return fallback


def resolve_image_path(case, project_dir):
    rel = case.get("image_file") or ""
    if not rel:
        raise ValueError(f"case missing image_file: {case.get('testCaseName')}")
    path = rel if os.path.isabs(rel) else os.path.join(project_dir, rel)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"image not found: {path}")
    return path


def fill_messages_with_image(messages, img_bytes, mime):
    b64 = base64.b64encode(img_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    filled = copy.deepcopy(messages)
    blob = json.dumps(filled, ensure_ascii=False)
    if IMAGE_PLACEHOLDER in blob:
        return json.loads(blob.replace(IMAGE_PLACEHOLDER, b64))
    for msg in filled:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                part.setdefault("image_url", {})["url"] = data_url
    return filled


def _prompt_text_from_messages(messages):
    parts = []
    for msg in messages or []:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text") or "")
    return "\n".join(parts)


def enable_thinking_of(case):
    extra = case.get("extra_body") or {}
    kwargs = extra.get("chat_template_kwargs") or {}
    return bool(kwargs.get("enable_thinking", False))


def openai_create_kwargs(case, model_id, messages):
    """只抽出 Chat Completions 官方字段 + extra_body，其它键全部丢掉。"""
    kwargs = {"model": model_id}
    for key in GPU_OPENAI_KEYS:
        if key == "model":
            continue
        if key == "messages":
            kwargs["messages"] = messages
            continue
        if key in case:
            kwargs[key] = case[key]
    extra = case.get("extra_body")
    if extra:
        kwargs["extra_body"] = extra
    return kwargs


def measure_performance(client, model_id, case, project_dir):
    img_path = resolve_image_path(case, project_dir)
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    mime = case.get("image_mime") or _detect_mime(img_bytes)
    messages = fill_messages_with_image(case.get("messages") or [], img_bytes, mime)
    prompt_text = _prompt_text_from_messages(messages)
    create_kwargs = openai_create_kwargs(case, model_id, messages)

    start_time = time.time()
    first_token_time = None
    full_response = ""
    usage = None

    response = client.chat.completions.create(**create_kwargs)

    for chunk in response:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None)
        if piece:
            if first_token_time is None:
                first_token_time = time.time() - start_time
            full_response += piece

    total_time = time.time() - start_time
    decode_time = (
        total_time - first_token_time if first_token_time is not None else None
    )

    if usage is not None:
        prompt_token_len = int(getattr(usage, "prompt_tokens", 0) or 0)
        response_token_len = int(getattr(usage, "completion_tokens", 0) or 0)
    else:
        prompt_token_len = len(prompt_text)
        response_token_len = len(full_response)

    expect = case.get("expect")
    prediction = (
        extract_prediction(case, full_response) if extract_prediction else None
    )
    correct = verify_case(case, full_response) if verify_case else False

    return {
        "testCaseName": case.get("testCaseName", ""),
        "index": case.get("index"),
        "benchmark": case.get("benchmark"),
        "question": case.get("question"),
        "category": case.get("category"),
        "l2_category": case.get("l2_category"),
        "question_type": case.get("question_type"),
        "dataset": case.get("dataset"),
        "expect": expect if isinstance(expect, str) else json.dumps(expect, ensure_ascii=False),
        "response": full_response,
        "prediction": prediction,
        "prompt_token_len": prompt_token_len,
        "response_token_len": response_token_len,
        "first_token_time": first_token_time,
        "total_time": total_time,
        "decode_time": decode_time,
        "correct": bool(correct),
        "get_ans": extend_metrics.GET_answer(full_response),
        "repeat": extend_metrics.calculate_repetition_rate(full_response),
        "entropy": extend_metrics.calculate_char_entropy(full_response),
        "enable_thinking": enable_thinking_of(case),
    }
