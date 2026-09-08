"""Read parquet image fields and dump raw bytes next to generated jsonl."""

import os

MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


def coerce_image_bytes(image_field):
    """parquet 里的 image 列可能是 bytes，也可能是 {'bytes': ..., 'path': ...}。"""
    if image_field is None:
        return None
    if isinstance(image_field, (bytes, bytearray)):
        return bytes(image_field)
    if isinstance(image_field, dict):
        if image_field.get("bytes") is not None:
            return bytes(image_field["bytes"])
        if image_field.get("path"):
            with open(image_field["path"], "rb") as f:
                return f.read()
    if hasattr(image_field, "tobytes"):
        return bytes(image_field.tobytes())
    raise TypeError(f"Unsupported image field type: {type(image_field)}")


def detect_mime(img_bytes):
    if not img_bytes or len(img_bytes) < 12:
        return "image/jpeg"
    head = img_bytes[:12]
    if head.startswith(b"\x89PNG"):
        return "image/png"
    if head[:3] == b"GIF":
        return "image/gif"
    if head[:2] == b"\xff\xd8":
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def mime_to_ext(mime):
    return MIME_EXT.get(mime, ".jpg")


def save_image(img_bytes, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(img_bytes)
