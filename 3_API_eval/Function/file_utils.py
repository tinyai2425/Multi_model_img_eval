def iter_lines_safely(file_path):
    """逐行读取 API 日志，优先 utf-8，失败则 latin1 兜底。"""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin1"):
        try:
            with open(file_path, "r", encoding=enc, errors="replace") as f:
                for line in f:
                    yield line
            return
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码文件 {file_path}")
