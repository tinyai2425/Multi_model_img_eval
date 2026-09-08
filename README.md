# 多模态图像评估（MMStar / OCRBench）

在 `Ceval_ref` 的目录形态上，复用 `Multi_model_eval` 的图像评测逻辑，并把 **GPU（vLLM）** 与 **API（鸿蒙 PC curl）** 拆成同一批题。不做 OMC，也不做 Video-MME。

`model_config` 只写该模型官方有的采样项（Qwen3.5 模型卡：`temperature` / `top_p` / `top_k` / `min_p` / `presence_penalty` / `repetition_penalty`）。config 里没有的键不会写进 jsonl。

| 来源 | 放哪里 |
| --- | --- |
| temperature / top_p / presence_penalty / seed | GPU、API 都在 Chat Completions **顶层** |
| max_tokens / stream_options | **仅 GPU 顶层**（vLLM 用） |
| top_k / min_p / repetition_penalty | **仅 GPU `extra_body`**（vLLM 扩展，不能当 create() 关键字） |
| enable_thinking | GPU → `extra_body.chat_template_kwargs`；鸿蒙 → **顶层** `enable_thinking` |

同一 `project-N` 里 GPU / API 的题目、图片、采样数值相同，请求信封不同。

## 仓库结构

```
Multi_model_img_eval/
├── README.md
├── requirements.txt
├── model_config/Qwen3.5-4B-test-config.json
├── Data_set/                    ← 不进 git，从 HuggingFace clone
├── 1_Data_gen/                  ← 抽样 + 写 GPU/API jsonl
├── 2_GPU_run_eval/              ← 读 GPU jsonl + 本地图片，打 vLLM
└── 3_API_eval/
    ├── Machine_test/run_chat.sh ← 拷到鸿蒙：只有 curl
    └── Eval_*_API_results.py    ← 把 PC 日志拷回来再评分
```

生成结果写到仓库外的 `../model-eval-storage/{model_name}/project-N/`（与 Ceval / BFCL 同一套存储）。

## 一、环境

推荐 Python 3.10+。评测机（生成用例、GPU、解析 API 日志）安装：

```bash
conda create -n mm_img_eval python=3.10 -y
conda activate mm_img_eval
pip install -r requirements.txt
```

**不依赖** transformers / 本地 tokenizer。GPU token 数来自 vLLM `usage`；API 有 usage 就用，没有则退回字符长度。

鸿蒙 PC **不要**装上述 Python 包，也 **不要** python。那边只需要 `curl`。

### vLLM（必须支持多模态）

旧版 vLLM 往往没有图像接口。参考 Qwen3-VL / Qwen3.5 说明：

```bash
pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen3.5-4B \
    --served-model-name qwen35_4b \
    --host 0.0.0.0 --port 8895 --max-model-len 32768
```

内网评估前先 `unset http_proxy https_proxy all_proxy`，否则 openai 客户端会把内网 IP 打到代理上。

## 二、数据集

不上传 git。在本目录下：

```bash
git lfs install
mkdir -p Data_set && cd Data_set
git clone https://huggingface.co/datasets/Lin-Chen/MMStar
git clone https://huggingface.co/datasets/echo840/OCRBench
```

需要存在：

```
Data_set/MMStar/mmstar.parquet
Data_set/OCRBench/data/test-00000-of-00001.parquet
```

若已经在 `../Multi_model_eval/Data_set` 下过，可以软链过来，不必再下一遍。

## 三、生成用例（1_Data_gen）

编辑 `1_Data_gen/run_gen_img.sh`：

```bash
CONFIG_PATH="../model_config/Qwen3.5-4B-test-config.json"
MMSTAR_N=120          # 0 = 跳过
OCRBENCH_N=120        # 0 = 跳过
ENABLE_THINKING=false # 写入每条用例，覆盖 model_config 里的同名字段
```

```bash
cd 1_Data_gen
bash run_gen_img.sh
```

同一 `N` 抽到的题 **永远相同、顺序相同**（按类别 round-robin，没有随机种子）：

| 数据集   | 分层字段           |
| -------- | ------------------ |
| MMStar   | `l2_category`（18 类） |
| OCRBench | `question_type`（10 类） |

产出示例：

```
../model-eval-storage/Qwen3.5-4B/project-1/
  project-1-IMG-GPU-120-120.jsonl    ← MMStar+OCRBench 合成一份，给 vLLM
  project-1-IMG-API-120-120.jsonl    ← 同上，鸿蒙只拷这一份
  MMStar_images/                     ← 只给 GPU 用
  OCRBench_images/
  sampling_config.json
```

文件名里两个数字依次是 MMStar 条数、OCRBench 条数。每条用例带 `benchmark` 字段，评测时按它拆开计分。

GPU jsonl（vLLM / OpenAI Python 客户端）：

```json
{
  "model": "Qwen3-4B",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "..."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{base64_image}"}}
    ]
  }],
  "stream": true,
  "stream_options": {"include_usage": true},
  "temperature": 0.7,
  "top_p": 0.8,
  "presence_penalty": 1.5,
  "seed": 99,
  "max_tokens": 5000,
  "extra_body": {
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": false}
  },
  "image_file": "MMStar_images/000042.jpg",
  "benchmark": "MMStar",
  "testCaseName": "project-1-MMStar-test-...",
  "expect": "A"
}
```

API jsonl（鸿蒙，只拷这一个文件）：

```json
{
  "model": "Qwen3-4B",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "..."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}}
    ]
  }],
  "stream": true,
  "seed": 99,
  "top_p": 0.8,
  "temperature": 0.7,
  "presence_penalty": 1.5,
  "enable_thinking": false,
  "benchmark": "MMStar",
  "testCaseName": "project-1-MMStar-test-...",
  "expect": "A"
}
```

`testCaseName` / `expect` 等是评分元数据。GPU 客户端发送前会剥掉；鸿蒙 curl 整行 POST，服务端应忽略未知键。

要改 thinking 或温度：改 `model_config` 或 shell 里的 `ENABLE_THINKING`，然后重新生成。

## 四、GPU 评测（2_GPU_run_eval）

改 `run_eval_gpu.sh` 顶部的 jsonl 路径、vLLM IP/端口/served-model-name。一次跑完 jsonl 里两个数据集。

```bash
cd 2_GPU_run_eval
bash run_eval_gpu.sh
```

结果落到同一 project 下的日期目录，同日重跑自动加 `-2`、`-3`，不会覆盖：

```
.../project-1/GPU-IMG-YYYYMMDD/
  GPU-Results-1.parquet
  MMStar/
    GPU_Summary.xlsx
    analysis_v1.png
    summary_GPU_avg.md
    breakdown_GPU_avg.md
  OCRBench/
    GPU_Summary.xlsx
    analysis_v1.png
    ...
```

`VERSION_FLAG>1` 时会连跑多轮再对 sheet 求平均。

## 五、API 评测（3_API_eval）

### 5.1 鸿蒙 PC 上跑推理

只拷这两样：

```
project-1-IMG-API-120-120.jsonl
3_API_eval/Machine_test/run_chat.sh
```

不要拷图片目录，也不要 python。

```sh
# 默认 POST http://localhost:11434/v1/chat/completions
sh run_chat.sh /path/to/project-1-IMG-API-120-120.jsonl /path/to/project-1-IMG-API-120-120.txt
```

jsonl 每行已经是完整请求体（含 base64），等价于：

```sh
curl http://localhost:11434/v1/chat/completions -d '{
  "model":"Qwen3-4B",
  "messages":[{"role":"user","content":[
    {"type":"text","text":"..."},
    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}
  ]}],
  "stream":true,"seed":99,"top_p":0.8,"temperature":0.7,
  "presence_penalty":1.5,"enable_thinking":false
}'
```

换地址：`API_URL=http://127.0.0.1:11434/v1/chat/completions sh run_chat.sh ...`

脚本用 `curl -d @file`，避免超大 base64 撑爆命令行。日志每条 3 行：请求 jsonl / 压成一行的响应 / `API_total_time: <秒>`。

### 5.2 拷回日志后评分

```bash
cd 3_API_eval
# 改 run_eval_api.sh 里的 txt 路径
bash run_eval_api.sh
```

端侧 curl 没有 TTFT 切分，`TTFT` 为 NaN，`TPS` 用整段 `API_total_time` 近似。精度、repeat、entropy 与 GPU 同一套规则。

## 六、评分口径

- **MMStar**：抽取 `Final answer: A/B/C/D`，与标准答案比。
- **OCRBench**：官方子串匹配；HME 题会去掉 LaTeX 空白后再比。满分 1000 时 `OCRBench` 列即官方 Score；子集上该列等于答对条数。

## 七、常见问题

**Q: 鸿蒙上要装 python 吗？**  
A: 不要。`run_chat.sh` 只有 `sh` + `curl`。

**Q: 鸿蒙上报图片找不到？**  
A: API 用例不要带图片目录。确认拷的是 `*-IMG-API-*.jsonl`（messages 里已有 `data:image/...;base64,...`），不是 GPU 那份。

**Q: 想换 30 条 / 500 条？**  
A: 改 `run_gen_img.sh` 的 `MMSTAR_N` / `OCRBENCH_N` 重新生成。同样的 N 永远是同一批题。

**Q: GPU 和 API 分数差很多？**  
A: 先确认是同一次 `project-N` 里成对的 GPU/API jsonl。再核对端侧 `model` 名、以及 jsonl 里的 `enable_thinking` / `extra_body.chat_template_kwargs` 是否一致。
