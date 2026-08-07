# ToolBench 目录功能与数据流

> 扫描范围：`D:\Work\PythonProgram\agent\repos\ToolBench` 目前检出的本地版本。
> 仓库只带 `data_example/`；README 中的完整 `data/` 需要另行下载。

## 1. 总目录功能表

| 目录/文件 | 功能 | 主要输入 | 主要输出/下游 |
|---|---|---|---|
| `README.md` / `README_ZH.md` | 数据下载、训练、推理、评测说明 | — | 官方操作说明 |
| `data/` | 完整数据包的约定位置；当前 Git 仓库未包含 | 下载的数据包 | 预处理、训练、推理、评测 |
| `data_example/` | 小规模结构样例 | 仓库内置样例 | 理解 query、answer、toolenv 格式 |
| `preprocess/` | 将原始标注转换成 SFT 或 Retriever 格式 | `instruction/`、`answer/`、测试 id | SFT JSON、Retriever TSV/TXT |
| `toolbench/train/` | ToolLLaMA 全参/LoRA SFT | `toolllama_*_train.json`、eval JSON | Hugging Face 模型或 PEFT adapter |
| `toolbench/inference/` | 闭域/开放域推理、搜索、工具执行 | query、工具环境、LLM、可选 Retriever | 每题一个搜索树和答案 JSON |
| `toolbench/retrieval/` | API Retriever 训练、评估、独立调用示例 | `corpus.tsv`、query、qrels | SentenceTransformer Retriever |
| `toolbench/model/` | 模型/对话模板适配、delta 权重与压缩工具 | 模型与模板名 | 训练和推理使用的模型对象 |
| `toolbench/tool_conversation.py` | 对话模板和分隔符的底层数据结构 | 角色消息 | 格式化 prompt |
| `toolbench/utils.py` | 名称标准化、system function 注入、Retriever 文档拼接、RoPE 扩长 | API schema、DataFrame | 训练/推理通用转换 |
| `toolbench/tooleval/` | 答案格式转换、pass rate、偏好评测、排行榜 | 推理结果 JSON | 评测 CSV/JSON |
| `scripts/` | 官方预处理、训练和推理实验命令 | 环境变量和数据路径 | 调用上述 Python 入口 |
| `ds_configs/` | DeepSpeed ZeRO stage 2/3 配置 | 训练参数 | 分布式训练配置 |
| `reproduction_data/` | README 约定的复现实验结果目录；仓库未包含完整数据 | 各模型推理结果 | 论文复现/ToolEval |

## 2. `data/`：原始、训练与评测数据如何组织

README 给出的完整数据包结构如下：

```text
data/
├── instruction/                  # 原始生成指令，G1/G2/G3
├── answer/                       # CoT/DFSDT 解路径、工具返回、搜索树
├── toolenv/                      # API 文档、Python 实现、响应样例
│   ├── tools/
│   │   └── <Category>/
│   │       ├── <tool>.json       # 工具和 API schema
│   │       └── <tool>/api.py     # 本地/自定义 API 实现
│   └── response_examples/        # API 响应 schema/样例
├── retrieval/                    # Retriever 预处理结果
│   └── G1/
│       ├── train.json
│       ├── test.json
│       ├── corpus.tsv
│       ├── train.query.txt
│       ├── test.query.txt
│       ├── qrels.train.tsv
│       └── qrels.test.tsv
├── test_instruction/             # 各测试集的 query
├── test_query_ids/               # SFT/推理测试划分 id
├── retrieval_test_query_ids/     # Retriever 测试划分 id
├── toolllama_G123_dfs_train.json # 可直接用于 SFT 的训练集
└── toolllama_G123_dfs_eval.json  # 可直接用于 SFT 的验证集

reproduction_data/
├── chatgpt_cot/
├── chatgpt_dfs/
└── toolllama_dfs/                # 不同模型/算法的复现结果
```

### 数据层次

| 层次 | 典型路径 | 内容 |
|---|---|---|
| 原始指令 | `data/instruction/G1_query.json` | `query_id`、自然语言 `query`、候选 `api_list`、正例 `relevant APIs` |
| 原始工具环境 | `data/toolenv/tools/<category>/<tool>.json` | 工具描述、API 名、方法、必选/可选参数、返回结构 |
| 原始答案标注 | `data/answer/G1_answer/*_DFS_woFilter_w2.json` | 完整搜索树、候选链、真实 observation、`answer_generation` |
| SFT 训练数据 | `data/toolllama_G123_dfs_train.json` | JSON 数组，每项为一个“轨迹前缀 → 下一次 assistant 调用”的监督样本 |
| SFT 验证数据 | `data/toolllama_G123_dfs_eval.json` | 与训练集格式一致 |
| Retriever 训练/测试 | `data/retrieval/G1/*` | corpus、query、正例 qrels |
| 推理测试 | `data/test_instruction/*.json` | 闭域 query 通常带 `api_list`；开放域可只依赖 query 和 Retriever |
| 推理结果 | `output/<query_id>_<method>.json` | 搜索树、候选、函数 schema、训练轨迹、最终答案 |

G1/G2/G3 的语义：

- **G1**：单工具场景。
- **G2**：同类别内多工具场景。
- **G3**：跨类别/集合多工具场景。

## 3. `toolbench/`：核心代码入口

项目没有一个统一的 `main.py`。核心入口按任务分散：

| 任务 | 实际入口 | 核心下游 |
|---|---|---|
| 全参 SFT | `toolbench/train/train_mem.py` | FlashAttention patch → `train.py::train()` |
| 基础 SFT | `toolbench/train/train.py` | Hugging Face `Trainer` |
| LoRA SFT | `toolbench/train/train_lora.py` | PEFT + DeepSpeed + `Trainer` |
| 闭域推理 | `toolbench/inference/qa_pipeline.py` | `pipeline_runner(args)` |
| 开放域推理 | `toolbench/inference/qa_pipeline_open_domain.py` | `pipeline_runner(args, add_retrieval=True)` |
| Retriever 训练 | `toolbench/retrieval/train.py` | SentenceTransformer |
| Retriever 独立测试 | `toolbench/retrieval/inference_example.py` | top-5 cosine retrieval |
| ToolEval | `toolbench/tooleval/*.py` 和 `run_*.sh` | 答案转换、pass rate、偏好比较 |

推理内部职责：

```text
qa_pipeline[_open_domain].py
└── Downstream_tasks/rapidapi.py::pipeline_runner
    ├── LLM/*                         # ToolLLaMA / LoRA / ChatGPT / Davinci
    ├── LLM/retriever.py              # 开放域 API 检索
    ├── rapidapi_wrapper              # query、functions、工具状态与 step()
    ├── Algorithms/single_chain.py    # CoT@N
    ├── Algorithms/DFS.py             # DFS / DFSDT
    ├── Tree/Tree.py                   # 搜索树与训练轨迹回收
    └── server.py                      # 自有 RapidAPI key 或自定义 API 的执行后端
```

## 4. `toolbench/train/`：SFT 入口与训练格式

### 入口

- 全参官方入口：`toolbench/train/train_mem.py`。先应用 FlashAttention monkey patch，再调用 `train.py::train()`。
- LoRA 官方入口：`toolbench/train/train_lora.py`。
- 两者共享 `train.py` 中的数据加载、prompt 格式化和 label masking。

### SFT 文件格式

文件是 **JSON 数组，不是 JSONL**：

```json
[
  {
    "id": "Step 4: <user query>",
    "conversations": [
      {"from": "system", "value": "<system prompt + available API schemas>"},
      {"from": "user", "value": "<query>"},
      {
        "from": "assistant",
        "value": "\nThought: ...\nAction: api_name_for_tool_name\nAction Input: {\"x\": 1}"
      },
      {
        "from": "function",
        "value": "{\"error\": \"\", \"response\": \"...\"}"
      },
      {
        "from": "assistant",
        "value": "\nThought: ...\nAction: Finish\nAction Input: {\"return_type\": \"give_answer\", \"final_answer\": \"...\"}"
      }
    ]
  }
]
```

### 训练目标

1. `get_conversation_template("tool-llama-single-round")` 将四角色拼成 prompt。
2. tokenizer 右侧 padding/truncation 到 `model_max_length`。
3. `labels = input_ids.clone()`。
4. system、user、function 和历史 assistant 全部 mask 为 `-100`。
5. **只对当前样本最后一个 assistant 回复计算 causal LM loss**。

这与预处理生成的“逐步轨迹前缀”配套：一条完整工具链被拆成多个训练样本，每个样本学习下一次 Action。

### 数据加载和训练差异

| 项目 | 全参 | LoRA |
|---|---|---|
| 入口 | `train_mem.py` | `train_lora.py` |
| 参数更新 | 全模型 | 默认只注入 `q_proj`、`v_proj` |
| 默认 LoRA | — | r=8, alpha=16, dropout=0.05, bias=none |
| 官方分布式 | torchrun + FSDP | DeepSpeed stage 2 |
| 官方 epoch | 2 | 5 |
| 保存 | 完整模型 state dict | PEFT adapter 权重 |

`lazy_preprocess=True` 表示按样本首次访问时 tokenize 并缓存；不提供 `eval_data_path` 时，代码内部随机按 98%/2% 划分。

## 5. `toolbench/inference/`：如何逐步调用工具

### 闭域与开放域

- **闭域**：query JSON 已提供 `api_list`，环境直接加载这些 API。
- **开放域**：先用 Retriever 从 `corpus.tsv` 召回 top-k API，再加载本地完整 API JSON；之后与闭域共用同一执行流程。

### 一次工具循环

```text
用户 query
  ↓
准备 system/user prompt + function schemas + Finish
  ↓
LLM 生成纯文本 Thought / Action / Action Input
  ↓ react_parser
assistant {content, function_call{name, arguments}}
  ↓
搜索树新增 Thought → Action → Action Input
  ↓
rapidapi_wrapper.step(action_name, action_input)
  ├─ Finish：校验 return_type/final_answer，结束
  ├─ 自有 RapidAPI key / 自定义 API：server.py 本地执行
  └─ ToolBench key：请求远程 ToolBench API 服务
  ↓
得到 observation + status code
  ↓
追加 assistant function_call 和 function observation 到消息历史
  ↓
未结束则再次调用 LLM
```

### 搜索方法

| method | 实现 | 行为 |
|---|---|---|
| `CoT@N` | `Algorithms/single_chain.py` | 从头运行最多 N 条独立链，找到答案即停 |
| `DFS_woFilter_w2` | `Algorithms/DFS.py` | DFSDT；每层宽度 2，生成候选后立即深挖，不额外用 LLM 排名 |
| `DFS_*_wN` | `Algorithms/DFS.py` | 每层 N 个候选；启用 filter 时先由 LLM 比较排序，再深挖 |

默认上限：单链/单分支深度 12；DFS 最多 200 次模型调用。

关键 status code：0 正常、1 函数名幻觉、2 参数错误、3 最终答案、4 放弃并回溯、5 超时、6 API 不工作、7/8 订阅或授权错误、9/10 限流、11 API 错误、12 请求失败。

## 6. `toolbench/retrieval/`：API Retriever 如何训练与调用

### 预处理后的检索文档

每个 API document 由下列字段拼成字符串：

```text
category_name, tool_name, api_name, api_description,
required_params: <JSON>, optional_params: <JSON>, return_schema: <JSON>
```

### 训练

```text
query + 正例 API document
  ↓
bert-base-uncased Transformer + Pooling
  ↓
MultipleNegativesRankingLoss
  └─ batch 内其他 API document 作为负例
  ↓
APIEvaluator（NDCG）
  ↓
output_path/<timestamp>/
```

官方默认：5 epochs、batch 32、lr 2e-5、warmup 500、max length 256。

### Pipeline 调用

1. `ToolRetriever` 加载 `corpus.tsv` 和训练好的 SentenceTransformer。
2. 启动时一次性编码整个 API corpus。
3. 每个用户 query 编码后，以 cosine similarity 做 semantic search。
4. 内部先召回 `10 * top_k`，再过滤本地不存在或排除的工具。
5. `rapidapi_wrapper` 收满 top-k 后停止，并恢复完整 API schema。
6. top-k API 转成 functions，交给 ToolLLaMA/ChatGPT + DFS/CoT。

## 7. `preprocess/`：原始数据如何转训练格式

### `preprocess_toolllama_data.py`

```text
data/answer/G1_answer/*_DFS_woFilter_w2.json
  ↓ 只保留 answer_generation.valid_data=true
answer_generation.train_messages
  ↓ 每个轨迹前缀单独处理
OpenAI messages + function_call
  ↓
system/user/function/assistant conversations
  ↓
data/answer/toolllama_G1_dfs.json
```

转换规则：

- assistant 普通 content → 累积成 `Thought:`。
- assistant function_call.name → `Action:`。
- assistant function_call.arguments → `Action Input:`。
- function content → observation，角色保留为 `function`。
- system 通过 `process_system_message()` 注入 function schema。
- 每个前缀最后一条 assistant function_call 是该样本的监督目标。
- 支持 `CoT@1` 和 `DFS_woFilter_w2`。

脚本只处理单个 answer 目录；官方 G123 train/eval 还包含对 G1/G2/G3 分别切分后再合并的步骤，但本仓库没有提供独立的合并脚本，下载包已直接给出合并后的 train/eval JSON。

### `preprocess_retriever_data.py`

```text
instruction/G1_query.json + test_query_ids/*.json
  ↓ 按 query_id 切分
train.json / test.json
  ↓ api_list 建 corpus；relevant APIs 建正例关系
corpus.tsv + *.query.txt + qrels.*.tsv
```

qrels 只显式保存正例 `label=1`；负例由 MultipleNegativesRankingLoss 的 batch 内其他文档提供。

## 8. `scripts/`：官方实验脚本

| 脚本 | 实验/用途 | 核心配置 |
|---|---|---|
| `preprocess_toolllama_data.sh` | G1 DFSDT answer → SFT JSON | method=`DFS_woFilter_w2` |
| `preprocess_retriever_data.sh` | G1 query → Retriever 数据 | 测试 id 切分 |
| `train_toolllama.sh` | ToolLLaMA 全参训练 | 2 GPU、FSDP、bf16、8192 context、2 epoch |
| `train_toolllama_lora.sh` | ToolLLaMA LoRA | DeepSpeed stage2、bf16、5 epoch |
| `train_retriever.sh` | API Retriever | BERT、batch 32、5 epoch |
| `inference_toolllama_pipeline.sh` | 全参 ToolLLaMA 闭域 | DFSDT w2 |
| `inference_toolllama_lora_pipeline.sh` | LoRA ToolLLaMA 闭域 | LLaMA base + adapter |
| `inference_toolllama_lora_pipeline_open_domain.sh` | LoRA 开放域 | Retriever top-5 + DFSDT |
| `inference_chatgpt_pipeline.sh` | ChatGPT 闭域 | ToolBench key |
| `inference_chatgpt_pipeline_w_rapidapi_key.sh` | ChatGPT 闭域 | 自有 RapidAPI key |
| `inference_davinci_pipeline.sh` | Davinci 闭域 | text-davinci-003 |

## 9. 阅读和运行时应注意的版本问题

- 仓库来自 2023 年代码栈，默认模型、OpenAI 模型名、依赖 API 和远程 ToolBench 服务地址可能已经失效。
- `scripts/*.sh` 是 Bash 脚本；Windows PowerShell 不能原样直接运行。
- `train.py` 的 label offset 有 LLaMA tokenizer 专用硬编码 `-2`；换 tokenizer 要重新验证 masking。
- `preprocess_toolllama_data.py` 不负责 G1/G2/G3 的 train/eval/test 合并。
- `ToolRetriever.retrieving()` 本身返回最多 `10 * top_k`，真正的 top-k 截止发生在 `rapidapi_wrapper`。
- 推理代码中有部分旧式非包限定 import；从不同工作目录启动时可能遇到模块解析问题，应从仓库根目录并设置 `PYTHONPATH=./`。
- 当前环境如需执行 Python，使用 `D:\soft\miniforge3\python.exe`。

