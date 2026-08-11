# ToolBench 项目认知与源码导航

> 本文基于本地仓库 `D:\work\agent-lab\repos\ToolBench` 的 `master` 分支，提交
> `d56fdd89faf8c91fa135090b212bb9057ee5cfc2`（2025-05-21）逐文件核对。
> 它描述的是这份代码实际怎样工作，而不是只复述论文或 README。

## 0. 当前状态与一句话结论

- ToolBench 已作为 Git submodule 放在 `repos/ToolBench/`，上游仓库目前干净。
- 上游只提交了 `data_example/`，没有完整 `data/`；完整数据需从项目 README 给出的网盘下载。
- `data_example/` 可用来理解字段，但不是完整可运行数据环境：它有 10 条 G1/G2/G3 query、15 条 answer、15 个 `api.py` 和 15 个 response schema，却没有推理代码必需的 `<tool>.json` API 元数据。
- 当前默认 Python 环境没有安装 `torch`，所以源码可通过 `compileall` 语法检查，但训练、推理和预处理尚不能直接运行。
- 整个项目的主线是：

```text
API 文档和工具实现
    -> 生成 instruction
    -> ChatGPT + DFS/DFSDT 调工具，生成 answer trajectory
    -> preprocess
       ├─ ToolLLaMA SFT JSON -> 全参训练或 LoRA
       └─ Retriever TSV     -> API Retriever
    -> close-domain / open-domain inference
    -> answer JSON -> ToolEval 转换 -> pass rate / win rate
```

注意：仓库发布了 instruction、answer 和两个 preprocess 脚本，但没有发布最初的 API 抓取、instruction generation 和批量 answer generation 数据生产代码。因此，本仓库能复现的是“从已发布原始/标注数据开始”的下游流程，不是论文数据集从零采集的全流程。

## 1. 项目目录与核心入口

```text
ToolBench/
├─ data_example/                 # 少量字段样例，不是完整实验数据
├─ preprocess/
│  ├─ preprocess_toolllama_data.py
│  └─ preprocess_retriever_data.py
├─ scripts/                      # 预处理、训练、推理的官方 shell 配方
├─ ds_configs/                   # DeepSpeed stage 2/3 配置
└─ toolbench/
   ├─ train/                     # ToolLLaMA SFT
   ├─ inference/                 # agent loop、工具环境、搜索算法和模型适配
   ├─ retrieval/                 # API Retriever 训练和离线评测
   ├─ tooleval/                  # 自动 pass-rate / preference 评测
   ├─ model/                     # HF 模型适配、delta 应用/生成
   ├─ tool_conversation.py       # SFT prompt 模板
   └─ utils.py                   # 名称标准化、system prompt、检索文档格式等
```

入口速查：

| 目的 | 直接入口 | 真正核心实现 |
| --- | --- | --- |
| 全参数 SFT | `toolbench/train/train_mem.py` | `toolbench/train/train.py::train` |
| LoRA SFT | `toolbench/train/train_lora.py` | 同文件 `train`，复用 `train.py` 的 data module |
| 封闭域推理 | `toolbench/inference/qa_pipeline.py` | `Downstream_tasks/rapidapi.py::pipeline_runner` |
| 开放域推理 | `toolbench/inference/qa_pipeline_open_domain.py` | 同上，额外构建 `ToolRetriever` |
| 工具执行 | `Downstream_tasks/rapidapi.py::rapidapi_wrapper.step` | `inference/server.py::get_rapidapi_response` 或远端 ToolBench 服务 |
| CoT/ReAct 单链 | `Algorithms/single_chain.py` | `single_chain.do_chain` |
| DFS/DFSDT | `Algorithms/DFS.py` | `DFS_tree_search.DFS` |
| Retriever 训练 | `toolbench/retrieval/train.py` | SentenceTransformers + `MultipleNegativesRankingLoss` |
| Retriever 调用 | `inference/LLM/retriever.py` | `ToolRetriever.retrieving` |
| ToolEval 预处理 | `tooleval/convert_to_answer_format.py` | 把搜索树/消息变成统一 execution graph |
| 通过率 | `tooleval/eval_pass_rate.py` | `ReinforceToolLearningEvaluator` |
| 胜率 | `tooleval/eval_preference.py` | `BaseEvaluator.annotate_preference` + RTL evaluator |

## 2. `data/`：原始、训练、评测数据怎样组织

### 2.1 完整数据包约定

README 描述的完整目录是：

```text
data/
├─ instruction/                  # 原始生成指令；G1/G2/G3 query
├─ answer/                       # 带搜索树和工具 observation 的解题轨迹
├─ toolenv/
│  ├─ tools/<category>/
│  │  ├─ <tool>.json             # tool/API 文档，推理时构造 function schema
│  │  └─ <tool>/api.py           # 本地执行器；自带 RapidAPI key 模式使用
│  └─ response_examples/         # API 返回 schema，用于 filter observation
├─ retrieval/                    # Retriever 的 corpus/query/qrels 文件
├─ test_instruction/             # 六个评测子集的 query
├─ test_query_ids/               # 六个 ToolEval 子集的固定 query id
├─ retrieval_test_query_ids/     # Retriever 的固定测试 id
├─ toolllama_G123_dfs_train.json # 已预处理 SFT train
└─ toolllama_G123_dfs_eval.json  # 已预处理 SFT eval
```

ToolEval 还使用 `reproduction_data/`。README 的目录树把它画成 `data/` 的同级目录，但命令和说明又使用 `data/reproduction_data/`；实际运行应以脚本参数为准，建议统一放在 `data/reproduction_data/`。

### 2.2 三种任务难度

- G1：single-tool，query 所需 API 来自一个工具。
- G2：multi-tool in one category，涉及同一类别内多个工具。
- G3：multi-tool in collection，跨类别/全集的多工具组合。

ToolEval 不是只评 G1/G2/G3 三份，而是六个测试子集：

```text
G1_instruction, G1_category, G1_tool,
G2_instruction, G2_category,
G3_instruction
```

每个子集在 `test_query_ids/` 中固定抽取 200 个实例。

### 2.3 原始 instruction 格式

`instruction/G*_query.json` 是对象数组，每项核心字段如下：

```json
{
  "query_id": 1,
  "query": "natural-language task",
  "api_list": [
    {
      "category_name": "Logistics",
      "tool_name": "SQUAKE",
      "api_name": "Projects",
      "api_description": "...",
      "required_parameters": [],
      "optional_parameters": [],
      "method": "GET"
    }
  ],
  "relevant APIs": [["SQUAKE", "Projects"]]
}
```

- `api_list` 是候选/可用 API 集合，也是 Retriever corpus 的来源。
- `relevant APIs` 是 query 的正相关 API 标注，用于 Retriever qrels。
- 封闭域推理输入保留 `api_list`；开放域推理输入只需要 `query` 和 `query_id`，由 Retriever 补 API。

### 2.4 原始 answer / trajectory 格式

`answer/G*_answer/<query_id>_<method>.json` 由推理 pipeline 的 `chain.to_json()` 产生，顶层包含：

```text
win
tree                   # Thought / Action / Action Input 搜索树
forward_args           # depth、beam width、query budget、filter 等
compare_candidates     # 有 LLM ranking 时的比较信息
answer_generation
```

`answer_generation` 是预处理和 ToolEval 真正读取的部分：

```text
valid_data             # 是否找到可用 terminal node
query_count            # 模型调用次数
total_tokens
final_answer           # 实际是 Finish 的 arguments JSON 字符串
function               # 可用 function schema，包括 Finish
chain                  # 最终路径
train_messages         # 每一步的 OpenAI message 前缀样本
query
finish_type            # give_answer / give_up（部分版本存在）
```

一次成功路径会保留每步的 `system/user/assistant/function` 消息、模型 Thought、function call、参数和真实 observation。这既是 ToolLLaMA 的行为克隆来源，也是 ToolEval 判断轨迹质量的依据。

### 2.5 训练数据和评测数据不是一回事

| 层次 | 文件 | 消费者 |
| --- | --- | --- |
| 原始 query | `instruction/G*_query.json` | Retriever preprocess、数据分析 |
| 解题轨迹 | `answer/G*_answer/*.json` | ToolLLaMA preprocess、ToolEval convert |
| SFT train/eval | `toolllama_G123_dfs_{train,eval}.json` | HF Trainer |
| Retriever train/test | `retrieval/<dataset>/*.tsv, *.txt` | SentenceTransformers |
| 推理测试 query | `test_instruction/*.json` | inference pipeline |
| 固定评测 id | `test_query_ids/*.json` | ToolEval |
| 原始模型预测 | `reproduction_data/model_predictions/<model>/<test_set>/*.json` | convert script |
| 统一评测输入 | `model_predictions_converted/<model>/<test_set>.json` | pass/preference evaluator |

## 3. `toolbench/train/`：SFT 入口与训练格式

### 3.1 入口关系

```text
train_mem.py
  -> 先 monkey patch LLaMA FlashAttention
  -> import train.py::train
  -> HF AutoTokenizer / AutoModelForCausalLM / Trainer

train_lora.py
  -> FlashAttention + RoPE condense
  -> PEFT LoraConfig(q_proj, v_proj)
  -> 复用 train.py::make_supervised_data_module
  -> HF Trainer + DeepSpeed
```

- `train.py` 本身也可直接运行，但官方全参脚本选择 `train_mem.py` 以降低显存占用。
- 当 `model_max_length > source_model_max_length` 时，代码用 condense RoPE 把 LLaMA 2048 上下文扩到 8192；官方比例是 4。
- tokenizer 使用 right padding，并把 `unk_token` 当 `pad_token`。
- 若 output 目录已有 `checkpoint-*`，训练会自动 resume。

### 3.2 SFT JSON 精确格式

文件顶层是数组；每个样本是：

```json
{
  "id": "Step 4: <original query>",
  "conversations": [
    {"from": "system", "value": "system prompt + all function schemas"},
    {"from": "user", "value": "query"},
    {
      "from": "assistant",
      "value": "\nThought: ...\nAction: api_for_tool\nAction Input: {...}"
    },
    {"from": "function", "value": "tool observation"},
    {
      "from": "assistant",
      "value": "\nThought: ...\nAction: Finish\nAction Input: {...}"
    }
  ]
}
```

这里有两个容易误解的关键点：

1. 一条完整 trajectory 不只生成一个 SFT 样本。`preprocess_toolllama_data.py` 遍历 `train_messages`，把每一个“历史前缀 -> 下一次 function call”变成独立样本。
2. 虽然样本里有多轮历史，`train.py::preprocess` 会把 system、user、function 和历史 assistant 全部 mask 为 `IGNORE_TOKEN_ID`，只对最后一个 `Assistant:` 段计算 causal-LM loss。

`tool-llama-single-round` 模板的四种角色是 `System/User/Function/Assistant`，消息之间以换行分隔，最后一个 assistant 后用 `</s>`。模型学习输出的不是原生 JSON function call，而是合并后的 ReAct 文本：

```text
Thought: ...
Action: <function name>
Action Input: <JSON arguments>
```

推理时 `react_parser()` 再把这段文本还原成类 OpenAI `function_call` 消息。

### 3.3 官方训练参数

- 全参：2 × A100 80GB，FSDP full shard，2 epochs，bf16，batch/device=2，grad accumulation=8。
- LoRA：DeepSpeed stage 2，5 epochs，默认 `r=8, alpha=16, dropout=0.05`，只挂 `q_proj/v_proj`。
- 两者默认 max length 8192、gradient checkpointing、lazy preprocess。

## 4. `toolbench/inference/`：如何逐步调用工具

### 4.1 封闭域与开放域

- 封闭域入口 `qa_pipeline.py`：query 自带 `api_list`。
- 开放域入口 `qa_pipeline_open_domain.py`：query 不带 API，先用 `ToolRetriever` 取 top-k API，再走完全相同的 agent loop。

### 4.2 一次任务的真实调用链

```text
qa_pipeline[_open_domain].py
  -> pipeline_runner(args)
  -> generate_task_list()
     ├─ 读取 query JSON
     ├─ 加载 ToolLLaMA / LoRA，或记录 OpenAI backbone 名称
     └─ 校验 tool whitelist
  -> run_single_task()
     -> rapidapi_wrapper(...)
        ├─ [open domain] Retriever 补 api_list
        ├─ 从 <category>/<tool>.json 找到 API 文档
        ├─ api_json_to_openai_json() 生成 function schema
        └─ 添加强制终止函数 Finish
     -> method_converter()
        ├─ CoT@n -> single_chain
        └─ DFS... -> DFS_tree_search
     -> LLM.parse(functions)
     -> 得到 content + function_call(name, arguments)
     -> env.step(action_name, action_input)
     -> observation 作为 role=function 消息回灌
     -> 重复直到 Finish / prune / depth limit / query budget
     -> chain.to_json() 写 <query_id>_<method>.json
```

### 4.3 模型怎样决定并调用工具

`rapidapi_wrapper` 将 API 名标准化为：

```text
<api_name>_for_<standardized_tool_name>
```

同时把 required/optional parameters 转成 OpenAI function schema。每轮：

1. system prompt 告诉模型必须输出 Thought、Action、Action Input，并最终调用 `Finish`。
2. ToolLLaMA 生成 ReAct 文本；`react_parser` 拆成 thought/action/arguments。ChatGPT backbone 则直接返回 function call。
3. 搜索算法创建 `Thought -> Action -> Action Input` 三类 tree node。
4. `rapidapi_wrapper.step` 查找函数、执行、截断 observation，并返回 observation + status code。
5. 搜索算法把 assistant function call 和 role=function 的 observation 加回 messages，再让模型决策下一步。

`Finish` 也是一个 function：

```json
{"return_type": "give_answer", "final_answer": "..."}
```

或：

```json
{"return_type": "give_up_and_restart"}
```

前者 status=3 并终止成功；后者 status=4 并剪枝/回溯。

### 4.4 工具在哪里真正执行

有三条路径：

- 默认：POST 到代码中写死的旧 ToolBench RapidAPI 服务地址，带 `toolbench_key`。
- `--use_rapidapi_key`：本地 `server.py` 动态 import `data.toolenv.tools.<category>.<tool>.api`，调用 Python 函数并注入用户 RapidAPI key。
- `--api_customization`：同样本地执行，但不注入 RapidAPI key。

本地执行使用字符串拼接后的 `exec`/`eval`；这是研究原型，不应直接作为不可信生产输入的执行沙箱。

observation status 主要包括：正常 0、幻觉函数名 1、参数错误 2、完成 3、主动放弃 4、超时/404/未订阅/未授权/限流/服务错误等 5–12。

### 4.5 CoT 与 DFS/DFSDT

- `CoT@n`：最多独立尝试 n 条 single chain，某条成功即停。
- `DFS_wN`：每个分叉生成 N 个候选，用额外 LLM ranking 排序后 DFS。
- `DFS_woFilter_wN`：不做 ranking，候选生成后立即深搜，即论文/代码所称 DFSDT。
- 官方默认 `DFS_woFilter_w2`：beam width=2、max depth=12、最多 200 次模型 query，找到 1 个 `give_answer` terminal node 即停。

## 5. `toolbench/retrieval/`：API Retriever 如何训练与调用

### 5.1 数据预处理

`preprocess_retriever_data.py` 输入：

- `query_file`：带 `api_list` 和 `relevant APIs` 的 G1/G2/G3 query。
- `index_file`：固定测试 query id。
- `dataset_name`：只用于命名逻辑，当前脚本最终输出名固定。
- `output_dir`。

输出：

```text
train.json / test.json           # 原 query 切分
corpus.tsv                       # docid, document_content(JSON)
train.query.txt / test.query.txt # qid<TAB>query
qrels.train.tsv / qrels.test.tsv # qid, 0, docid, 1
```

每个 API 是一篇 document，正例由 `[tool_name, api_name] in relevant APIs` 决定。用于编码的 document text 由 `process_retrieval_ducoment()` 拼成：category、tool、API name、description、required params、optional params 和 return schema。

### 5.2 模型与损失

`retrieval/train.py`：

```text
bert-base-uncased Transformer
  + SentenceTransformers Pooling
  + MultipleNegativesRankingLoss
```

一个 batch 内其他 query 的正 API 自动充当 in-batch negatives；预处理没有显式写负样本。每个 epoch 用 cosine retrieval 评估 NDCG@1、NDCG@3、NDCG@5，evaluator 返回三者最小值供 SentenceTransformers 选择 checkpoint。

### 5.3 运行时调用

`ToolRetriever` 初始化时：

1. 读取整个 `corpus.tsv`。
2. 加载 SentenceTransformer checkpoint。
3. 一次性把整个 corpus 编码到 tensor 并常驻内存/GPU。

`retrieving(query, top_k=5)` 对 query 编码，用 cosine semantic search 先取 `10 * top_k`，再映射为：

```json
{"category": "...", "tool_name": "...", "api_name": "..."}
```

注意：`ToolRetriever.retrieving()` 自身没有在结果列表达到 `top_k` 时 break，因此直接调用时最多会返回 `10 * top_k`；开放域 pipeline 的 `rapidapi_wrapper.retrieve_rapidapi_tools()` 会再次筛掉本地不存在的 API，并在有效项达到 top-k 时停止。

## 6. `toolbench/tooleval/`：自动评测怎样实现

### 6.1 第一步：统一模型输出

`convert_to_answer_format.py` 把每个 `<query_id>_<method>.json` 转为：

```json
{
  "query": "...",
  "available_tools": [],
  "answer": {
    "method": "DFS_woFilter_w2",
    "total_steps": 3,
    "final_answer": "...",
    "answer_details": {"role": "...", "message": "...", "next": []}
  }
}
```

有效路径从最后一条 `train_messages` 构造 execution graph；无效 CoT/DFS 输出则从 chain/tree 尽量恢复图结构。后续两个 evaluator 都只读这个统一格式。

### 6.2 Pass rate

`eval_pass_rate.py` 对每个 query 重复评判 `evaluate_times` 次，并发调用 evaluator：

1. 静态检查轨迹是否调用了未提供的 function（hallucination）。
2. 没有以 `Finish` 结束，直接 failed。
3. LLM judge 判断 final answer 是 `Solved / Unsolved / Unsure`。
4. LLM judge 判断任务在给定工具下是 `Solvable / Unsolvable / Unsure`。
5. `is_passed` 使用状态真值表：
   - answer solved -> pass；
   - answer unsolved + task solvable -> fail；
   - answer unsolved + task unsolvable -> pass；
   - unsure 情况可能返回 unsure，再随机落成 pass/fail。
6. 多次结果多数投票，写 JSON 明细和 TSV CSV。

因此 ToolEval 的“通过”不是简单字符串匹配，也不是仅看 API 调用成功；它允许模型正确识别不可解任务。

### 6.3 Win rate / preference

`eval_preference.py` 比较 reference model 与 candidate model：

- 如果已有 pass-rate 且一方 pass、一方 fail，直接让 pass 方胜，不再调用 preference judge。
- 否则 `annotate_preference` 会随机打乱两个 answer，再由 evaluator 根据 solved status、步数、最终答案质量和探索质量选择更好答案。
- 重复多轮后多数票产生 candidate win / lose / tie rate。

### 6.4 Evaluator 插件结构

```text
evaluators/<evaluator_name>/
├─ config.yaml       # registered class、模型、functions、template
└─ template.txt

evaluators/registered_cls/
├─ base.py           # BaseEvaluator / ToolEvalEvaluator
├─ tooleval.py       # OpenAI evaluator / normalized evaluator
├─ rtl.py            # 新版 pass/preference 状态逻辑
└─ utils.py          # registry + OpenAI key pool
```

默认 evaluator 是 `tooleval_gpt-3.5-turbo_default`，注册类为 `ReinforceToolLearningEvaluator`。key pool 可通过 `API_POOL_FILE` 环境变量覆盖 config 中的 `apis_json`。

### 6.5 当前实现的复现风险

- 代码使用旧版 `openai.ChatCompletion.create`、旧模型名 `gpt-3.5-turbo-16k` / `-0613`，不能假设在 2026 年仍可直接调用。
- `eval_preference.py:209` 的条件 `i % 2 == 0 or i >= 0` 永远为真，本意的候选顺序交替实际上不会发生。
- `eval_pass_rate.py` 对票数相等的单条 CSV label 随机选 pass/fail，但最终汇总用 `failed <= passed`，把 tie 固定算 pass；两处语义不一致。
- ToolEval 是 LLM-as-judge，输出依赖旧模型、prompt、temperature、key pool 和随机投票；复现实验需固定这些条件。

## 7. `preprocess/`：原始数据怎样变成训练格式

### 7.1 ToolLLaMA preprocess

```text
answer/G*_answer/*.json
  -> 只保留文件名含指定 method 的文件
  -> 只保留 answer_generation.valid_data
  -> 对每条 train_messages 前缀：
       system: 加入 ReAct 输出格式和全部 function schema
       user/function: 原样加入 history
       assistant content: 累积为 Thought
       assistant function_call: 转为 Action + Action Input
       最后一个 assistant call: 作为该 SFT 样本的 target
  -> [{id, conversations}, ...]
```

它不负责 train/eval/test 切分，也不负责合并 G1/G2/G3。官方发布的 `toolllama_G123_dfs_train/eval.json` 已提前完成这些操作；如果自己 preprocess，需要另写合并和切分步骤。

### 7.2 Retriever preprocess

```text
instruction/G*_query.json + fixed test ids
  -> train/test query split
  -> 每个 api_list item 建 corpus doc
  -> relevant APIs 建 positive qrels
  -> corpus/query/qrels 五个检索训练文件
```

## 8. `scripts/`：官方提供的实验脚本

| 脚本 | 用途 | 主要产物/说明 |
| --- | --- | --- |
| `preprocess_toolllama_data.sh` | G1 answer -> SFT JSON | 示例只处理 G1，未合并/切分 G123 |
| `preprocess_retriever_data.sh` | G1 query -> Retriever 文件 | 使用 G1 instruction test ids |
| `train_toolllama.sh` | 2 卡 FSDP 全参 SFT | 输出 `toolllama/` |
| `train_toolllama_lora.sh` | DeepSpeed stage 2 LoRA | 输出 `toolllama_lora/` |
| `train_retriever.sh` | 训练 BERT API Retriever | 输出目录下再按时间戳建 checkpoint |
| `inference_toolllama_pipeline.sh` | 全参 ToolLLaMA 封闭域 DFS | query 自带 API |
| `inference_toolllama_lora_pipeline.sh` | LoRA 封闭域 DFS | base + adapter |
| `inference_toolllama_lora_pipeline_open_domain.sh` | Retriever + LoRA 开放域 DFS | query 不带 API |
| `inference_chatgpt_pipeline.sh` | ChatGPT function calling + ToolBench service | 需要 OpenAI/ToolBench key |
| `inference_chatgpt_pipeline_w_rapidapi_key.sh` | ChatGPT + 本地 tool code + 自有 RapidAPI key | 使用 `--use_rapidapi_key` |
| `inference_davinci_pipeline.sh` | text-davinci-003 + DFS | 历史实验入口，当前基本不可复现 |

`toolbench/tooleval/` 另有三份评测脚本：

| 脚本 | 用途 |
| --- | --- |
| `run_convert_answer.sh` | 六个测试集的 raw prediction -> converted JSON |
| `run_pass_rate.sh` | 单模型 pass rate |
| `run_preference.sh` | reference vs candidate win rate |

所有官方脚本都是 Bash，并假设从仓库根目录（评测脚本则假设从 `toolbench/tooleval/`）运行、`PYTHONPATH=./`、Linux/CUDA 和旧依赖版本。当前主机有 Git Bash，但默认 Python 依赖未安装。

## 9. 推荐阅读顺序

1. `data_example/instruction/inference_query_demo.json`：先看封闭域输入。
2. `toolbench/inference/Downstream_tasks/rapidapi.py`：理解工具 schema、环境和总 pipeline。
3. `toolbench/inference/Algorithms/single_chain.py`，再读 `DFS.py`：理解 agent loop 与 DFSDT。
4. `data_example/answer/G1_answer/*.json`：把树节点、messages、observation 和代码对应起来。
5. `preprocess/preprocess_toolllama_data.py` + `train/train.py`：理解轨迹如何变成监督 token。
6. `preprocess/preprocess_retriever_data.py` + `retrieval/train.py` + `inference/LLM/retriever.py`。
7. `tooleval/convert_to_answer_format.py` + `eval_pass_rate.py` + `eval_preference.py` + `evaluators/registered_cls/rtl.py`。
8. 最后看 `scripts/`，把参数与上述入口对应起来。

## 10. 下一步可执行清单

若目标是实际复现，而不只是阅读：

1. 下载最新版 `data.zip`，解压为 `repos/ToolBench/data/`，不要使用旧的 `data_0801.zip`。
2. 先只安装并验证 Retriever 或 ToolEval 的最小环境；训练环境严格按锁定版本单独建虚拟环境，不要污染工作区通用 Python。
3. 先用 `data_example` 补齐/映射 metadata 后跑一个 query 的封闭域 mock，验证消息循环；不要先启动 7B 全参训练。
4. 如果要使用现代 OpenAI API，需先迁移 `ChatCompletion`、模型名和 function-calling 字段，再建立新的 judge 基线；不能把迁移后的结果与论文旧榜单直接视为同分布。
5. 若要严谨复现 ToolEval，先修 preference 顺序条件和 pass-rate tie 语义，并记录修改后的 evaluator 版本。
