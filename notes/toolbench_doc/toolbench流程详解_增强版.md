# ToolBench 端到端流程与 Agent 抽象映射（增强版）

ToolBench 需要区分三件不同的事情：

1. 离线数据与模型训练；
2. 在线 Agent 推理循环；
3. 推理结束后的 ToolEval。

真正的 Agent 抽象主要发生在第 2 部分。instruction → answer → SFT 是训练管线，不是 Agent 每一步的运行过程。

## 一、ToolBench 完整端到端架构

![1786439830716](image/toolbench流程详解/1786439830716.png)

### 核心关系：
- Retriever 负责缩小可选 API 范围；
- ToolLLaMA 负责提出下一步动作；
- DFS/DFSDT 负责探索和选择动作分支；
- rapidapi_wrapper 是 Agent 环境适配层；
- server.py 或远程 RapidAPI 服务是真正的工具执行器；
- ToolEval 是 episode 结束后的外部评测器。

### 项目目录结构概览：
```
ToolBench/
├── data/                  # 完整数据包（需另行下载）
├── data_example/          # 小规模结构样例
├── preprocess/           # 原始标注转换
├── toolbench/
│   ├── train/           # ToolLLaMA SFT训练
│   ├── inference/       # 闭域/开放域推理
│   ├── retrieval/       # API Retriever训练与调用
│   ├── model/          # 模型/对话模板适配
│   └── tooleval/       # 答案格式转换与评估
├── scripts/             # 官方实验脚本
└── ds_configs/         # DeepSpeed配置
```

## 二、抽象流程到 ToolBench 项目的映射

抽象流程：
```text
Instruction
    ↓
Candidate APIs / Tool Schema
    ↓
Model Policy
    ↓
Tool Call Action
    ↓
Tool Executor
    ↓
Observation
    ↓
Next Action or Final Answer
    ↓
ToolEval / Metrics
```

在 ToolBench 中具体对应为：

![1786440056526](image/toolbench流程详解/1786440056526.png)

### 具体代码入口：
| 任务 | 实际入口文件 | 核心功能 |
|------|-------------|----------|
| 全参SFT训练 | `toolbench/train/train_mem.py` | FlashAttention + 全模型训练 |
| LoRA训练 | `toolbench/train/train_lora.py` | PEFT adapter训练 |
| 闭域推理 | `toolbench/inference/qa_pipeline.py` | 带api_list的推理 |
| 开放域推理 | `toolbench/inference/qa_pipeline_open_domain.py` | 带Retriever的推理 |
| Retriever训练 | `toolbench/retrieval/train.py` | SentenceTransformer训练 |
| ToolEval评估 | `toolbench/tooleval/*.py` | Pass Rate/Preference评估 |

## 三、ToolBench 中 Agent 的六个核心抽象

### 1. State：状态是什么

ToolBench 没有单独定义名为 AgentState 的类。状态分散在 tree_node 和 rapidapi_wrapper 中，可以抽象为：

$$
s_t=(q,\mathcal A,M_t,E_t,N_t)
$$

| 状态组成 | ToolBench 中的实际对象 |
|----------|----------------------|
| $q$：当前任务 | rapidapi_wrapper.input_description |
| $\mathcal A$：当前可选动作集合 | rapidapi_wrapper.functions |
| $M_t$：对话历史 | tree_node.messages |
| $E_t$：环境状态 | tree_node.io_state |
| $N_t$：搜索状态 | tree_node 的 depth、father、children、pruned、is_terminal |

**代码位置**：`toolbench/inference/Tree/Tree.py::tree_node`

一个节点保存：
```text
node_type
description
messages
io_state
observation
observation_code
father / children
depth
pruned
is_terminal
prior_score
```

因此 ToolBench 中的状态不是单纯的聊天记录，而是：
```text
对话历史
+ 可调用 API
+ 工具环境快照
+ 当前搜索树位置
+ 当前分支是否失败或结束
```

DFS 每创建一个分支都会复制环境和消息：
```python
child_io_state = deepcopy(parent.io_state)
child_messages = deepcopy(parent.messages)
```

所以每条搜索分支都有自己的状态副本。

### 2. Action：动作是什么

抽象动作是：
$$
a_t=(\text{function name},\text{arguments})
$$

ToolLLaMA 生成的原始文本：
```text
Thought: I should search for the project.
Action: projects_for_squake
Action Input: {}
```

随后 react_parser() 解析为：
```json
{
  "role": "assistant",
  "content": "I should search for the project.",
  "function_call": {
    "name": "projects_for_squake",
    "arguments": "{}"
  }
}
```

搜索树把一次动作拆成三个节点：
```text
Thought
  ↓
Action
  ↓
Action Input
```

真正传给环境的是：
```python
env.step(
    action_name=function_name,
    action_input=function_arguments
)
```

需要区分：
- Thought 是内部推理文本；
- Action 是 API/function 名；
- Action Input 是 API 参数；
- 真正影响环境的是 Action 和 Action Input。

### 3. Observation：环境反馈是什么

环境执行动作后返回：
$$
o_{t+1}=(\text{response},\text{status code})
$$

实际调用：
```python
observation, status = child_io_state.step(
    action_name=...,
    action_input=...
)
```

例如：
```json
{
  "error": "",
  "response": "API 返回的数据"
}
```

结果保存在：
```text
tree_node.observation
tree_node.observation_code
```

同时追加到模型消息：
```json
{
  "role": "function",
  "name": "projects_for_squake",
  "content": "{\"error\":\"\",\"response\":\"...\"}"
}
```

因此下一次模型调用看到：
```text
用户问题
+ 之前采取的 Action
+ API 返回的 Observation
```

模型再根据这些信息决定下一步。

### 4. Policy：策略是什么

ToolBench 的 Policy 不是一个对象，而是两层策略组合。

#### 第一层：模型动作策略
$$
\pi_\theta(a_t\mid s_t)
$$

对应：
```text
toolbench/inference/LLM/tool_llama_model.py
└── ToolLLaMA.parse()
```

其他实现：
```text
ChatGPTFunction.parse()
Davinci.parse()
ToolLLaMALoRA.parse()
```

它负责：给定当前消息状态和 functions，生成一个候选 Action。

#### 第二层：搜索/规划策略

对应：
```text
Algorithms/single_chain.py
Algorithms/DFS.py
```

它负责：决定生成多少候选、先探索哪个候选、何时回溯以及何时停止。

因此 Policy 应当拆为：
```text
Policy
├── Action proposal policy：ToolLLaMA.parse()
└── Planning/search policy：DFS/DFSDT
```

ToolLLaMA 和 DFSDT 不能画成同一模块：
- ToolLLaMA 是下一步动作生成器；
- DFSDT 决定如何多次调用动作生成器并搜索分支。

### 5. Tool Executor：工具执行器是什么

环境执行入口：
```text
rapidapi_wrapper.step()
└── rapidapi_wrapper._step()
```

它先把模型生成的 function 名映射回：
```text
category
tool_name
api_name
tool_input
```

然后有两条执行路径。

ToolBench 远程后端：
```python
requests.post(self.service_url, ...)
```

自有 RapidAPI 或自定义 API：
```text
toolbench/inference/server.py
└── get_rapidapi_response()
```

因此：
- rapidapi_wrapper 更像 Agent 环境适配层；
- server.py / RapidAPI 服务才是真正的工具执行器。

### 6. Evaluator 和 Feedback：分别是什么

#### 即时 Feedback
Agent 每执行一个动作立刻得到：
```text
observation
status code
```

这是环境反馈：
```text
Action
  ↓
API response
  ↓
Observation
  ↓
更新 State
```

状态码同时影响控制流：

| 状态码 | 含义 | 对搜索的影响 |
|-------|------|-------------|
| 0 | 正常 observation | 继续 |
| 1 | 函数名幻觉 | 写回错误，允许模型修正 |
| 2 | 参数错误 | 写回错误 |
| 3 | Finish/give_answer | terminal |
| 4 | give_up_and_restart | prune、回溯 |
| 5–12 | 超时、授权、限流等 | 写回错误或结束分支 |

#### 在线终止检查器
```text
rapidapi_wrapper.check_success()
```

它只检查模型是否成功调用：
```text
Finish → give_answer
```

它不能判断答案事实是否正确。因此它是 Terminal checker，不是严格意义上的答案质量 Evaluator。

#### 离线 Evaluator
真正的答案质量评测位于：
```text
toolbench/tooleval/
```

主要包括：
```text
eval_pass_rate.py
eval_preference.py
```

评测对象是完整 episode 的最终结果，而不是每一步动作。

#### Retriever Evaluator
Retriever 有独立 evaluator：
```text
toolbench/retrieval/api_evaluator.py
```

它使用 NDCG 评估召回 API 是否正确，与 ToolEval 不是同一个 evaluator。

## 四、数据组织与预处理流程

### 数据目录结构
```
data/
├── instruction/                  # 原始生成指令，G1/G2/G3
├── answer/                       # CoT/DFSDT 解路径、工具返回、搜索树
├── toolenv/                      # API 文档、Python 实现、响应样例
├── retrieval/                    # Retriever 预处理结果
├── test_instruction/             # 各测试集的 query
├── test_query_ids/               # SFT/推理测试划分 id
├── retrieval_test_query_ids/     # Retriever 测试划分 id
├── toolllama_G123_dfs_train.json # 可直接用于 SFT 的训练集
└── toolllama_G123_dfs_eval.json  # 可直接用于 SFT 的验证集
```

### G1/G2/G3 语义：
- **G1**：单工具场景
- **G2**：同类别内多工具场景  
- **G3**：跨类别/集合多工具场景

### SFT 训练数据格式
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

### 预处理脚本
- `preprocess_toolllama_data.py`：将原始标注转换成 SFT 格式
- `preprocess_retriever_data.py`：生成 Retriever 训练数据

## 五、ToolBench 是否是标准强化学习 Agent

**不是**。

虽然可以用 MDP/POMDP 抽象：
```text
State → Policy → Action → Environment → Observation → State
```

但当前代码没有真正的 RL reward 和在线策略更新。

证据是：
```python
rapidapi_wrapper.get_score()
```
固定返回：
```python
0.0
```

ToolBench 中存在几种不同反馈：

| 反馈类型 | 来源 | 是否更新模型 |
|----------|------|-------------|
| 工具 observation | API 执行结果 | 否，只更新当前上下文 |
| DFS 分支反馈 | terminal、pruned、status code、候选排名 | 否，只控制搜索 |
| ToolEval 指标 | pass rate、preference | 否，只做离线评测 |
| SFT supervision | train_messages 中的下一步正确 Action | 是，离线训练模型 |

更准确的定义是：

> ToolBench 是"经过 SFT 训练的工具调用策略 + 搜索规划器 + API 环境 + 离线评测器"，而不是基于 reward 在线学习的 RL Agent。

## 六、用一次具体 Episode 串起来

假设 query 是：
```text
查询 SQUAKE 平台项目列表。
```

### 第 0 步：构造动作空间

闭域时读取：
```json
{
  "query": "查询 SQUAKE 平台项目列表",
  "api_list": [
    {
      "category_name": "Logistics",
      "tool_name": "SQUAKE",
      "api_name": "Projects"
    }
  ]
}
```

转换后：
```text
functions = [
  projects_for_squake,
  Finish
]
```

初始状态：
```text
s0:
query = 查询 SQUAKE 项目
functions = [projects_for_squake, Finish]
messages = [system, user]
search_node = root
success = 0
```

### 第 1 步：Policy 生成动作

模型输出：
```text
Thought: I need to retrieve the projects.
Action: projects_for_squake
Action Input: {}
```

因此：
```text
a0 = (projects_for_squake, {})
```

### 第 2 步：Executor 执行动作
```text
rapidapi_wrapper.step()
→ _step()
→ RapidAPI
```

返回：
```json
{
  "error": "",
  "response": "[project1, project2]"
}
```

所以：
```text
o1 = API response
```

### 第 3 步：状态转移

消息更新为：
```text
system
user
assistant:
  function_call projects_for_squake {}
function:
  [project1, project2]
```

得到：
```text
s1 = s0 + action + observation
```

### 第 4 步：Policy 再决策

模型看到项目列表后输出：
```text
Thought: I have enough information.
Action: Finish
Action Input:
{
  "return_type": "give_answer",
  "final_answer": "项目包括 project1 和 project2。"
}
```

### 第 5 步：终止
```text
rapidapi_wrapper._step(Finish)
→ success = 1
→ status = 3
→ tree_node.is_terminal = true
```

### 第 6 步：保存和评测

保存：
```text
<query_id>_DFS_woFilter_w2.json
```

结果包括：
```text
搜索树
所有 Action/Observation
最终答案
train_messages
functions
query_count
token 数
```

ToolEval 随后判断：
```text
任务是否完成
最终答案是否优于另一候选答案
```

## 七、最终抽象

```text
Instruction
    ↓
Action-space construction
    ├─ 闭域：query.api_list
    └─ 开放域：API Retriever
    ↓
State = query + functions + message history + environment + search node
    ↓
Policy = LLM action proposal + DFS planning
    ↓
Action = function name + arguments
    ↓
Executor = rapidapi_wrapper + Tool server
    ↓
Feedback = observation + status code
    ↓
State transition
    ├─ 继续生成下一 Action
    ├─ 失败剪枝并回溯
    └─ Finish 得到 Final Answer
    ↓
Episode JSON
    ↓
ToolEval / Retriever NDCG
```

## 八、实践提示与注意事项

### 关键入口文件
- **训练**：`toolbench/train/train_mem.py` (全参) / `train_lora.py` (LoRA)
- **推理**：`toolbench/inference/qa_pipeline.py` (闭域) / `qa_pipeline_open_domain.py` (开放域)
- **评估**：`toolbench/tooleval/eval_pass_rate.py` / `eval_preference.py`

### 搜索方法对比
| method | 实现 | 行为 |
|--------|------|------|
| `CoT@N` | `Algorithms/single_chain.py` | 从头运行最多 N 条独立链，找到答案即停 |
| `DFS_woFilter_w2` | `Algorithms/DFS.py` | DFSDT；每层宽度 2，生成候选后立即深挖 |
| `DFS_*_wN` | `Algorithms/DFS.py` | 每层 N 个候选；启用 filter 时先由 LLM 比较排序 |

### 版本与兼容性注意
1. 代码基于 2023 年代码栈，部分依赖可能已过时
2. `train.py` 的 label offset 有 LLaMA tokenizer 专用硬编码 `-2`
3. 推理代码中有部分旧式非包限定 import，需注意模块解析
4. Retriever 训练使用 MultipleNegativesRankingLoss，batch 内其他文档作为负例

### 数据流关键点
1. **SFT 训练**：只对当前样本最后一个 assistant 回复计算 causal LM loss
2. **Retriever**：每个 API document 由多个字段拼接成字符串进行编码
3. **推理结果**：保存为 JSON 格式，包含搜索树、轨迹、最终答案等完整信息
4. **评估指标**：Pass Rate (任务是否完成) + Preference (两个答案谁更好)

---

*本增强版整合了 ToolBench 的理论抽象与工程实现，既保留了 Agent 的核心概念，也提供了实际使用的参考信息。*