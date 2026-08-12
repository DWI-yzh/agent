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

ToolBench 没有单独定义名为 `AgentState` 的类。完整状态分散在三个对象中：

1. `rapidapi_wrapper`：任务、候选工具和工具执行环境；
2. `tree_node`：当前搜索分支的对话历史、环境副本和节点控制信息；
3. `DFS_tree_search`：整棵搜索树共享的预算、终止节点和搜索进度。

因此，更完整的抽象是：

$$
S_t=(q,\mathcal A,H_t,E_t,N_t,B_t)
$$

| 状态组成 | 含义 | ToolBench 中的实际对象 |
|----------|------|----------------------|
| $q$ | 当前任务 | `rapidapi_wrapper.input_description` |
| $\mathcal A$ | 当前可选动作集合 | `rapidapi_wrapper.functions` |
| $H_t$ | 当前分支的消息历史 | `tree_node.messages` |
| $E_t$ | 当前分支的工具环境副本 | `tree_node.io_state` |
| $N_t$ | 当前搜索节点状态 | `father/children/depth/pruned/is_terminal` |
| $B_t$ | 全局搜索预算和进度 | `query_count/total_tokens/terminal_node` |

这里必须区分：

```text
完整 Agent State
    ≠
模型本轮看到的上下文
```

模型通常只看到 `messages + functions`；环境内部映射、API key、其他搜索分支和全局预算并不会自动提供给模型。

#### 1.1 第一层：`rapidapi_wrapper`——任务和环境状态

**代码位置**：`toolbench/inference/Downstream_tasks/rapidapi.py::rapidapi_wrapper`

初始化时保存：

```python
self.input_description = query_json["query"]
self.functions = []
self.api_name_reflect = {}
self.tool_names = []
self.cate_names = []
self.success = 0
```

可以分成四部分。

##### 任务状态

```text
input_description：用户真正提出的问题
task_description：工具使用规则、工具描述和必须调用 Finish 等约束
```

##### 动作空间

```python
self.functions
```

例如：

```text
functions = [
    projects_for_squake,
    checkhealth_for_squake,
    Finish
]
```

闭域和开放域主要在动作空间构造阶段不同：

```text
闭域：query.api_list → functions
开放域：Retriever(query) → top-k API → functions
```

开放域 Retriever 只在 episode 初始化阶段运行。进入搜索分支后，代码把：

```python
child_io_state.retriever = None
```

因此 ToolBench 不是每一步都重新检索工具，而是先固定本次任务的候选 API 集合，再在集合内执行搜索。

##### 工具路由状态

```text
api_name_reflect
tool_names
cate_names
```

它们负责把模型生成的：

```text
projects_for_squake
```

映射回底层执行需要的：

```text
category = Logistics
tool_name = squake
api_name = projects
tool_input = {...}
```

##### 可变环境状态

当前 `rapidapi_wrapper` 真正会随 episode 改变的关键字段主要是：

```python
self.success
```

初始为 `0`。模型调用：

```json
{
  "name": "Finish",
  "arguments": {
    "return_type": "give_answer",
    "final_answer": "..."
  }
}
```

之后，环境设置：

```python
self.success = 1
```

并返回状态码 `3`。

因此当前 RapidAPI 环境实际上接近“无状态查询工具环境”：普通 API 调用通常不会修改本地业务状态，最主要的内部变化是任务是否已经结束。代码中的 `restart()` 是空实现，`get_score()` 也固定返回 `0.0`。

#### 1.2 第二层：`tree_node`——单条搜索分支的状态快照

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
pruned / is_terminal / finished
expand_num / Elo / prior_score
```

可以抽象为：

$$
N_t=(H_t,E_t,a_t,o_t,c_t,\text{search metadata})
$$

##### `messages`：模型可见的分支历史

```python
tree_node.messages
```

它保存当前分支积累的 OpenAI 风格消息：

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "查询项目列表"},
  {
    "role": "assistant",
    "content": "I should retrieve the projects.",
    "function_call": {
      "name": "projects_for_squake",
      "arguments": "{}"
    }
  },
  {
    "role": "function",
    "name": "projects_for_squake",
    "content": "{\"error\":\"\",\"response\":\"[...]\"}"
  }
]
```

下一次模型决策主要依赖：

```text
messages + functions
```

这部分更准确地说是 Policy 当前获得的 Observation，而不是系统完整 State。

##### `io_state`：当前分支的环境副本

初始化根节点时：

```python
self.tree.root.io_state = deepcopy(self.io_func)
```

创建子节点时：

```python
child_io_state = deepcopy(temp_now_node.io_state)
```

所以每条分支都有自己的环境视图：

```text
父节点
├── 分支 A：io_state_A
└── 分支 B：io_state_B
```

假设分支 A 调用了 `Finish`：

```text
io_state_A.success = 1
```

分支 B 的环境仍可保持：

```text
io_state_B.success = 0
```

一个分支终止不会直接把其他候选分支标记为成功。

##### `observation`：最近一次工具返回

```text
tree_node.observation
tree_node.observation_code
```

例如：

```python
observation = '{"error":"","response":"[project1, project2]"}'
observation_code = 0
```

完整历史 observation 会以 `role=function` 进入 `messages`；节点又单独保存最近一次 observation，方便打印、剪枝和导出搜索树。

##### 搜索控制状态

```text
father / children：节点在搜索树中的位置
pruned：该分支是否停止探索
is_terminal：该节点是否已经结束任务
finished：用于把成功路径向祖先标记
expand_num：全局展开顺序
depth：通过 father 递归计算
```

这些不是业务环境状态，而是 DFS/DFSDT 的规划状态。

#### 1.3 第三层：`DFS_tree_search`——全局搜索状态

**代码位置**：`toolbench/inference/Algorithms/DFS.py::DFS_tree_search`

它保存：

```python
self.tree
self.status
self.terminal_node
self.give_up_node
self.now_expand_num
self.query_count
self.total_tokens
```

| 字段 | 含义 |
|------|------|
| `tree` | 整棵候选搜索树 |
| `terminal_node` | 已找到的成功终止节点 |
| `give_up_node` | 主动放弃的节点 |
| `query_count` | 整次搜索共享的模型调用次数 |
| `total_tokens` | 整次搜索共享的 token 消耗 |
| `now_expand_num` | 全局节点展开顺序 |
| `status` | 搜索是否找到成功答案 |

这些状态通常不会传给模型。模型并不知道还有多少未探索分支、全局已经调用多少次 LLM，除非搜索算法通过额外 prompt 显式告诉它。

#### 1.4 一次 Action 为什么拆成三个节点

ToolBench 把模型的一次输出拆成：

```text
Thought
  ↓
Action
  ↓
Action Input
```

例如：

```text
Thought: I should retrieve projects.
Action: projects_for_squake
Action Input: {}
```

树中表示为：

```text
Thought node
description = "I should retrieve projects."

Action node
description = "projects_for_squake"

Action Input node
description = "{}"
observation = "API result"
observation_code = 0
```

真正执行环境转移的是 `Action Input` 节点：

```python
observation, status = child_io_state.step(
    action_name=temp_now_node.description,
    action_input=function_input
)
```

因此一个 Agent 交互步可能增加三个树深度：

$$
N_t
\rightarrow N_t^{Thought}
\rightarrow N_t^{Action}
\rightarrow N_{t+1}^{ActionInput}
$$

这意味着 `single_chain_max_step=12` 限制的是树节点深度，而不严格等于 12 次工具调用。如果每轮都有三个节点，通常大约容纳 4 轮完整动作。

#### 1.5 一次完整的状态转移

假设初始状态：

```text
query = 查询 SQUAKE 项目列表
functions = [projects_for_squake, Finish]
messages = [system, user]
io_state.success = 0
```

可以表示为：

$$
S_0=(q,\mathcal A,H_0,E_0,N_0,B_0)
$$

##### 第一步：模型读取当前可见状态

```python
self.llm.change_messages(temp_now_node.messages)
new_message = self.llm.parse(self.io_func.functions)
```

模型实际获得的主要是：

```text
H_0 + functions
```

并输出：

```text
Thought: 需要先查询项目列表
Action: projects_for_squake
Action Input: {}
```

##### 第二步：复制父分支环境

```python
child_io_state = deepcopy(temp_now_node.io_state)
```

工具不会直接修改父节点环境，而是在新的候选分支副本上执行。

##### 第三步：执行动作

```python
observation, status = child_io_state.step(
    action_name="projects_for_squake",
    action_input="{}"
)
```

得到：

```text
observation = [project1, project2]
status = 0
```

##### 第四步：保存反馈并更新消息

结果写入 Action Input 节点：

```python
temp_node.observation = observation
temp_node.observation_code = status
temp_node.io_state = child_io_state
```

同时追加 assistant function call 和 function observation：

```text
H_1 = H_0 + Action + Observation
```

新的分支状态为：

$$
S_1=(q,\mathcal A,H_1,E_1,N_1,B_1)
$$

##### 第五步：根据状态码控制搜索

```text
0  → 正常 observation，继续
1  → 函数名幻觉，错误写回 messages
2  → 参数错误，错误写回 messages
3  → Finish/give_answer，terminal
4  → give_up_and_restart，pruned 并回溯
5–12 → API 超时、授权、限流或服务错误
```

#### 1.6 DFS 分支如何隔离和回溯

假设同一个父状态生成两个候选：

```text
父状态 S_0
├── A：projects_for_squake({})
└── B：checkhealth_for_squake({})
```

每个候选都复制父分支的 `messages` 和 `io_state`：

```text
S_0
├── S_1^A
│   ├── messages_A
│   ├── io_state_A
│   └── observation_A
└── S_1^B
    ├── messages_B
    ├── io_state_B
    └── observation_B
```

DFS 深入 A 失败后，可以返回父节点继续探索 B，不需要显式恢复 A 之前的 Python 对象。

但是，`deepcopy` 只能复制 Python 包装器，不能回滚外部世界。如果工具执行的是：

```text
send_email
delete_file
place_order
write_database
```

DFS 回溯不会撤销已经发生的副作用。因此 ToolBench 的分支复制更适合查询型、无副作用或可重复执行的研究工具，不适合直接照搬到生产写操作。

#### 1.7 从 MDP/POMDP 角度理解

完整系统状态：

$$
S_t=(q,\mathcal A,H_t,E_t,N_t,B_t)
$$

模型实际获得的 observation 通常只有：

$$
O_t=(H_t,\mathcal A)
$$

模型看不到：

```text
其他未探索分支
DFS frontier 和回溯栈
全局 query budget
terminal_node 列表
API 服务真实内部状态
```

所以从严格意义上说：

```text
完整 State ≠ 模型 Observation
```

ToolBench 代码没有显式使用这两个术语进行类型区分，但工程上可以这样理解：

```text
tree_node + rapidapi_wrapper + DFS_tree_search
    = 完整系统 State

messages + functions
    = Policy 当前 Observation

function_call
    = Action

role=function 消息 + status code
    = 环境 Feedback

新的 child node
    = Next State
```

#### 1.8 这种 State 设计是不是当前标准

结论是：

> ToolBench 的“状态分层、分支隔离、Action–Observation 转移”思想仍然是标准思想；但“状态散落在多个可变对象中，并靠 `deepcopy` 复制整个环境”的实现属于研究原型，不是当前生产级 Agent 的最佳实践。

##### 仍然值得保留的设计

```text
模型消息历史与工具环境分离
每条搜索分支拥有独立状态
Action → Environment → Observation → Next State
模型动作策略与 DFS 搜索策略分离
搜索节点保存 terminal/pruned 等控制状态
```

##### 不建议直接照搬的实现

```text
State 没有显式 schema
消息、环境和搜索状态边界不清晰
依赖 deepcopy 复制整个环境
模型、Retriever、连接和密钥可能混入状态对象
没有中间 checkpoint 和持久化恢复协议
无法真正回滚外部工具副作用
没有工具调用幂等、审批和 transaction 状态
```

#### 1.9 当前流行的 Agent State 分层

当前并不存在所有框架共用的唯一 `AgentState` 标准类，但主流实现大体收敛为以下分层：

```text
Agent Definition          不可变 Agent 配置
    ↓
Run State                 当前任务可序列化状态
    ↓
Model Context             本轮真正提供给模型的信息
    ↓
Runtime Context           工具和代码依赖
    ↓
Session / Memory          跨轮和跨任务持久化信息
    ↓
Orchestration State       工作流、搜索、并发与恢复状态
```

##### Agent Definition：不可变配置

```text
Agent 名称
Instructions/System Prompt
模型配置
工具 schema
Guardrails
最大轮数
输出格式
```

它回答“Agent 是什么”，不应在每一步复制。

##### Run State：当前任务状态

推荐显式定义可序列化 schema：

```python
class AgentState(TypedDict):
    run_id: str
    session_id: str
    task: str
    messages: list[dict]
    available_tools: list[str]
    pending_tool_calls: list[dict]
    tool_results: list[dict]
    status: str
    current_step: int
    budget: dict
    error_history: list[dict]
    final_answer: str | None
```

它应满足：

```text
可 JSON 序列化
可以从 checkpoint 恢复
有明确状态机
不包含数据库连接、线程锁、GPU tensor 或打开的文件句柄
```

常见状态机：

```text
CREATED
  → RUNNING
  → WAITING_APPROVAL
  → EXECUTING_TOOL
  → RUNNING
  → COMPLETED

也可能：
RUNNING → FAILED → RETRYING → RUNNING
```

##### Model Context：模型本轮可见信息

完整 State 不应全部塞进 prompt，而应通过 Context Builder 选择：

```python
model_context = build_model_context(
    state=state,
    session=session,
    retrieved_memories=memories,
)
```

通常只选择：

```text
System instructions
用户请求
必要消息历史
相关记忆或检索结果
最近工具结果
当前工具 schema
必要任务约束
```

敏感权限、数据库连接、其他搜索分支和内部审计信息不应自动发送给模型。

##### Runtime Context：本地依赖

```python
@dataclass
class RuntimeContext:
    db: DatabaseClient
    http: HttpClient
    tool_registry: ToolRegistry
    credential_provider: CredentialProvider
    logger: Logger
    artifact_store: ArtifactStore
```

一个实用原则是：

> State 保存“发生了什么”；Runtime Context 提供“如何执行”。

##### Session、Memory 与 Artifact

现代 Agent 通常区分：

| 层次 | 内容 | 生命周期 |
|------|------|----------|
| Working Memory | 当前计划、工具结果摘要、当前消息 | 一次 run |
| Session Memory | 本次多轮对话历史、临时偏好 | 一个 session |
| Long-term Memory | 用户稳定偏好、历史经验、领域知识 | 跨 session |
| Artifact Store | 文件、代码、网页全文、报告、大结果 | 独立持久化 |

大对象不应全部塞入 State，推荐保存摘要和引用：

```json
{
  "summary": "检索到 38 条相关法条",
  "artifact_ref": "artifact://law-results/456"
}
```

#### 1.10 当前流行的 State 更新机制

现代 graph/workflow Agent 通常让节点返回“状态增量”，而不是任意修改全局对象：

```python
def node(state: AgentState, runtime: RuntimeContext) -> dict:
    return {
        "messages": [new_message],
        "current_step": state["current_step"] + 1,
        "status": "waiting_tool",
        "pending_tool_calls": [tool_call],
    }
```

再由 reducer 合并：

| Reducer | 适用字段 | 行为 |
|---------|----------|------|
| Replace | `status/final_answer/last_observation` | 新值覆盖旧值 |
| Append | `messages/tool_results/errors/events` | 追加历史 |
| Merge | `facts/subtask_results/artifact_refs` | 按 key 合并 |
| Monotonic | `tokens/steps/retries/tool_calls` | 只增加不回退 |
| State transition | `running → waiting_tool → completed` | 校验合法状态转移 |

这种方式比随处执行：

```python
state.messages.append(...)
state.success = 1
```

更容易验证、并行合并、持久化和调试。

#### 1.11 Event Log、Checkpoint 与恢复

只保存最新 State 无法回答“为什么变成这样”，因此常同时记录 append-only event：

```text
UserMessageReceived
ModelResponseReceived
ToolCallProposed
ToolCallApproved
ToolExecutionStarted
ToolExecutionSucceeded
ToolExecutionFailed
StateUpdated
RunPaused
RunCompleted
```

三者关系：

```text
Event：发生了什么
State：所有事件应用后的当前结果
Checkpoint：某一时刻 State 的持久化快照
```

Checkpoint 通常还要记录：

```python
class Checkpoint(TypedDict):
    checkpoint_id: str
    run_id: str
    parent_checkpoint_id: str | None
    state_version: int
    state: AgentState
    next_node: str | None
    pending_tasks: list[dict]
    interruptions: list[dict]
```

`next_node` 很重要，因为恢复时不仅要知道数据，还要知道下一步应该：

```text
调用模型
执行工具
等待审批
合并子任务
生成最终答案
```

推荐 checkpoint 边界：

```text
收到用户请求后
模型生成 Tool Call 后
执行有副作用工具前
工具执行完成后
等待人工审批时
Agent handoff 后
子任务完成后
最终结束时
```

#### 1.12 工具调用状态与副作用治理

生产系统通常单独建模 Tool Call：

```python
class ToolCallState(TypedDict):
    call_id: str
    tool_name: str
    arguments: dict
    effect_type: str
    status: str
    idempotency_key: str
    attempts: int
    result_ref: str | None
    error: dict | None
```

典型状态：

```text
proposed
  → waiting_approval
  → approved
  → running
  → succeeded / failed / rejected
```

`call_id` 和 `idempotency_key` 用于防止进程恢复或重试时重复发送邮件、重复下单或重复写数据库。

对于有副作用工具，还需要：

```text
权限校验
人工审批
幂等键
transaction
compensation action
dry-run
执行审计
```

#### 1.13 搜索型 Agent 的现代 State

对 ToolBench 这种 DFS/Tree Search Agent，可以把搜索状态单独定义为：

```python
class SearchNodeState(TypedDict):
    node_id: str
    parent_id: str | None
    checkpoint_id: str
    action: dict | None
    observation_ref: str | None
    depth: int
    score: float
    terminal: bool
    pruned: bool

class SearchState(TypedDict):
    root_node_id: str
    frontier: list[str]
    visited: list[str]
    terminal_nodes: list[str]
    expanded_nodes: int
    model_calls: int
    max_model_calls: int
```

Search Node 不复制模型、Retriever、HTTP Client、数据库连接和真实 Tool Registry，只引用：

```text
checkpoint_id
parent_id
action
observation_ref
score
```

分支通过 checkpoint fork 或 parent + delta 表达：

```text
Checkpoint 10
├── Checkpoint 11A
└── Checkpoint 11B
```

比 `deepcopy(io_state)` 更容易持久化、并行化和跨机器运行。

#### 1.14 ToolBench 到现代 State 的映射

| ToolBench | 更现代的设计 |
|-----------|---------------|
| `tree_node.messages` | `AgentState.messages` / Model Context 来源 |
| `tree_node.observation` | `ToolResultEvent` 或 artifact reference |
| `tree_node.observation_code` | `ToolCallState.status/error` |
| `rapidapi_wrapper.functions` | `AgentDefinition.tools` |
| API key、Retriever、HTTP client | `RuntimeContext` |
| `rapidapi_wrapper.success` | `AgentState.status` |
| `query_count/total_tokens` | `AgentState.budget/usage` |
| `father/children/pruned` | `SearchNodeState` |
| `deepcopy(io_state)` | checkpoint fork / parent + delta |
| 最终搜索树 JSON | Event Log + Checkpoint History |
| `Finish` | 显式 `COMPLETED` 状态转移 |

一套最小但完整的现代架构可以是：

```text
AgentDefinition
├── instructions
├── model
└── tool schemas

AgentState
├── task / messages
├── pending tool calls / tool results
├── status / budget / errors
└── final answer

RuntimeContext
├── model client / tool registry
├── credentials / db / http clients
└── artifact store

SessionState / MemoryStore
├── conversation history
├── summary
└── long-term memories

EventLog / CheckpointStore
├── append-only events
└── serialized state + next node
```

执行循环：

```text
1. 加载 AgentState
2. 从 State、Session 和 Memory 编译 Model Context
3. 调用模型
4. 将模型输出记录为 Event
5. 验证 Tool Call，必要时等待审批
6. 执行 Tool
7. 保存 Tool Result Event
8. reducer 生成 New State
9. 保存 Checkpoint
10. 决定继续、暂停或结束
```

判断一套 Agent State 是否成熟，可以检查：

```text
State schema 是否显式？
哪些字段模型可见？
哪些字段只供 Runtime 使用？
是否可以安全序列化？
是否能从 checkpoint 恢复？
是否记录下一执行节点？
工具调用是否有 call_id 和幂等键？
并行分支如何合并状态？
Memory、Session、Artifact 是否分开？
是否支持暂停、审批、取消和重放？
State schema 升级时旧 checkpoint 如何迁移？
```

进一步参考：

- [LangGraph：Graph State 与 Reducer](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph：Persistence 与 Checkpoint](https://docs.langchain.com/oss/python/langgraph/persistence)
- [OpenAI Agents SDK：Context management](https://openai.github.io/openai-agents-python/context/)
- [OpenAI Agents SDK：Sessions](https://openai.github.io/openai-agents-python/sessions/)
- [OpenAI Agents SDK：Run State](https://openai.github.io/openai-agents-python/ref/run_state/)
- [AutoGen：Managing State](https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/state.html)

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

#### 2.1 第一层补充：ToolBench 当前 Action 设计

ToolBench 的 Action 不只是模型输出的一段文字，而是一个由动态动作空间、动作提议、解析协议和环境调用共同组成的对象：

~~~text
候选 functions
    ↓
ToolLLaMA / ChatGPT 提出动作
    ↓
function name + arguments
    ↓
env.step(action_name, action_input)
~~~

候选动作空间来自：

~~~text
闭域：query.api_list → env.functions
开放域：Retriever top-k → env.functions
~~~

环境还会为每个 episode 追加 Finish。因此 ToolBench 的动作空间不是固定词表，而是随任务动态生成的有限 function 集合。

Action 中实际存在三种语义：

| 内容 | 是否改变环境 | 作用 |
|------|--------------|------|
| Thought | 否 | 保存局部推理，支持搜索轨迹 |
| function name + arguments | 是 | 选择并调用真实工具 |
| Finish | 是，元动作 | 给出答案或声明放弃重启 |

ToolLLaMA 依赖 ReAct 字符串解析；ChatGPT Function 模型直接返回 function_call。二者最终被适配为同一种环境调用协议。

#### 2.2 第二层：与现代 Agent Action 标准设计的异同

ToolBench 仍符合现代 Action 设计的部分：

~~~text
动作与最终自然语言答案分离
动作受候选工具 schema 约束
工具名称和参数被显式记录
模型只提出动作，由 Runtime 执行
Finish 是显式终止动作
失败动作和成功动作都进入轨迹
~~~

但它仍有明显的研究原型特征：

~~~text
Action Input 本质是 JSON 字符串
ToolLLaMA 依赖 Thought/Action/Action Input 文本切割
没有统一类型化 Action 类
没有 action_id、attempt、deadline
没有 risk_level、required_scope、requires_approval
没有 idempotency_key、dry_run 和 compensation
工具名匹配依赖字符串规范化
模型生成动作与动作可执行之间缺少显式校验层
~~~

现代 Agent 通常使用以下边界：

~~~text
Model Output
    ↓ Parse
Typed Decision
    ↓ Validate
Authorized Action
    ↓ Approval / Risk Gate
Executable Command
    ↓ Executor
Execution Event
~~~

推荐的类型化 Action 可以包含：

~~~python
class ToolAction(TypedDict):
    action_id: str
    tool_id: str
    operation: str
    arguments: dict
    timeout_seconds: int
    idempotency_key: str | None
    risk_level: str
    requires_approval: bool
    retry_policy: dict
~~~

核心原则是：

> Policy 产生意图明确、可校验的 Decision；Runtime 才把 Decision 转成具有副作用的 Command。

现代终止动作也不应只有 Finish/give_answer，通常还应区分：

~~~text
Complete(final_answer, evidence)
NeedMoreInformation(question)
WaitForApproval(action)
Handoff(target, context)
Abort(reason)
Retry(action_id, repair)
~~~

#### 2.3 第三层：现代 Action 设计思路和优秀实例

##### Schema-first Tool Action

先定义工具输入输出 schema，再让模型填槽，而不是依赖自由文本切割。

例：订单物流 Agent 生成：

~~~json
{
  "type": "tool_call",
  "action_id": "act_102",
  "tool_id": "logistics.get_tracking",
  "arguments": {"order_id": "O-20260812-01"},
  "risk_level": "read",
  "requires_approval": false
}
~~~

Runtime 使用 JSON Schema 或 Pydantic 校验订单号。参数无效时产生 ValidationFailed Event，不进入真实工具。

##### Command 与 Domain Event 分离

例：退款 Agent。

~~~text
Policy Decision：RequestRefund(order_id, amount, reason)
    ↓ Guardrail
Executable Command：PaymentGateway.refund(...)
    ↓ Executor
Domain Event：RefundSucceeded / RefundRejected
~~~

模型不能直接声称 RefundSucceeded；该事件只能由 Executor 根据真实支付网关结果产生。

##### 副作用动作使用审批和幂等

例：批量邮件 Agent。

~~~text
生成 SendCampaign Action
→ 参数校验
→ 影响范围预览
→ 用户审批
→ 以 idempotency_key 执行一次
→ 记录发送结果
~~~

DFS 搜索阶段不能让每条候选分支真实发送邮件，只能产生计划或 dry-run；确认后的唯一分支才允许执行。

##### 分层 Action

大动作空间可以按以下层次逐步缩小：

~~~text
Goal
→ Skill
→ Tool
→ Arguments
~~~

例：企业故障处理先选择 diagnose_service，再开放日志、指标和 trace 工具；只有进入 recover_service 且审批通过后，才开放回滚工具。

ToolBench 的 Retriever 是这种思想的早期形式，但它通常只在 episode 开始时召回一次工具，不会随 Goal、权限和阶段动态改变动作空间。

#### 2.4 第四层：Action 关联训练模式

| 训练模式 | 数据单位 | 作用 |
|----------|----------|------|
| Step SFT / 行为克隆 | 当前 Observation → 正确 ToolAction | 学习下一步工具调用 |
| Argument SFT | query + tool schema → arguments | 提高参数合法性 |
| Tool Routing | task + state → tool_id/skill_id | 缩小动作空间 |
| DPO / Preference | 同一 State 下 chosen Action > rejected Action | 学习低风险、高成功动作 |
| Action Verifier | State + proposed Action → label/reason | 调用前校验或搜索排序 |
| Value/Q Model | State + Action → eventual success | 选择值得探索的分支 |
| Offline/Online RL | trajectory + reliable reward | 优化成功、成本和长度 |

ToolBench 自身主要采用：

~~~text
DFSDT 生成成功轨迹
→ train_messages 拆成逐步前缀
→ Observation/History 预测 Next Action
→ ToolLLaMA SFT
~~~

Retriever 训练负责动作空间召回，ToolLLaMA SFT 负责候选集合内的具体动作选择，两者不是同一个训练任务。

#### 2.5 Action 训练和工程注意点

~~~text
训练和推理必须使用相同 Action Schema
工具名和参数标准化协议必须一致
保存 raw model output、parsed action 和 validation result
不要只保存成功动作，也要保存无效参数及其修复动作
Action mask 必须反映当前权限和 workflow phase
不能把模型生成了 Action 当成 Action 已执行
副作用动作必须有 action_id 和 idempotency_key
搜索候选与真实执行必须分离
敏感权限必须由确定性 Runtime 判断
Finish 应由 Verifier 检查，而不是只相信模型声明
~~~

训练记录至少应包含：

~~~text
state/context version
available tool schemas
raw model output
typed action
validation result
executed command
tool result
retry/repair relation
terminal outcome
cost/latency
human approval
~~~

#### 2.6 Action 章节结论

ToolBench 当前设计：

~~~text
Dynamic functions
+ ReAct text / function_call
+ function name and arguments
+ Finish meta-action
+ DFS branch recording
~~~

现代化方向：

~~~text
Dynamic Action Space
→ Typed Decision
→ Schema Validation
→ Risk / Permission / Approval
→ Idempotent Command
→ Execution Event
~~~

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

#### 3.1 第一层补充：ToolBench 当前 Observation 设计

ToolBench 的 Observation 实际有两个消费方：

~~~text
模型 Policy
    读取 role=function 的 observation 内容

搜索控制器
    读取 tree_node.observation_code
~~~

因此同一次工具返回同时承担事实反馈和控制反馈。状态码语义如下：

| 状态码 | 含义 | 控制效果 |
|-------:|------|----------|
| 0 | 正常返回 | 继续 |
| 1 | 不存在的函数名 | 写回幻觉错误 |
| 2 | 参数或 Finish JSON 错误 | 写回错误 |
| 3 | Finish/give_answer | terminal |
| 4 | give_up_and_restart | prune、回溯 |
| 5 | 超时 | 工具失败 |
| 6 | API 不工作/404 | 工具失败 |
| 7–8 | 订阅或授权问题 | 工具失败 |
| 9–10 | 请求或速率限制 | 延迟、重试或失败 |
| 11 | API message error | 工具失败 |
| 12 | 请求发送或响应解析失败 | 工具失败 |

Observation 超过 max_observation_length 后会被截断。observ_compress_method 同时传给后端，但 wrapper 最终仍可能进行字符级截断。

#### 3.2 第二层：与现代 Agent Observation 标准设计的异同

ToolBench 做对的部分：

~~~text
工具结果进入下一轮 Model Context
成功和失败都反馈给模型
原始 response 与控制 status 同时保存
Observation 绑定具体 function name
搜索树保留每个分支的 observation
~~~

与现代实现相比的不足：

~~~text
Observation 主要是无类型字符串
error 和 response 没有统一 Error Taxonomy
status code 混合执行状态、业务结果和搜索控制
字符截断可能破坏 JSON 和关键字段
没有 artifact reference、分页和流式结果协议
没有 provenance、时间戳、latency、cost、trace_id
没有标记结果是否完整、过期、缓存或重试所得
模型可见内容与内部执行元数据没有明确分层
没有针对不可信工具输出的 prompt-injection 隔离
~~~

现代系统通常区分四层：

~~~text
Raw Tool Result
    原始响应、HTTP 状态、headers、trace

Normalized Observation
    类型化数据、标准错误、provenance

Model Observation
    经过筛选、压缩、脱敏后提供给模型

Control Event
    ToolSucceeded / ToolFailed / RateLimited 等 Runtime 事件
~~~

推荐结构：

~~~python
class ToolObservation(TypedDict):
    action_id: str
    tool_id: str
    status: str
    data: dict | list | str | None
    error: dict | None
    is_partial: bool
    artifact_refs: list[str]
    provenance: dict
    latency_ms: int
    cost: float | None
    retryable: bool
~~~

完整 Observation 不必全部进入 Prompt。Context Builder 只选择与当前目标相关的数据、标准化错误、必要来源和恢复提示。

#### 3.3 第三层：现代 Observation 设计思路和优秀实例

##### 结果与错误使用判别联合类型

天气工具成功时：

~~~json
{
  "type": "tool_succeeded",
  "action_id": "act_3",
  "data": {
    "city": "Shanghai",
    "temperature_c": 32,
    "observed_at": "2026-08-12T14:00:00+08:00"
  },
  "provenance": {
    "provider": "weather_service",
    "freshness_seconds": 120
  }
}
~~~

失败时：

~~~json
{
  "type": "tool_failed",
  "action_id": "act_3",
  "error": {
    "code": "RATE_LIMITED",
    "message": "Retry after 30 seconds",
    "retry_after_seconds": 30
  },
  "retryable": true
}
~~~

Policy 可以据此选择等待、换工具或向用户说明，而不必从任意错误文本中猜测。

##### 大结果进入 Artifact Store

SQL 查询返回十万行时：

~~~text
Raw result → parquet/csv artifact

Model Observation →
row_count: 100000
columns: [...]
preview: 前 20 行
artifact_ref: artifact://query/123/result.parquet
~~~

这比字符截断更可靠，也能支持后续工具读取完整结果。

##### Observation 携带 provenance

研究 Agent 的每个事实应关联：

~~~text
source URL / document id
retrieved_at
content hash
quotation span
extraction method
~~~

最终 Verifier 才能检查结论是否被真实工具结果支持。

##### 工具输出按不可信输入处理

网页、邮件和文档可能包含恶意提示。进入模型前应：

~~~text
标记来源和数据边界
限制工具输出不能覆盖 system policy
解析结构化字段而非直接拼接全文
脱敏凭据和个人信息
检测或隔离 prompt injection
~~~

#### 3.4 第四层：Observation 关联训练模式

| 训练模式 | 输入 → 目标 | 作用 |
|----------|-------------|------|
| Error-recovery SFT | State + typed error → retry/repair/fallback | 学习错误恢复 |
| Observation Grounding | Observation → supported answer/action | 减少忽略工具结果 |
| Process Verifier | State + Observation + Next Action → score | 评估每步合理性 |
| Result Summarization | large raw result → compact faithful context | 压缩长结果 |
| Error Classification | raw exception → standard error code | 统一错误 taxonomy |
| Contrastive Recovery | 有效恢复 > 重复失败或编造结果 | DPO/Ranker |
| Citation Training | result + sources → cited answer | 提高可验证性 |

错误恢复数据还应记录失败类别、是否可重试、修正后的 Action、是否换用 fallback、重试次数和最终是否恢复。

#### 3.5 Observation 训练和工程注意点

~~~text
训练和推理的错误表示必须一致
不要只给模型数字状态码而缺少语义
截断前先结构化解析，保证 JSON/schema 完整
Observation 必须与 action_id 对齐
区分空结果、失败、权限不足和任务本身无答案
保存 freshness、provenance、latency 和 cost
大对象使用 artifact reference
重试结果标记 attempt 和 parent action
工具输出必须视为不可信数据
不要把堆栈、密钥和敏感 headers 暴露给模型
最终答案训练应检查是否由 observation 支持
~~~

必须防止 train–inference distribution shift。例如训练数据用自然语言表示错误，推理时只返回整数状态码，模型就难以学会正确恢复。

#### 3.6 Observation 章节结论

ToolBench 当前：

~~~text
API response string
+ integer status code
+ role=function message
+ tree_node observation
~~~

现代化方向：

~~~text
Raw Result
→ Normalize
→ Typed Observation / Error
→ Provenance + Artifact
→ Context Projection
→ Recovery Decision
~~~

### 4. Policy：策略是什么

ToolBench 没有单独定义名为 `Policy` 的类。和前面的 State 一样，Policy 也是一个**分散式实现**：模型负责提出下一动作，推理算法负责组织搜索，候选 Ranker 负责估计分支质量，工具环境和预算规则负责决定继续、终止、剪枝或回溯。

因此，不能把 ToolLLaMA、DFSDT 或 `rapidapi_wrapper` 中的任意一个对象单独等同于完整 Policy。更完整的抽象是：

$$
\Pi=(\pi_\theta,\mu,\rho,\tau)
$$

| 组成 | 含义 | ToolBench 实现 |
|------|------|----------------|
| $\pi_\theta$ | 根据当前 Observation 提出下一步动作 | `ToolLLaMA.parse()`、`ChatGPTFunction.parse()`、`Davinci.parse()` |
| $\mu$ | 决定怎样采样、展开和回溯候选 | `single_chain.py`、`DFS.py` |
| $\rho$ | 判断哪个候选分支更有希望 | `LLM_rank/rank_candidate.py` |
| $\tau$ | 根据环境结果和预算决定继续、终止或剪枝 | `rapidapi_wrapper` 状态码、深度和 query budget |

换一种更直观的表达：

```text
ToolBench Policy
├── Action Proposal Policy      模型提出 Thought / Action / Action Input
├── Exploration/Search Policy   Single Chain、CoT@N、DFS、DFSDT
├── Ranking/Value-like Policy   LLM 对候选路径排序
└── Transition/Stop Policy      状态码、Finish、深度、预算与回溯规则
```

这里的 `Policy` 有两种口径：

- **狭义 Policy**：模型在当前 Observation 下生成下一动作，即 $\pi_\theta(a_t\mid o_t)$；
- **广义 Policy**：整个 Agent 如何生成、校验、执行、选择、恢复和终止，即 $\Pi$。

后文讨论 ToolBench 时会同时使用这两个口径，并明确指出具体指哪一层。

#### 4.1 Policy 与 State 如何联动

前面已经说明，ToolBench 的完整状态可以抽象为：

$$
S_t=(q,\mathcal A,H_t,E_t,N_t,B_t)
$$

但模型 Policy 并不会看到完整 $S_t$。它主要看到从 State 投影出来的 Observation：

$$
o_t=\phi(S_t)\approx(H_t,\mathcal A)
$$

其中：

- $H_t$：当前搜索分支上的 system、user、assistant、function 消息；
- $\mathcal A$：当前候选 API 的 function schema，加上 `Finish`；
- $E_t$、$N_t$、$B_t$ 中的大部分运行时信息由 DFS 和环境使用，不直接进入模型上下文。

模型根据 Observation 生成动作：

$$
a_t\sim\pi_\theta(a\mid o_t)
$$

环境执行动作并返回 Observation 与状态码：

$$
(r_t,z_t)=\operatorname{EnvStep}(E_t,a_t)
$$

搜索控制器再综合工具结果、状态码、节点深度和预算，得到下一状态：

$$
S_{t+1}=T(S_t,a_t,r_t,z_t;\mu,\tau)
$$

所以 ToolBench 的真实运行关系是：

```text
完整 State S_t
    ↓ 取当前分支的 messages 和 functions
模型 Observation o_t
    ↓ ToolLLaMA.parse()
候选 Action a_t
    ↓ rapidapi_wrapper.step()
工具 Observation r_t + status z_t
    ↓ DFS / single_chain 更新树、环境和预算
新 State S_t+1
```

这也解释了为什么 State 和 Policy 必须联动理解：

- State 不记录失败次数，Policy 就无法避免重复失败；
- State 不记录当前分支，DFS 就无法正确回溯；
- State 不记录工具结果，模型就无法根据 Observation 修正动作；
- State 不记录预算，Runtime 就无法限制无限搜索；
- Policy 输出格式不稳定，Environment 就无法把它解释为合法 Action。

#### 4.2 模型动作 Policy：ToolLLaMA 具体生成什么

核心入口：

```text
toolbench/inference/LLM/tool_llama_model.py
└── ToolLLaMA.parse()
```

其他可替换实现：

```text
ChatGPTFunction.parse()
Davinci.parse()
ToolLLaMALoRA.parse()
```

`ToolLLaMA.parse()` 的主要过程是：

1. 从当前分支提取 messages；
2. 把可用 `functions` 写入 system prompt；
3. 拼接 `Assistant:`，要求模型继续生成；
4. 生成 ReAct 格式文本；
5. 使用 `react_parser()` 提取 Thought、Action 和 Action Input；
6. 转换成类似 OpenAI Function Calling 的字典。

模型输出格式：

```text
Thought: I need to query the project list first.
Action: projects_for_squake
Action Input: {"page": 1}
```

解析后：

```json
{
  "role": "assistant",
  "content": "I need to query the project list first.",
  "function_call": {
    "name": "projects_for_squake",
    "arguments": "{\"page\": 1}"
  }
}
```

因此 ToolLLaMA 的动作不是单纯的工具名称，而是：

$$
a_t=(\text{Thought},\text{Action},\text{Action Input})
$$

其中：

- `Thought`：局部推理或局部计划；
- `Action`：function 名称；
- `Action Input`：JSON 风格参数；
- 当 `Action=Finish` 时，动作还承担停止或放弃分支的职责。

需要注意，`react_parser()` 主要依赖：

```text
Thought:
Action:
Action Input:
```

这些字符串标记做切分，因此它不是严格的类型化 Action 协议。模型少输出换行、输出额外标题或生成非法 JSON，都可能导致解析或执行异常。现代 Agent 更常使用 JSON Schema 或原生 tool call 来约束 Action。

#### 4.3 `Finish`：停止也是模型动作的一部分

ToolBench 把 `Finish` 放进 function 列表，因此停止不是独立的神经网络 head，而是普通动作空间的一部分：

$$
\mathcal A=\mathcal A_{tools}\cup
\{\text{Finish/give_answer},\text{Finish/give_up_and_restart}\}
$$

成功结束：

```json
{
  "return_type": "give_answer",
  "final_answer": "..."
}
```

放弃当前路径：

```json
{
  "return_type": "give_up_and_restart"
}
```

这意味着动作 Policy 同时要学习：

```text
何时继续调用工具
何时已有足够信息
何时输出最终答案
何时当前分支已经无解
```

但是调用 `Finish/give_answer` 只表示模型声明完成，不表示答案已经被严格验证。ToolBench 的在线 `check_success()` 主要检查是否成功调用 `Finish`；实际答案质量由 episode 结束后的 ToolEval 判断。

因此：

$$
\text{Episode Termination}\neq\text{Answer Correctness}
$$

#### 4.4 Single Chain：单轨 ReAct Policy

实现：

```text
toolbench/inference/Algorithms/single_chain.py
```

单次链条的循环是：

```text
当前消息状态
    ↓
模型生成一个动作
    ↓
立即执行工具
    ↓
Observation 加入消息历史
    ↓
模型生成下一个动作
    ↓
Finish / prune / depth limit
```

形式上：

$$
a_t\sim\pi_\theta(a\mid o_t),\qquad
S_{t+1}=T(S_t,a_t)
$$

它在同一状态下不会先比较多个候选，而是接受一次采样并立即向前执行。因此它属于典型的反应式 ReAct Policy。

`CoT@N` 表示最多运行 $N$ 次相互独立的单链，而不是在每个节点生成 $N$ 个候选：

```text
尝试 1：A → B → C → 失败
尝试 2：D → E → Finish → 成功
```

如果单次成功概率是 $p$，理想化情况下至少一次成功的概率是：

$$
P(\text{success})=1-(1-p)^N
$$

它的优点是简单；缺点是单次尝试没有前瞻、候选比较和局部回溯能力。

#### 4.5 DFS/DFSDT：用搜索控制器扩展动作 Policy

实现：

```text
toolbench/inference/Algorithms/DFS.py
```

ToolLLaMA 只负责提出候选，DFS 负责：

```text
生成多少候选
按什么顺序展开
何时进入下一层
何时回溯
何时剪枝
何时停止整棵搜索树
```

假设当前 State 下搜索宽度为 2，第一次采样得到：

```text
候选 A：
Action: search_hotel
Action Input: {"city": "Beijing"}
```

为了生成不同候选，DFS 会临时加入 `DIVERSITY_PROMPT`，其中包含已经尝试的 action、arguments 和 observation，再要求模型选择不同动作：

```text
之前已经尝试过 search_hotel({"city":"Beijing"})，
请提出不同的动作。
```

第二次可能得到：

```text
候选 B：
Action: search_location
Action Input: {"query": "Beijing downtown"}
```

所以它不是从模型 logits 中直接读取 Top-K，而是重复采样：

$$
a^{(1)}\sim\pi_\theta(\cdot\mid o_t)
$$

$$
a^{(2)}\sim\pi_\theta(\cdot\mid o_t,\text{avoid }a^{(1)})
$$

这个 diversity prompt 会被标记为临时、无效消息，不作为普通历史继续传播，也不应该成为后续训练轨迹中的真实用户消息。

##### `DFS_woFilter_wN`：无 Ranker 的 DFSDT

`wN` 表示每个节点最多尝试 $N$ 个候选；`woFilter` 表示不先调用 LLM 对候选排序。

近似过程：

```text
生成候选 A
└── 立即沿 A 深度探索
    ├── 成功：停止
    └── 失败：回溯

再生成或探索候选 B
```

这是深度优先控制策略：

$$
\mu_{DFSDT}=\text{sample}\rightarrow
\text{execute}\rightarrow
\text{recurse}\rightarrow
\text{backtrack}
$$

它没有独立 Value Model，第一个采样到的候选通常先被深入探索。搜索能够弥补动作模型的不稳定，但无法保证优先进入全局最优分支。

##### `DFS_wN`：带候选排序的 DFS

带 Filter 时，流程变为：

```text
先生成当前节点的 N 个候选
        ↓
LLM Ranker 两两比较
        ↓
按分数从高到低排序
        ↓
优先深入更有希望的候选
```

所以 ToolBench 的搜索 Policy 可以进一步拆成：

```text
Proposer：ToolLLaMA
Diversifier：DIVERSITY_PROMPT
Ranker：LLM pairwise comparison
Controller：DFS recursion/backtracking
```

#### 4.6 LLM Ranker：Value-like Policy，而不是真正的 Critic

实现：

```text
toolbench/inference/LLM_rank/rank_candidate.py
toolbench/inference/Prompts/rank_prompts.py
```

Ranker 找到两个候选的共同祖先，构造共同轨迹和候选 A/B 的后缀，然后询问 LLM：

```text
哪一条轨迹更接近解决任务？只回答 A 或 B。
```

为降低位置偏差，它会执行对称比较：

```text
第一次：A vs B
第二次：B vs A
```

多个候选时执行两两比较，胜者得 1 分，平局各得 0.5 分，再按总分排序。

功能上它近似：

$$
Q(S_t,a_t)\approx
\text{从该候选继续是否更可能完成任务}
$$

但它不是通过 rollout reward 训练得到的 Value Network，也没有输出经过校准的期望回报。因此更准确的名称是：

```text
LLM-as-a-Judge
LLM-based heuristic ranker
Value-like heuristic
```

候选数为 $N$ 时，对称两两比较大约需要：

$$
2\binom{N}{2}
$$

次额外模型调用。因此宽度增加会快速增加成本。

还要注意一个实现细节：Rank Prompt 希望模型只输出 `A/B`，但本地 ToolLLaMA 的 `parse()` 默认走 ReAct 解析。它没有一个完全独立、严格类型化的 ranking 接口，因此本地模型配合 filtered DFS 时可能较脆弱。官方实验中常见的 `DFS_woFilter_w2` 可以绕开这层排序。

#### 4.7 Environment 和状态码构成控制 Policy

动作模型并不独立决定下一步。`rapidapi_wrapper.step()` 返回的 status 会约束搜索控制：

| status | 含义 | 控制行为 |
|--------|------|----------|
| 0 | 正常工具结果 | Observation 写回历史，继续 |
| 1 | 幻觉函数名 | 写回错误，让模型尝试修正 |
| 2 | 参数错误 | 写回参数反馈，允许重试 |
| 3 | `Finish/give_answer` | terminal success |
| 4 | `Finish/give_up_and_restart` | prune 当前分支并回溯 |
| 5–12 | API、网络、授权、限流等错误 | 作为 Observation 或错误控制信号处理 |

完整的停止条件包括：

```text
调用 Finish/give_answer
调用 Finish/give_up_and_restart
达到最大树深度
达到最大模型 query 数
当前分支被剪枝
已找到要求数量的答案
所有候选均探索完毕
```

因此，停止策略并不是只由模型学习出来的：

```text
模型学习何时提出 Finish
环境解释 Finish 和工具状态码
DFS 根据 terminal/pruned 进行回溯或退出
Runtime 根据深度与预算强制停止
```

#### 4.8 ToolBench Policy 哪些是学出来的，哪些是人工规则

| 层次 | 是否训练 | 具体内容 |
|------|----------|----------|
| 动作生成 $\pi_\theta$ | 是 | 根据历史和 functions 生成 Thought/Action/Input |
| 输出格式约束 | 主要不是 | ReAct Prompt、function schema、字符串 parser |
| 候选多样性 | 不是 | 重复采样和 `DIVERSITY_PROMPT` |
| DFS 展开顺序 | 不是 | Python 递归和固定搜索规则 |
| 宽度、深度、回溯长度 | 不是 | 命令行参数和硬编码配置 |
| 候选 Ranker | 通常不专门训练 | 使用 LLM 做 A/B 启发式判断 |
| 工具执行 | 不是 | `rapidapi_wrapper`、server/RapidAPI |
| 环境状态码 | 不是 | 人工定义的控制协议 |
| 在线完成验证 | 很弱 | 主要检查是否调用 `Finish/give_answer` |
| 最终质量判断 | 不更新 Policy | ToolEval 离线评测 |

因此，ToolBench 的广义 Policy 不是一个端到端学习系统，而是：

```text
学习到的局部动作 Policy
+ Prompt 约束
+ 人工搜索算法
+ 可选的 LLM 排序启发式
+ 环境转移和停止规则
```

#### 4.9 ToolLLaMA Policy 的训练本质

预处理：

```text
preprocess/preprocess_toolllama_data.py
```

训练入口：

```text
toolbench/train/train.py
toolbench/train/train_mem.py
toolbench/train/train_lora.py
```

成功轨迹：

```text
User
Assistant Action 1
Function Result 1
Assistant Action 2
Function Result 2
Assistant Finish
```

会被拆成多个前缀监督样本：

```text
样本 1：User → Action 1

样本 2：
User + Action 1 + Result 1 → Action 2

样本 3：
User + Action 1 + Result 1 + Action 2 + Result 2 → Finish
```

即构造：

$$
D=\{(o_t,a_t^*)\}
$$

训练目标近似为：

$$
\mathcal L_{SFT}=-\sum_t\log\pi_\theta(a_t^*\mid o_t)
$$

训练时历史部分主要充当条件，只对目标 Assistant 回复计算 loss。这属于：

```text
Supervised Fine-Tuning
Behavior Cloning
Next-action imitation
```

它不等于完整强化学习，因为当前实现没有：

```text
可学习的 reward model
advantage
policy gradient
critic/value network
在线 rollout 后更新参数
```

`rapidapi_wrapper.get_score()` 固定返回 `0.0` 也说明推理环境没有形成真正的在线 RL reward 闭环。

DFS 本身没有被训练。它只在推理时通过多采样和回溯补偿 $\pi_\theta$ 的局部错误。成功搜索轨迹再转成 SFT 数据时，才会间接把搜索发现的行为蒸馏进模型。

#### 4.10 ToolBench 是不是标准 Agent Policy

需要分三个层次回答。

##### 从 Agent 基本循环看：是

ToolBench 符合经典循环：

$$
S_t\xrightarrow{\pi}A_t
\xrightarrow{Environment}O_{t+1}
\xrightarrow{Update}S_{t+1}
$$

它具备：

```text
状态输入
动作空间
工具执行
环境反馈
循环决策
主动终止
```

##### 从强化学习定义看：不完全是

ToolLLaMA 可以被看成参数化动作策略，但整体主要通过成功轨迹 SFT 学习，而不是通过 reward、advantage 和 policy gradient 做端到端策略优化。

##### 从现代 Agent 工程看：不是完整标准实现

当前 Agent 领域不存在所有框架共用的唯一 Policy 标准，但现代工程通常强调：

```text
显式结构化 State
类型化 Decision/Action
Policy 与副作用执行分离
统一 Reducer/Event 更新
在线 Verifier
权限和审批边界
Checkpoint 与恢复
Tracing 与审计
```

ToolBench 更像研究型、搜索增强的 ReAct Agent：非常适合研究工具学习、搜索、轨迹生成和自动评测，但如果用于生产，还需要补充上述运行时能力。

#### 4.11 现代 Agent Policy 的统一抽象

现代设计的核心变化，是不再让 LLM 同时隐式承担记忆、决策、执行、验证和终止，而是明确分层：

$$
\boxed{
\text{State 保存事实}
\quad
\text{Policy 产生决策}
\quad
\text{Runtime 执行动作}
\quad
\text{Reducer 更新状态}
\quad
\text{Verifier 判断结果}
}
$$

完整循环：

```text
完整 Agent State S_t
    ↓ Context Builder：o_t = φ(S_t)
模型可见 Observation o_t
    ↓ Policy
类型化 Decision d_t
    ↓ Validator / Guardrail
允许、拒绝、修复或请求审批
    ↓ Executor
Result Event e_t
    ↓ Reducer
新 State S_t+1
    ↓ Verifier / Router
继续、重试、重新规划、handoff 或完成
```

狭义 Policy：

$$
d_t=\pi_\theta(o_t)
$$

广义控制系统可以拆成：

$$
\Pi=(
\pi_{route},
\pi_{plan},
\pi_{act},
\pi_{verify},
\pi_{recover},
\pi_{stop})
$$

现代 Action 一般使用类型化联合，而不是自由字符串：

```text
Decision
├── ToolCall
├── FinalAnswer
├── Replan
├── Handoff
├── RequestApproval
├── AskUser
├── Wait
├── Retry
└── Abort
```

例如：

```json
{
  "type": "tool_call",
  "tool": "cancel_order",
  "arguments": {"order_id": "A1024"},
  "reason": "用户要求取消尚未发货的订单"
}
```

Policy 只提出动作意图，不能在内部偷偷执行副作用：

```python
decision = policy.decide(observation)
validation = guardrail.validate(state, decision)
event = executor.execute(decision)
new_state = reducer.apply(state, event)
verdict = verifier.check(new_state)
```

关键原则是：

> LLM 可以建议动作，但 Runtime 才拥有执行权；LLM 可以声明完成，但 Verifier 和终止规则才决定任务是否真正完成。

#### 4.12 现代架构一：ReAct / Tool-Calling Policy

##### 决策结构

```text
State → LLM → ToolCall → Observation → State → LLM
```

对应：

$$
a_t\sim\pi_\theta(a\mid\phi(S_t))
$$

推荐 State：

```python
class ReActState:
    task
    messages
    available_tools
    action_history
    observations
    known_facts
    last_error
    retry_count
    remaining_budget
    status
```

##### 实际例子：查询订单物流

用户：

```text
帮我查订单 A1024 到哪里了。
```

第一步 Policy：

```json
{
  "type": "tool_call",
  "tool": "get_order",
  "arguments": {"order_id": "A1024"}
}
```

工具返回：

```json
{"shipment_id": "SF9382", "status": "shipped"}
```

Reducer 把 `shipment_id` 写入 State。第二步 Policy：

```json
{
  "type": "tool_call",
  "tool": "get_shipment",
  "arguments": {"shipment_id": "SF9382"}
}
```

工具返回上海转运中心和预计送达时间，Policy 最后输出 `FinalAnswer`。

如果 State 不保存 `shipment_id`，下一步 Policy 就缺少必要决策信息；如果不保存 retry count，错误时就可能重复调用同一参数。

##### 训练思路

1. **SFT/行为克隆**：使用 $(o_t,a_t^*)$ 训练下一工具、参数和停止动作；
2. **工具自生成数据**：采样工具调用，真实执行，只保留能改善结果的样本；
3. **偏好优化**：构造正确调用与错误、重复、昂贵调用的 chosen/rejected；
4. **可验证环境中的 RL**：用任务成功减去工具次数、延迟和非法调用作为 reward。

优先顺序通常是：先把 Action schema、错误 Observation 和评测做好，再做 SFT；有稳定偏好数据再做 DPO；只有 reward 可靠且环境可重复时才考虑 RL。

#### 4.13 现代架构二：Planner–Executor–Replanner Policy

##### 决策结构

```text
Planner 生成结构化计划
        ↓
Executor 执行当前步骤
        ↓
Verifier 检查步骤结果
    ↙              ↘
继续             Replanner 修改计划
```

可以表示为：

$$
\Pi=(\pi_{plan},\pi_{execute},\pi_{replan})
$$

推荐 State：

```python
class PlanningState:
    task
    constraints
    plan_version
    plan_steps
    current_step_id
    completed_steps
    blocked_steps
    facts
    artifacts
    errors
    replan_count
    status
```

计划步骤应包含：

```json
{
  "id": "step-2",
  "goal": "查询目的地天气",
  "depends_on": ["step-1"],
  "status": "pending",
  "expected_output": {
    "date": "string",
    "weather": "string"
  }
}
```

##### 实际例子：上海三日旅行方案

用户要求在 3000 元预算内规划行程，并考虑天气、酒店位置和景点开放时间。

Planner 生成：

```text
1. 明确日期和预算
2. 查询三天天气
3. 查询景点开放时间
4. 搜索交通方便的酒店区域
5. 生成每日路线
6. 汇总费用并校验预算
```

执行时发现原定周一参观的博物馆闭馆。Verifier 产生：

```json
{
  "type": "step_rejected",
  "step_id": "step-5",
  "reason": "目标景点周一闭馆"
}
```

Replanner 读取已经确认的天气、酒店区域和开放时间，只调整受影响的行程，而不是从头丢弃全部 State。

##### 训练思路

1. **Planner SFT**：任务和约束 $\rightarrow$ 带依赖、产物、风险和验证方式的结构化计划；
2. **Executor SFT**：当前 State + 当前 plan step $\rightarrow$ 下一工具动作；
3. **Replanner SFT**：原计划 + 已完成步骤 + 新冲突 $\rightarrow$ 局部修订计划；
4. **计划偏好优化**：偏好可执行、依赖正确、成本适中且可验证的计划；
5. **步骤级 reward + 终局 reward**：缓解只在最终成功时给分造成的长程信用分配困难。

Replanner 数据要主动包含 API 超时、数据缺失、预算变化、用户改需求和步骤验证失败，不能只有理想成功路径。

#### 4.14 现代架构三：Graph / Workflow Policy

##### 决策结构

```text
Receive → Extract → Validate → Approval → Execute → Verify
                         ↘ Reject
```

Graph Policy 把控制流显式写成节点和条件边：

- Node 读取 State，执行 LLM、代码或工具；
- Edge 根据 State 选择下一节点；
- 确定性规则处理权限、金额、状态机和不可违反的业务约束；
- LLM 只承担模糊语义判断。

推荐 State：

```python
class WorkflowState:
    request
    extracted_fields
    phase
    eligibility
    pending_approval
    approval_result
    execution_result
    verification_result
    status
```

##### 实际例子：员工费用报销

```text
解析发票
  ↓
检查金额、日期和发票号
  ↓
检查报销政策
  ↓
金额是否超过 5000？
  ├── 否：自动提交
  └── 是：等待经理审批 → 提交财务
```

LLM 可以提取发票字段、分类消费用途、生成解释；代码负责金额比较、发票号去重、审批权限和最终提交。

即使 LLM 说“建议批准”，只要：

```json
{"manager_approval": null}
```

Graph 就不能进入支付节点。

##### 训练思路

Graph 的关键控制流通常**不训练**：

```python
if amount > 5000:
    next_node = "manager_approval"
```

可训练部分包括：

1. 发票文本到结构化字段的抽取模型；
2. travel/meal/equipment 等语义分类 Router；
3. 模糊政策解释节点；
4. 低置信度转人工的分类阈值。

原则是：

> 确定性业务规则写成代码和状态机；只有模糊语义判断才交给可训练 Policy。

#### 4.15 现代架构四：Supervisor–Worker Policy

##### 决策结构

Supervisor 负责分解和路由：

$$
w_t\sim\pi_{route}(w\mid S_t)
$$

Worker 在局部状态下执行：

$$
a_t\sim\pi_w(a\mid S_t^{(w)})
$$

推荐 State：

```python
class MultiAgentState:
    task
    shared_facts
    assignments
    active_agent
    agent_results
    handoff_history
    pending_tasks
    ownership
    conflicts
    budget_by_agent
    status
```

Handoff 不应只发一句“请继续”，而应包含：

```json
{
  "assignee": "database_agent",
  "goal": "检查最近 24 小时数据库错误",
  "known_facts": ["API 从 10:15 开始出现 500"],
  "allowed_tools": ["query_database_metrics", "read_database_logs"],
  "required_output": {
    "root_cause_candidates": "array",
    "evidence": "array"
  }
}
```

##### 实际例子：线上支付服务故障排查

Supervisor 分派：

```text
日志 Agent：检查错误日志
数据库 Agent：检查连接数和慢查询
部署 Agent：检查最近发布
监控 Agent：检查 CPU、内存和网络
```

返回结果：

```text
日志：大量 database connection timeout
数据库：连接池使用率 100%
部署：错误前 3 分钟的新版本把连接池从 100 改为 20
```

Supervisor 汇总证据，提出回滚；高风险回滚再进入带审批的 Workflow，而不是由任何 Worker 直接执行。

##### 训练思路

1. **Router 监督训练**：任务 + Worker 描述 $\rightarrow$ 最合适 Worker；
2. **对比或偏好训练**：最佳 Worker 对比“看似相关但能力或权限不匹配”的 Worker；
3. **Worker 专项 SFT**：每个 Worker 在自己的工具和任务分布上训练；
4. **Handoff SFT**：学习传递必要事实、约束、工具权限和期望产物；
5. **团队级优化**：用成功率减去 handoff 次数、重复工作、延迟和 token 成本。

多 Agent 的终局奖励存在明显信用分配问题，所以工程上更常先分别训练 Router 和 Worker，再做团队级离线评测，而不是一开始就做端到端多 Agent RL。

#### 4.16 现代架构五：Proposer–Verifier Policy

##### 决策结构

```text
Proposer 生成候选
        ↓
Verifier 检查候选
    ↙             ↘
失败并反馈       通过并输出
    ↓
Proposer 修订
```

形式上：

$$
c_t\sim\pi_{propose}(c\mid S_t),\qquad
v_t=V(S_t,c_t)
$$

推荐 State：

```python
class VerificationState:
    task
    candidate
    evidence
    test_results
    verifier_feedback
    revision_count
    accepted
    status
```

##### 实际例子：生成 SQL

用户要求统计过去 30 天每个城市的付费用户数。Proposer 首次生成：

```sql
SELECT city, COUNT(DISTINCT user_id)
FROM orders
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY city;
```

Verifier 检查：

```text
SQL 是否可解析
表和字段是否存在
是否只统计 paid 订单
时间边界是否正确
是否是只读查询
查询成本是否可接受
```

发现缺少 `status = 'paid'`，返回结构化反馈。Proposer 修订后再次验证，通过才允许正式执行。

##### 训练思路

1. **Proposer SFT**：使用正确 SQL、补丁或结构化答案；
2. **Verifier 监督训练**：$(S,c,y,feedback)$，学习正确性标签和具体错误；
3. **Pairwise Reward Model**：学习 $V(S,c^+)>V(S,c^-)$；
4. **Rejection Sampling**：生成多个候选，运行测试，只保留通过候选；
5. **DPO**：通过验证的候选作为 chosen，失败候选作为 rejected；
6. **Process Reward**：对读取 schema、检查字段、使用只读验证等中间步骤分别给分。

验证信号的推荐优先级：

```text
确定性测试
  > 外部权威状态回查
  > 业务规则
  > LLM Verifier
```

同一个模型生成并自我判断容易产生相关性错误，因此代码、SQL 和 API 任务应尽量使用真实执行结果作为 verifier signal。

#### 4.17 现代架构六：Search / Tree Policy

##### 决策结构

Search Policy 在同一 State 下生成多个候选：

$$
a_t^{(1)},a_t^{(2)},\ldots,a_t^{(k)}
$$

再使用：

$$
Q(S_t,a_t^{(i)})
$$

选择展开顺序。常见控制器包括 Best-of-N、Beam Search、DFS、BFS、Tree of Thoughts 和 MCTS。

推荐 State：

```python
class SearchNode:
    node_id
    parent_id
    state_snapshot
    proposed_action
    observation
    prior_score
    value_score
    visit_count
    cumulative_reward
    depth
    status

class SearchState:
    root
    frontier
    terminal_nodes
    best_node
    total_queries
    max_depth
    max_width
    budget
```

##### 实际例子：自动修复代码 Bug

测试显示折扣计算错误，Proposer 生成：

```text
A：修正 calculate_discount 的边界判断
B：修改测试预期值
C：修正订单初始化逻辑
```

Evaluator 认为 A 最有希望，先生成补丁并运行测试。若局部测试通过但完整测试失败，State 中记录测试结果，该分支分数下降，控制器回溯到 A 的其他补丁或候选 C。

最终找到全部测试通过的补丁后，Verifier 才把节点标为 terminal success。

##### 训练思路

1. **Proposer SFT**：训练搜索节点上的高质量、多样候选；
2. **Value/Q Model**：用 rollout 结果训练 $V(S)$ 或 $Q(S,a)$；
3. **Monte Carlo Return**：将终局成功回传给中间节点；
4. **Pairwise Ranker**：训练同一父状态下哪个分支更好；
5. **搜索蒸馏**：用大预算搜索找到成功轨迹，再转成逐步 SFT 数据；
6. **策略迭代**：当前 Policy 搜索、Verifier 选优、用更优轨迹更新 Policy，再重新搜索。

搜索本身是推理算法，不训练也能工作。训练的重点通常是 Proposer 和 Value/Ranker，而不是把 DFS 递归代码变成神经网络。

#### 4.18 现代架构七：Hierarchical Skill Policy

##### 决策结构

当动作空间很大时，先选择 Goal/Skill，再选择具体工具：

$$
z_t\sim\pi_{high}(z\mid S_t)
$$

$$
a_t\sim\pi_{low}(a\mid S_t,z_t)
$$

即：

```text
Goal → Skill → Tool → Arguments
```

推荐 State：

```python
class HierarchicalState:
    task
    current_goal
    selected_skill
    skill_state
    available_tools
    retrieved_tools
    skill_result
    status
```

##### 实际例子：企业运维 Agent

任务是诊断支付服务并在确认安全后恢复。高层 Policy 依次选择：

```text
incident_diagnosis
deployment_analysis
rollback_service
post_recovery_validation
```

进入 `incident_diagnosis` 时只开放读日志、指标和 trace 工具；进入 `rollback_service` 时才开放生成回滚计划、申请审批和执行回滚的工具。

这能缩小动作空间，也能把权限限制到当前 Skill。ToolBench 的 API Retriever 与此有相似之处：先从大工具库中检索候选 API，再由 ToolLLaMA 选择具体动作。不过 ToolBench 通常只在 episode 开始前检索一次，不是完整的动态 Skill Policy。

##### 训练思路

1. **高层 Skill Router**：任务和 State $\rightarrow$ Skill；
2. **低层 Tool Policy**：每个 Skill 内训练局部工具轨迹；
3. **Retriever 训练**：任务–工具相关性、成功轨迹实际使用的工具；
4. **分层 reward**：高层奖励子目标完成，低层奖励工具成功、参数正确和低调用成本。

分层 Policy 可以缓解巨大动作空间和长程信用分配，但必须在 State 中明确保存当前 Goal、Skill、局部进度和权限，不能只依赖 Prompt 中一句自然语言。

#### 4.19 七种现代 Policy 架构的整体对比

这些架构不是互斥分类。生产系统常采用外层 Workflow、中层 Planner、局部 ReAct、关键节点 Verifier、复杂节点 Search、多领域任务 Supervisor 的组合。

| Policy 架构 | 核心决策 | 关键 State | 典型任务 | 主要训练对象 |
|-------------|----------|------------|----------|--------------|
| ReAct/Tool Calling | 下一步调用什么工具 | 消息、事实、工具结果、错误、预算 | 短链查询和操作 | 动作模型 |
| Planner–Executor | 先拆什么步骤、当前执行什么 | 计划、依赖、步骤进度、产物 | 长任务、多步骤任务 | Planner、Executor、Replanner |
| Graph/Workflow | 当前进入哪个节点 | phase、业务字段、审批状态 | 稳定流程、高风险操作 | 通常只训练语义节点 |
| Supervisor–Worker | 把任务交给谁 | assignments、ownership、handoff | 多领域复杂任务 | Router、Supervisor、Workers |
| Proposer–Verifier | 候选是否正确 | candidate、证据、测试、反馈 | 代码、SQL、严谨输出 | Proposer、Verifier |
| Search/Tree | 展开哪个候选分支 | 搜索树、分数、分支预算 | 复杂推理和工具搜索 | Proposer、Value/Ranker |
| Hierarchical Skill | 先选什么能力，再选什么动作 | Goal、Skill、局部状态、权限 | 大工具集、长程任务 | Skill Router、局部 Policy |

#### 4.20 Policy 训练数据的六种基本单位

##### Step 数据

```text
当前 Observation → 下一 Action
```

$$
D_{step}=\{(o_t,a_t^*)\}
$$

适用于 ReAct、Executor、Worker 和低层 Tool Policy，主要使用 SFT。

##### Plan 数据

```text
任务和约束 → 结构化计划
```

$$
D_{plan}=\{(x,p^*)\}
$$

适用于 Planner、Task Decomposer 和 Replanner。

##### Preference 数据

```text
同一 State 下，Action/Plan/Candidate A 优于 B
```

$$
D_{pref}=\{(o,a^+,a^-)\}
$$

适用于 DPO、Reward Model、候选排序和路由偏好。

##### Verification 数据

```text
State + Candidate → 标签、分数和错误反馈
```

$$
D_{verify}=\{(S,c,y,feedback)\}
$$

适用于 Verifier、Critic 和 Value Model。

##### Trajectory 数据

```text
S0, A0, O1, S1, A1, O2, ..., Result
```

$$
\tau=(S_0,a_0,r_0,\ldots,S_T,R)
$$

适用于行为克隆、搜索蒸馏、错误恢复和 RL。

##### Routing 数据

```text
任务和 State → Agent / Skill / Workflow
```

$$
D_{route}=\{(x,S,z^*)\}
$$

适用于 Supervisor、Skill Router、Intent Router 和 Tool Retriever。

#### 4.21 常见训练策略应如何选择

| 已有条件 | 优先策略 |
|----------|----------|
| 没有专用训练数据 | Prompt、Action Schema、Graph、评测先行 |
| 有成功执行轨迹 | SFT/行为克隆 |
| 有成对好坏候选，但没有稳定标量奖励 | DPO |
| 有自动测试或确定性结果 | Rejection Sampling、Verifier、RL |
| 最终奖励稀疏 | 步骤标签、Process Reward、课程学习 |
| 搜索成功但推理成本高 | 搜索蒸馏 |
| 路由标签清晰 | 分类/SFT，一般不必先做 RL |
| 涉及付款、删除、外发消息 | 硬规则、权限和审批，不能只靠训练 |
| 环境稳定且可重复模拟 | 可以考虑在线或离线 RL |
| 工具结果随机、评测不可靠 | 先改善 State、日志和 Eval，不要急于 RL |

推荐的训练演进顺序：

```text
1. 先定义结构化 State、Action 和 Eval
2. 用强模型、人工或搜索生成成功轨迹
3. 对动作 Policy 做 SFT
4. 收集失败轨迹与候选偏好
5. 训练 Verifier/Ranker 或做 DPO
6. 用搜索产生更优轨迹并蒸馏
7. 只有 reward 可靠时再做 RL
```

通用任务 reward 可以抽象为：

$$
R=R_{success}
-\lambda_1 C_{tool}
-\lambda_2 C_{latency}
-\lambda_3 C_{invalid}
-\lambda_4 C_{human}
$$

但付款、删除、权限等安全边界不能只作为一个负 reward；它们必须先由确定性 Guardrail 阻止未经授权的动作。

#### 4.22 State 设计如何决定 Policy 能否训练好

Policy 的训练样本不是抽象的“问题 → 答案”，而是：

$$
o_t=\phi(S_t)\rightarrow d_t
$$

因此 State 缺字段，Policy 就缺少决策依据：

| State 缺失内容 | Policy 可能出现的问题 |
|----------------|----------------------|
| 工具失败次数 | 无限重复同一失败调用 |
| 当前 plan step | 做了正确动作，但不是当前需要的动作 |
| 用户审批状态 | 未经授权执行高风险操作 |
| 事实来源和证据 | 无法验证最终答案 |
| 已用预算 | 不知道何时停止搜索 |
| 当前 owner/agent | 多 Agent 重复工作或互相覆盖 |
| action id / idempotency key | 重试时重复产生副作用 |
| 状态版本 | 并发执行覆盖更新 |

训练与推理还必须保持以下协议一致：

```text
相同 State Schema
相同 Context Builder
相同工具描述格式
相同 Action Schema
相同错误表示
相同终止条件
```

否则就会出现 train–inference distribution shift。例如训练数据把错误写成自然语言，推理时却只给数字状态码，模型就不一定能学会正确恢复。

完整 State 也不应全部放进 prompt。现代系统通常区分：

```text
Environment State：外部世界的真实状态
Agent State：当前任务的结构化事实和进度
Model Context：从 Agent State 投影出的有限 Observation
Memory：跨任务保存并按需检索的信息
Runtime Context：工具连接、凭据和不可序列化依赖
```

Policy 只读取经过 Context Builder 筛选的 Observation，权限、密钥、审计和执行控制仍保留在 Runtime 私有状态中。

#### 4.23 Reducer/Event：让 Policy 不直接篡改 State

现代实现倾向让 Policy 产生 Decision，让 Executor 产生 Event，再由 Reducer 更新 State：

```text
Policy Decision
    ↓
ToolCallRequested
    ↓ Executor
ToolCallSucceeded / ToolCallFailed
    ↓ Reducer
New Agent State
```

例如：

```python
def reduce(state, event):
    if isinstance(event, ToolCallSucceeded):
        state.observations.append(event.output)
        state.pending_action = None
    elif isinstance(event, PlanStepCompleted):
        state.plan[event.step_id].status = "completed"
    return state
```

这使轨迹天然可以转成训练数据：

```text
State Snapshot + Decision + Result Event + Verification
```

同时支持：

```text
审计
重放
Checkpoint
暂停与恢复
并发控制
失败归因
离线 Policy 训练
```

ToolBench 目前主要通过直接创建 `tree_node`、追加 messages、复制 `io_state` 和修改环境字段更新状态，没有统一 Reducer。研究实验足够直观，但难以直接扩展成持久化生产 Runtime。

#### 4.24 用现代架构重新表达 ToolBench

| 现代概念 | ToolBench 对应实现 |
|----------|--------------------|
| Agent State | `tree_node`、messages、`rapidapi_wrapper`、DFS 全局字段 |
| Context Builder | 从根到当前节点收集 messages，再加入 functions |
| Action Policy | `ToolLLaMA.parse()` |
| Decision | Thought、Action、Action Input |
| Planner | 没有独立 Planner，局部计划包含在 Thought 中 |
| Search Controller | Single Chain、DFS、DFSDT |
| Candidate Evaluator | LLM Ranker |
| Executor | `rapidapi_wrapper.step()` + API 服务 |
| Reducer | 没有统一对象，由节点创建和环境更新共同完成 |
| Recovery Policy | 错误 Observation、重采样、DFS 回溯 |
| Termination Policy | `Finish`、状态码、深度和 query budget |
| Online Verifier | 较弱，主要检查模型是否声明完成 |
| Offline Evaluator | ToolEval |
| Persistence | 保存轨迹 JSON，但不是完整可恢复 Runtime |

如果把 ToolBench 现代化，可以重构为：

```text
显式 AgentState
    ↓
ContextBuilder
    ↓
ToolLLaMAPolicy
    ↓
ActionValidator / Guardrail
    ↓
ToolExecutor
    ↓
StateReducer + EventLog
    ↓
OnlineVerifier
    ↓
Search / Replan / Approval / Finish
```

但应保留 ToolBench 已经做对的分层思想：

```text
Retriever 缩小动作空间
ToolLLaMA 提出局部动作
DFS 负责搜索而不是混进模型代码
Environment 统一工具反馈
ToolEval 独立评估最终结果
```

#### 4.25 Policy 设计检查表

设计一套现代 Agent Policy 时，可以逐项检查：

```text
Policy 看到的是完整 State，还是经过定义的 Observation？
Action 是否有严格 schema，而不是依赖字符串切割？
模型是否只提出动作，而不是绕过 Runtime 直接执行副作用？
工具调用前是否检查权限、参数和审批？
工具返回后是否通过统一 Reducer 更新 State？
是否记录 retry、budget、action id 和错误类型？
模型声明完成后是否有独立 Verifier？
失败时是 retry、repair、replan、handoff 还是 abort？
多 Agent handoff 是否传递结构化目标和最小必要事实？
搜索分支是否有独立 State、预算和环境快照？
训练数据的 State/Action/Error schema 是否和推理一致？
哪些规则应该训练，哪些必须确定性编码？
是否能从 Event Log 重建一次 Policy 决策的原因？
是否能暂停、恢复和安全重放？
```

#### 4.26 Policy 章节总结

ToolBench 的 Policy 可以概括为：

$$
\boxed{
\text{SFT 动作生成器}
+\text{Prompt 约束}
+\text{DFS/DFSDT 搜索}
+\text{可选 LLM 排序}
+\text{环境与终止规则}
}
$$

现代 Agent Policy 则进一步强调：

```text
State 是事实
Context 是 State 的受控投影
Policy 产生类型化决策
Runtime 执行副作用
Reducer 统一更新状态
Verifier 判断是否满足目标
Recovery Policy 决定重试、重规划或 handoff
确定性安全规则不交给模型自由学习
```

不同架构主要训练的内容也不同：

```text
ReAct：训练下一步做什么
Planner–Executor：训练怎么拆、当前怎么执行、失败怎么改计划
Graph/Workflow：控制流主要写代码，只训练语义节点
Supervisor–Worker：训练交给谁做、怎样 handoff、Worker 怎样完成局部任务
Proposer–Verifier：训练什么候选更正确、怎样验证和修订
Search/Tree：训练怎样提出候选、哪个分支值得探索
Hierarchical Skill：训练先选什么能力，再选择什么具体动作
```

因此，最准确的结论是：

> ToolBench 使用了标准的 Agent 决策循环和典型的搜索增强 ReAct 思想，但它不是统一、类型化、可持久化的现代 Policy Runtime。它真正学到的是局部下一动作，完整行为则由模型、搜索、环境和人工规则共同决定。

参考资料：

- [ReAct：Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Toolformer：Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [Tree of Thoughts：Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [LangGraph：Graph State 与 Reducer](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [OpenAI Agents SDK：Running Agents](https://openai.github.io/openai-agents-python/running_agents/)
- [OpenAI Agents SDK：Guardrails](https://openai.github.io/openai-agents-python/guardrails/)
- [AutoGen：Teams](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/teams.html)
- [AutoGen：Managing State](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/state.html)

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

#### 5.1 第一层补充：ToolBench 当前 Executor 职责分解

step() 负责调用 _step() 并限制 Observation 长度；_step() 负责：

~~~text
识别 Finish
解析和校验 Finish arguments
把 function name 映射回 category/tool/api
构造 tool payload
选择本地或远程执行路径
将后端错误映射为 status code
返回 observation + status
~~~

映射状态主要保存在：

~~~text
api_name_reflect
tool_names
cate_names
~~~

最终 payload 包含：

~~~text
category
tool_name
api_name
tool_input
strip
toolbench_key
rapidapi_key（可选）
~~~

更准确的职责划分是：

| 执行层次 | ToolBench 对应 |
|----------|----------------|
| Environment Adapter | rapidapi_wrapper |
| Tool Router | _step() 的名称映射和 payload 构造 |
| Tool Executor | server.py 或远程 ToolBench/RapidAPI 服务 |
| Result Normalizer | _step() 的错误映射和 JSON 序列化 |
| Termination Handler | _step() 对 Finish 的处理 |

#### 5.2 第二层：与现代 Agent Tool Executor 标准设计的异同

ToolBench 的合理设计：

~~~text
模型不直接执行 Python/API
统一 step 接口隔离模型和工具
工具 schema 与底层 API 路由分离
提供 timeout 和错误状态
支持远程后端与本地自定义 API
所有调用结果回到统一 Agent loop
~~~

现代生产系统通常还要求，而 ToolBench 当前缺少：

~~~text
Tool Registry 和版本管理
严格输入/输出 schema 校验
身份、租户、权限与 scope 检查
人类审批和风险策略
幂等键、去重和投递语义
并发、队列、取消和 deadline propagation
重试、指数退避、熔断、限流与 fallback
Sandbox、容器、网络和文件权限隔离
Secrets Provider，不把 key 混入 Agent State
完整 trace、audit、cost、latency 和 provenance
结果缓存与 freshness
副作用补偿或 Saga
工具健康检查和动态可用性
~~~

现代 Executor 是受策略控制的执行平面：

~~~text
Typed Action
    ↓
Validate
    ↓
Authorize
    ↓
Approval / Risk Gate
    ↓
Deduplicate / Idempotency
    ↓
Schedule / Execute
    ↓
Normalize Result
    ↓
Emit Event
~~~

#### 5.3 第三层：现代 Tool Executor 设计思路和优秀实例

##### Tool Registry + Adapter

推荐定义：

~~~python
class ToolDefinition:
    tool_id: str
    version: str
    input_schema: dict
    output_schema: dict
    risk_level: str
    required_scopes: list[str]
    timeout_seconds: int
    retry_policy: dict
    executor_ref: str
~~~

Runtime 通过 registry 解析 action.tool_id，再调用统一 adapter。这样比根据字符串后缀查找 API 更容易做版本、权限和测试。

##### 读操作与写操作采用不同策略

企业数据库 Agent：

~~~text
SELECT
→ 可自动执行
→ 只读凭据
→ 行数和超时限制

UPDATE / DELETE
→ 先生成影响预览
→ 要求审批
→ transaction
→ commit 前再次校验
→ 记录审计和补偿信息
~~~

##### 长任务采用异步 Job

视频转码：

~~~text
StartTranscode
→ JobAccepted(job_id)
→ Agent 进入 WAITING_TOOL
→ Poll/Webhook
→ JobCompleted(artifact_ref)
→ 从 checkpoint 恢复 Agent
~~~

Executor 必须支持暂停和恢复，而不是让一次模型调用同步等待很久。

##### 搜索与副作用执行隔离

旅行预订：

~~~text
Search / Planning
→ BookingProposal
→ Compare / Verify
→ User Approval
→ CommitBooking once
~~~

ToolBench 的 deepcopy 适合查询工具，但不能回滚付款、发送、删除等外部副作用。

##### Sandboxed Code Executor

代码 Agent 的命令执行器应限制：

~~~text
workspace root
network access
CPU / memory / time
allowed binaries
environment variables
secret exposure
filesystem mutation
~~~

结果以 Event 和 Artifact 返回，模型不应直接拥有无限制 shell。

#### 5.4 第四层：Executor 关联训练模式

Executor 本身主要是确定性 Runtime，应以工程测试为主，而不是用模型训练替代可靠性。可训练部分通常位于执行器前后：

| 可训练对象 | 训练任务 |
|------------|----------|
| Argument Generator | State → 合法参数 |
| Tool Router | State → tool_id |
| Retry/Recovery Policy | Error Event → retry/fallback/abort |
| Risk Classifier | Action → 风险类别；最终规则仍需硬约束 |
| Result Normalizer | 非结构化结果 → schema；必须验证 |
| Cost/Latency Predictor | Action → 预计成本和延迟 |
| Tool Success Predictor | State + Action → 成功概率 |

可靠 Executor 重点依赖：

~~~text
schema contract tests
mock/fake tool tests
timeout/retry tests
idempotency tests
permission tests
sandbox escape tests
fault injection
concurrency tests
replay tests
audit completeness tests
~~~

#### 5.5 Executor 训练和工程注意点

~~~text
模型不能持有或输出真实密钥
权限校验必须在 Executor 中重复执行
不要重试非幂等写操作，除非有幂等键
超时不等于未执行成功，需要查询最终状态
工具返回成功不等于任务目标完成
不要把 retry policy 完全交给 LLM
区分 transient 与 permanent error
每个结果必须绑定 action_id 和 attempt
高风险工具需要审批、dry-run 和影响预览
搜索分支不能任意执行不可逆副作用
本地自定义 Python 工具必须隔离运行
~~~

#### 5.6 Tool Executor 章节结论

ToolBench 当前：

~~~text
function_call
→ rapidapi_wrapper._step
→ name mapping / payload
→ local server or remote service
→ JSON response + status
~~~

现代化方向：

~~~text
Typed Action
→ Registry
→ Validate
→ Authorize
→ Approve
→ Idempotent Execute
→ Normalize
→ Event Log / Artifact
→ Reducer
~~~

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

#### 6.1 第一层补充：ToolBench 当前 Feedback 的四个时间尺度

ToolBench 中至少存在四种不同反馈：

| 时间尺度 | 实现 | 作用 |
|----------|------|------|
| Step Feedback | observation + status code | 提供事实并控制下一步 |
| Search Feedback | terminal/pruned、可选 LLM Ranker | 决定展开或回溯 |
| Terminal Check | Finish + check_success | 判断模型是否声明完成 |
| Offline Eval | ToolEval、Retriever NDCG | 比较完整 episode 或召回质量 |

推理结果先由 convert_answers.py 和 convert_to_answer_format.py 转换为统一 answer 与 ExecutionGraph 格式。

eval_pass_rate.py 主要检查：

~~~text
是否存在工具幻觉
任务是否可解
答案是否解决任务
最终是否通过
~~~

输出包含 query、solvable、available_tools、intermediate_steps、final_step、is_solved、pass_rate_label、reason 和 not_hallucinate。

eval_preference.py 比较 reference model 与 output model 的完整答案轨迹，并进行正反候选顺序比较，以减弱位置偏差。

带 filter 的 DFS 还可以用 LLM Ranker 比较候选分支。这是一种 value-like feedback，但不是来自真实环境 reward，也不是经过校准的 Value Model。

#### 6.2 第二层：与现代 Agent Evaluator/Feedback 标准设计的异同

ToolBench 的合理设计：

~~~text
Step Feedback 与离线评测分开
保留完整 Action/Observation 轨迹
同时评测 pass rate 和 pairwise preference
Retriever 使用独立检索指标
对工具幻觉做专项检查
Evaluator 与被评模型解耦
~~~

不足之处：

~~~text
在线完成判断主要相信 Finish
缺少确定性任务 Verifier
缺少系统化的逐步 Process Evaluation
没有统一 reward/score/feedback schema
LLM-as-Judge 存在随机性、位置偏差和自偏好
评测依赖较旧模型和 prompt 配置
pass/fail 难以定位失败发生在哪一步
没有系统覆盖成本、延迟、安全和副作用
没有按工具、错误类型和任务复杂度系统切片
没有明确 evaluator feedback 如何回流训练
~~~

现代 Evaluator Stack 通常分层：

~~~text
Deterministic Validators
    schema、测试、数据库约束、业务规则

Environment Verifier
    外部世界是否达到目标状态

Process Evaluator
    每一步是否合法、有效、基于证据

Outcome Evaluator
    最终答案正确性、完整性和可用性

Safety Evaluator
    权限、隐私、越权和副作用

Efficiency Evaluator
    工具次数、token、延迟和成本

LLM Judge
    难以规则化的语义判断

Human Review
    高风险、低置信度和抽样质检
~~~

原则是：能用确定性验证器解决的问题，不应只交给 LLM Judge。

#### 6.3 第三层：现代 Evaluator/Feedback 设计思路和优秀实例

##### 多层 Eval，而不是一个总分

SQL Agent 可以同时评测：

~~~text
Action Validity：SQL 是否可解析、是否只读
Execution：是否成功执行
Result Correctness：是否匹配标准结果
Safety：是否访问未授权表
Efficiency：扫描行数、延迟和工具次数
Final Answer：是否忠实总结查询结果
~~~

最终答案 Judge 不能替代这些确定性指标。

##### Online Verifier 与 Finish 解耦

文件修改 Agent 在模型声明完成后自动检查：

~~~text
目标文件是否存在
修改范围是否符合要求
单元测试是否通过
lint/type check 是否通过
是否修改无关文件
是否留下未处理错误
~~~

只有验证通过，State 才从 VERIFYING 转成 COMPLETED；否则生成结构化反馈并进入 REPAIRING。

##### Evaluator 输出可操作 Feedback

不要只返回 score=0，而应返回：

~~~json
{
  "verdict": "failed",
  "failure_stage": "tool_argument",
  "error_code": "MISSING_REQUIRED_FIELD",
  "evidence": ["action act_5 omitted order_id"],
  "repair_hint": "reuse order_id from observation obs_2",
  "retryable": true
}
~~~

这种反馈既能驱动在线修复，也能转换为训练标签。

##### 搜索分支 Value/Reward Model

搜索 Agent 可以训练：

$$
V(S_t)=P(\text{eventual success}\mid S_t)
$$

或者：

$$
Q(S_t,a_t)=\mathbb E[R\mid S_t,a_t]
$$

代码修复 Agent 可用测试通过率、剩余失败数、静态分析错误和变更风险训练 Ranker，比让通用 LLM 猜哪个分支更可靠。

##### 评测集必须切片

至少按以下维度报告：

~~~text
单工具 / 多工具
闭域 / 开放域
短链 / 长链
简单参数 / 复杂参数
正常工具 / 错误恢复
读操作 / 写操作
已知工具 / 未见工具
不同类别和语言
低风险 / 高风险
~~~

总 pass rate 会掩盖特定工具或错误恢复的系统性失败。

#### 6.4 第四层：Evaluator/Feedback 关联训练模式

| Feedback | 可形成的训练数据 | 训练方式 |
|----------|------------------|----------|
| 成功轨迹 | Observation → Next Action | SFT / 行为克隆 |
| 好坏候选 | State, chosen, rejected | DPO / Preference Learning |
| 步骤合法性 | State + Action → label/reason | Process Verifier |
| 终局成功 | Trajectory → return | RL / Offline RL / Value Model |
| 错误和修复 | Error Observation → Repair Action | Recovery SFT |
| 路由命中 | Query → relevant tools | Retriever/Router 训练 |
| 测试结果 | Candidate → passed/failed/evidence | Verifier / Ranker |
| 人工修改 | Before → After + rationale | SFT / Preference |

推荐训练闭环：

~~~text
运行 Agent
→ 记录完整 State/Action/Observation/Event
→ 多层 Evaluator 标注
→ 失败归因和数据切片
→ 选取高质量成功轨迹
→ 构造 SFT / Preference / Verification 数据
→ 更新 Policy / Retriever / Verifier
→ 固定评测集回归
→ 新版本灰度与在线监控
~~~

综合 reward 可以表示为：

$$
R=
R_{task\ success}
+\alpha R_{process}
-\lambda_1 C_{tool}
-\lambda_2 C_{latency}
-\lambda_3 C_{invalid}
-\lambda_4 C_{risk}
$$

但未经授权付款、删除生产数据、泄露隐私、绕过审批等安全边界必须由 Guardrail/Executor 硬性阻止，不能只设为负 reward。

#### 6.5 Evaluator 与 Feedback 注意点

##### LLM-as-Judge

~~~text
固定 evaluator 模型和版本
保存完整 judge prompt
多次采样并报告方差
交换候选顺序降低位置偏差
盲化模型名称
使用人工标注校准
检查 judge 与人类的一致率
防止被评答案中的 prompt injection
避免被评模型和 judge 的自偏好
区分任务不可解与 Agent 失败
保存 reason 和 evidence，而不仅是 label
~~~

##### 数据泄漏和过拟合

~~~text
训练轨迹不得混入固定测试集
搜索生成数据记录 generator 和版本
不要用测试集 judge feedback 反复调参
锁定评测工具和运行工具版本
对未见工具、未见类别单独测试
避免模型记住 query_id 或模板答案
~~~

##### 指标体系

建议同时报告：

~~~text
Task success / pass rate
Tool selection precision/recall/NDCG
Argument validity
Tool hallucination rate
Error recovery rate
Grounded final answer rate
Average tool calls
Token / latency / monetary cost
Approval and human-intervention rate
Safety violation rate
Success@budget
~~~

单个 pass rate 无法解释 Agent 为什么成功或失败。

#### 6.6 Evaluator 与 Feedback 章节结论

ToolBench 当前：

~~~text
Step：observation + status
Search：可选 LLM Ranker
Terminal：Finish/check_success
Retrieval：NDCG
Outcome：ToolEval pass rate + preference
~~~

现代化方向：

~~~text
Typed Events
→ Deterministic Validators
→ Online Verifier
→ Process + Outcome + Safety + Efficiency Eval
→ Structured Feedback
→ SFT / DPO / Verifier / Value / RL 数据闭环
~~~

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
