# Agent 训练项目记忆文档（8 周学习与实操路线图）

## 文档定位

这是一份面向个人长期项目使用的**记忆文档**，目标不是提供一次性建议，而是作为后续 8 周学习、实验、复盘和方向收束的统一参考基线。

文档用途：
- 作为 Agent 训练项目的长期上下文
- 作为每周学习与实操的执行手册
- 作为实验设计、方法选择和方向判断的决策依据
- 作为后续扩展到主动检索型 Agent、代码执行型 Agent 的基础方法论文档

---

# 1. 项目总目标

## 1.1 总体目标

在 8 周内，建立对 Agent 训练的系统化理解，并完成一个可运行、可评估、可诊断的小型 Agent 训练项目。

具体目标包括：
1. 理解 Agent 问题的统一抽象：状态、动作、反馈、评估。
2. 搭建一个可控的工具编排型 Agent sandbox。
3. 跑通 SFT 到 DPO 的最小训练闭环。
4. 建立 step-level evaluator 和 error taxonomy。
5. 从工具编排型 Agent 平滑过渡到主动检索型 Agent。
6. 为后续学习 PPO / GRPO 和更复杂 Agent 训练方法打基础。

## 1.2 项目边界

本阶段**不追求**：
- 通用开放世界 Agent
- 大规模在线 RL
- 多 Agent 协同系统
- 生产级工具系统集成
- 复杂长期记忆架构

本阶段**重点追求**：
- 可复现
- 可解释
- 可评估
- 可诊断
- 可逐步扩展

## 1.3 核心策略

先学工具编排型 Agent，再进入主动检索型 Agent。

原因：
- 工具编排型 Agent 的状态空间、动作空间和反馈信号最清晰。
- 更容易建立训练闭环、评测闭环和诊断闭环。
- 更适合先掌握 Agent 训练的基础范式。
- 主动检索型 Agent 的真正难点是长程策略优化，应建立在前者基础上。

---

# 2. 项目方法论总纲

## 2.1 Agent 问题统一抽象

### State
Agent 当前可见的全部信息，包括：
- 用户输入
- 历史上下文
- 工具 schema
- 工具执行结果
- 检索结果或环境观测

### Action
Agent 可执行的动作，包括：
- 选择工具
- 填写参数
- 发起检索
- 打开文档
- 停止或继续
- 重试或恢复

### Observation
环境返回给 Agent 的结果，包括：
- tool result
- error code
- 空结果
- 文档内容
- 证据片段

### Feedback
训练或评估信号，包括：
- demonstration trajectory
- preference pair
- rule-based reward
- verifier signal
- task success/failure

### Evaluator
用于衡量 Agent 行为质量的指标系统，包括：
- task success rate
- tool selection accuracy
- argument exact match
- schema valid rate
- recovery success rate
- retrieval recall
- cost / latency / step count

---

## 2.2 训练方法的阶段性定位

### SFT
定位：基础盘训练方法。

适合做的事：
- 学会标准工具调用格式
- 学会常见参数模式
- 学会成功轨迹模仿
- 建立可用 baseline

不适合单独解决的问题：
- 强判别式工具选择边界
- 长程失败恢复
- 探索性多步策略优化

### DPO
定位：在工具编排型 Agent 中的第一优先级后训练方法。

适合做的事：
- 纠正错误工具选择
- 压制错误参数填充
- 比较正确恢复轨迹和错误恢复轨迹
- 做能力归因分析

不适合单独解决的问题：
- 强探索驱动的策略改进
- 复杂长程 credit assignment

### PPO / GRPO
定位：在主动检索型 Agent 或更强闭环环境中的第二阶段方法。

适合做的事：
- 优化长程策略
- 优化检索路径
- 优化 stopping policy
- 优化多步决策中的总体收益

不建议过早进入的原因：
- reward 设计复杂
- rollout 成本高
- 训练稳定性要求更高
- 在没有 evaluator 和 error taxonomy 前难以归因

---

## 2.3 场景与方法的匹配原则

### 工具编排型 Agent
推荐顺序：
SFT → DPO → GRPO → PPO

### 主动检索型 Agent
推荐顺序：
SFT → PPO / GRPO → DPO（作为质量校正补充）

### 代码执行型 Agent
推荐顺序：
SFT → DPO（偏好/风格） + PPO / GRPO（测试驱动优化）

---

## 2.4 Agent 训练实战项目地图

为了避免学习路线停留在抽象方法层，本项目引入一组可对照学习的开源 Agent 训练项目与数据集。它们的作用不是直接替代本项目，而是作为不同训练阶段的参考样板。

### 2.4.1 项目分层

```text
第一层：Tool / Function Calling Agent
代表项目：ToolACE、ToolBench、xLAM
核心价值：学习工具选择、参数填充、多工具调用格式、轨迹数据构造。

第二层：Trajectory Agent / Code Agent
代表项目：OpenHands、SWE-Agent、SWE-bench
核心价值：学习状态—动作—观察轨迹、代码环境反馈、测试驱动评价、失败恢复。

第三层：Search / Deep Research Agent
代表项目：Search-R1、DeepResearcher、LiteResearcher
核心价值：学习主动检索、query reformulation、证据聚合、停止策略、检索型 RL。

第四层：Agent RL / Tool RL
代表项目：rStar / rStar2-Agent、DeepCoder 类项目
核心价值：学习 rollout、verifier、code execution reward、GRPO / PPO、长程策略优化。
```

### 2.4.2 与本项目训练阶段的对应关系

| 训练阶段 | 本项目关注点 | 对应开源项目 / 数据 | 主要借鉴点 |
|---|---|---|---|
| Sandbox / Evaluator | 工具 schema、executor、step-level log | ToolBench、ToolACE | 工具描述、API schema、tool call 数据格式 |
| SFT | 成功轨迹模仿 | ToolACE、ToolBench、xLAM | function calling 与 multi-tool trajectory SFT |
| DPO / ORPO | 正误轨迹偏好对 | ToolBench 失败轨迹、OpenHands trajectory | chosen / rejected pair 构造 |
| Step-level Diagnosis | 错误归因 | ToolBench、OpenHands、SWE-Agent | wrong tool、wrong arg、wrong order、no recovery |
| Active Search Agent | 主动检索、打开文档、证据聚合 | Search-R1、DeepResearcher | search / read / answer 多步策略 |
| RL / GRPO / PPO | 长程策略优化 | rStar、DeepCoder、Search-R1 | rollout、reward、verifier、cost-aware policy |

### 2.4.3 训练主线修正

原路线的主线是：

```text
工具编排型 Agent → SFT → DPO → 主动检索型 Agent → RL 准备
```

加入开源项目后，路线可以进一步落到：

```text
ToolACE / ToolBench
    ↓
Tool-use SFT
    ↓
ToolBench / OpenHands failure mining
    ↓
DPO / ORPO preference training
    ↓
OpenHands / SWE-Agent trajectory analysis
    ↓
Search-R1 / DeepResearcher search policy
    ↓
rStar / DeepCoder style GRPO / PPO
```

本阶段仍不建议一开始直接进入 PPO / GRPO。更合理的路径是先完成工具轨迹 SFT 与 DPO，再把主动检索和代码执行环境作为 RL 的入口。

---

# 3. 8 周总路线总览

## 3.1 开源项目增强版 8 周训练总表

| 周次 | 原始训练主线 | 加入的参考项目 / 数据 | 本周训练方法定位 | 新增实操重点 | 新增交付物 |
|---|---|---|---|---|---|
| Week 1 | 建立 Agent 统一抽象 | ToolBench、ToolACE、OpenHands、SWE-Agent | 不训练，先定义问题 | 对照真实项目拆解 State / Action / Observation / Feedback；明确 Tool Agent、Search Agent、Code Agent 的差异 | `notes/open_source_project_map.md` |
| Week 2 | 搭建工具编排型 sandbox | ToolACE、ToolBench、xLAM | 环境准备 | 参考 function calling 数据设计工具 schema；定义可注入错误的 executor；准备 tool selection / arg filling 评测字段 | `env/tool_registry.json`、`data/tool_schema_examples.jsonl` |
| Week 3 | 建立 baseline 与 evaluator | ToolBench、ToolACE、Berkeley Function-Calling 类任务 | Prompt baseline / zero-shot baseline | 建立 tool-level、arg-level、trajectory-level 评估；构造 50~100 条迷你 ToolBench 风格 eval | `eval/tool_eval.py`、`data/tool_eval_mini.jsonl` |
| Week 4 | SFT 基础训练 | ToolACE、ToolBench、xLAM-function-calling | Trajectory SFT | 用成功轨迹训练工具调用格式、参数补全、多步短程执行；可先使用 Qwen3-4B/7B + LoRA | `data/sft_tool_train.jsonl`、`train/sft_tool_lora.yaml` |
| Week 5 | DPO 偏好优化 | ToolBench failure mining、OpenHands trajectory | DPO / ORPO | 构造 chosen/rejected：正确工具 vs 错误工具、正确参数 vs 错误参数、正确恢复 vs 错误恢复 | `data/dpo_tool_pairs.jsonl`、`train/dpo_tool.yaml` |
| Week 6 | Error taxonomy 与 step-level diagnosis | OpenHands、SWE-Agent、SWE-bench Lite | 诊断，不急于 RL | 引入轨迹级 bad case 分析；区分工具选择错、参数错、步骤顺序错、观察结果误读、无恢复 | `eval/trajectory_error_report.md` |
| Week 7 | 过渡到主动检索型 Agent | Search-R1、DeepResearcher、LiteResearcher | Search Agent SFT / RL 准备 | 增加 `search`、`open_doc`、`cite`、`stop` 动作；评估 query reformulation、evidence coverage、citation correctness | `env/retrieval_env.py`、`data/search_agent_eval.jsonl` |
| Week 8 | 沉淀方法论并准备 RL | rStar、DeepCoder、Search-R1、DeepResearcher | GRPO / PPO 方案设计 | 明确何时进入 RL：reward 稳定、rollout 可控、verifier 可用、失败主要来自长程策略 | `notes/rl_readiness_checklist.md`、`notes/agent_training_method_map.md` |

### 3.2 推荐执行顺序

```text
第 1 优先级：ToolACE / ToolBench
用途：完成工具调用 SFT 与 DPO 的最小闭环。

第 2 优先级：OpenHands / SWE-Agent
用途：学习真实轨迹日志、错误恢复、代码执行环境与测试驱动评价。

第 3 优先级：Search-R1 / DeepResearcher
用途：进入主动检索型 Agent，观察 query、证据和停止策略的长程问题。

第 4 优先级：rStar / DeepCoder
用途：学习 Agent RL 的 rollout、verifier、GRPO / PPO 训练范式。
```

### 3.3 阶段性取舍原则

- 如果 Week 4 之前 schema valid rate 还不稳定，不进入 DPO。
- 如果 Week 5 之后 wrong-tool rate 仍然高，优先补充 tool selection pair，不急于 RL。
- 如果 Week 7 的失败主要是 query 写不好，可以先做检索轨迹 SFT；只有当失败集中在多步检索策略、停止策略和证据选择时，才进入 GRPO / PPO。
- 如果缺少可靠 verifier，不做开放式 Agent RL，最多只做离线 DPO / ORPO。

## 阶段划分

### 第 1 阶段（Week 1-4）
建立工具编排型 Agent 的基础训练闭环。

目标：
- 搭 sandbox
- 定义动作
- 建 evaluator
- 跑通 SFT

### 第 2 阶段（Week 5-6）
建立 DPO 与 step-level error diagnosis。

目标：
- 构造 preference pair
- 分析能力增益来源
- 建立错误分类框架

### 第 3 阶段（Week 7-8）
过渡到主动检索型 Agent。

目标：
- 增加 search action
- 增加 retrieval-specific metrics
- 建立进入 RL 思维的准备框架

---

# 4. Week 1 详细计划：建立 Agent 统一抽象

## 4.1 本周目标

把“Agent 训练”从模糊概念收束为一套明确的问题定义。

## 4.2 本周核心问题

需要明确回答：
1. Agent 的 state 是什么？
2. Agent 的 action 是什么？
3. 环境如何反馈？
4. 什么算成功？
5. 什么算失败？
6. 应该如何评估？

## 4.3 学习重点

### 重点一：区分语言输出与动作输出
核心认知：
Agent 训练不是单纯优化自然语言，而是在优化**环境中的可执行动作**。

### 重点二：区分 task-level 与 step-level
核心认知：
总任务成功率很重要，但不足以定位失败原因；必须建立 step-level evaluator。

### 重点三：将 Agent 建模为状态机
核心认知：
Agent 不是一个“聪明回答器”，而是一个在状态空间上进行决策的策略函数。

## 4.4 实操任务

### 任务 A：建立项目目录
建议目录：

```text
agent_lab/
├── env/
├── data/
├── train/
├── eval/
├── logs/
└── notes/
```

### 任务 B：编写问题定义文档
文件：`notes/problem_formulation.md`

内容至少包括：
- 状态定义
- 动作定义
- 反馈定义
- 成功标准
- 失败标准
- 评测指标草案

### 任务 C：绘制状态转移图
画出一个最小 Agent 流程：
用户输入 → 模型输出 tool call → executor 执行 → 返回 observation → 模型继续决策 / 结束

## 本周参考项目与数据接入

### 推荐查看
- ToolBench：观察一个工具学习任务如何组织 API、用户问题、工具调用轨迹和最终答案。
- ToolACE：观察 function calling 数据如何围绕工具选择和参数生成进行合成。
- OpenHands：观察真实 Agent 系统中 action / observation / trajectory 的工程形态。
- SWE-Agent：观察代码任务如何将 GitHub issue、代码修改和测试结果建模为 Agent 轨迹。

### 本周需要形成的判断

```text
ToolBench / ToolACE 更适合 Week 2-5 的工具编排训练；
OpenHands / SWE-Agent 更适合 Week 6 之后的轨迹分析和长程任务；
Search-R1 / DeepResearcher 更适合 Week 7 之后的主动检索与 RL 准备。
```

### 新增交付物
- `notes/open_source_project_map.md`
- `notes/agent_training_project_comparison.md`

## 4.5 本周交付物

- `notes/problem_formulation.md`
- `notes/agent_state_machine.md` 或流程图

## 4.6 本周验收标准

达到以下标准视为完成：
- 能清楚说明 tool selection 和 arg filling 是两类不同错误
- 能清楚说明为什么不能只看最终 task success
- 能写出一个完整的状态—动作—反馈定义

## 4.7 本周复盘问题

- 我是否已经把 Agent 训练从“模型回答”转化为了“策略优化”问题？
- 当前定义的动作空间是否足够清晰？
- 是否已经能为后续 sandbox 设计提供约束？

---

# 5. Week 2 详细计划：搭建工具编排型 Agent sandbox

## 5.1 本周目标

搭建一个不依赖真实外部 API 的可控工具环境。

## 5.2 设计原则

### 原则一：完全可控
所有工具行为必须可复现、可调试、可注入错误。

### 原则二：动作结构化
每个工具必须有明确 schema。

### 原则三：反馈标准化
所有执行结果必须统一封装，方便 evaluator 和日志系统消费。

## 5.3 工具集合建议

### 查询类
- `search_doc(keyword, topk)`
- `get_weather(city, date)`
- `get_order(order_id)`
- `lookup_customer(customer_id)`

### 操作类
- `create_ticket(title, priority, assignee)`
- `send_email(to, subject, body)`
- `schedule_meeting(date, attendees)`

### 计算/转换类
- `calculator(expr)`
- `date_convert(text_date, format)`
- `currency_convert(amount, from, to)`

## 5.4 必须实现的组件

### `env/tools.py`
定义工具逻辑。

### `env/schemas.py`
定义每个工具的 schema。

### `env/executor.py`
统一执行入口，负责：
- 解析 action
- 校验 schema
- 调用对应工具
- 返回标准结果

### 标准返回格式建议

```python
{
    "status": "success" | "error",
    "tool": "get_weather",
    "args": {...},
    "result": {...},
    "error_code": None,
    "error_message": None
}
```

## 5.5 错误类型建议

- missing required field
- invalid field format
- unknown tool
- empty result
- permission denied
- invalid value range

## 本周参考项目与数据接入

### 推荐参考
- ToolACE：用于设计轻量 function calling sandbox，重点学习工具 schema、函数描述和参数字段。
- ToolBench：用于设计更复杂的多工具环境，重点学习多步工具调用轨迹。
- xLAM：用于观察 action model 在多工具调用中的数据组织方式。

### 本周落地方式

将工具环境分成三层：

```text
tool_registry：工具名称、描述、参数 schema
executor：统一执行、schema 校验、错误注入
trace_logger：记录 state / action / observation / error
```

### 新增交付物
- `env/tool_registry.json`
- `env/error_injection.py`
- `data/tool_schema_examples.jsonl`

## 5.6 本周交付物

- `env/tools.py`
- `env/schemas.py`
- `env/executor.py`

## 5.7 本周验收标准

- 手动输入结构化动作后可以稳定执行
- 可以触发并捕获 schema 级错误
- 每次执行都有统一日志输出

## 5.8 本周复盘问题

- 工具集合是否足够覆盖单步和多步任务？
- 当前 schema 是否利于后续 arg-level 评估？
- executor 返回格式是否满足 evaluator 需求？

---

# 6. Week 3 详细计划：建立 baseline 与 evaluator

## 6.1 本周目标

在不做训练的情况下，先跑通一个最小 Agent 闭环，并完成评测器搭建。

## 6.2 本周核心原则

先评测，后训练。

没有 evaluator 的训练，本质上是盲调。

## 6.3 baseline 设计

先实现一个 rule/prompt baseline：
- 给模型固定输出格式要求
- 输出一个 tool call
- 调 executor
- 将结果返回模型生成最终回答

## 6.4 评估指标定义

### Tool Selection Accuracy
在应调用工具的步骤中，tool 是否选对。

### Argument Exact Match
参数是否完整且正确。

### Slot-level F1
字段粒度的参数质量。

### Schema Valid Rate
输出是否可解析、可执行。

### Execution Success Rate
executor 是否返回 success。

### Task Success Rate
是否最终完成任务。

## 6.5 eval 数据集构造建议

先做 50~100 条小规模任务：
- 单步调用
- 短程多步
- 参数歧义
- 工具混淆
- 简单失败恢复

## 6.6 日志格式建议

```python
{
    "task_id": "...",
    "step_id": 1,
    "user_query": "...",
    "model_output": "...",
    "parsed_tool": "get_order",
    "parsed_args": {"order_id": "48392"},
    "executor_status": "success",
    "executor_result": {...},
    "gold_tool": "get_order",
    "gold_args": {"order_id": "48392"}
}
```

## 本周参考项目与数据接入

### 推荐参考
- ToolBench：参考其工具调用任务，构造自己的迷你 eval。
- ToolACE：参考其 function calling 任务，构造单工具与多工具评测样本。
- Berkeley Function-Calling Leaderboard 风格任务：用于理解函数调用评估的常见维度。

### 本周 eval 数据建议

先构造 100 条小型评测集：

| 类型 | 数量 | 目标 |
|---|---:|---|
| 单工具调用 | 30 | 测 tool selection 与 arg filling |
| 多工具短链路 | 30 | 测 step order 与 observation 使用 |
| 参数歧义补全 | 20 | 测上下文槽位补全 |
| 工具混淆 | 10 | 测相似工具选择边界 |
| 失败恢复 | 10 | 测 recovery policy |

### 新增交付物
- `data/tool_eval_mini.jsonl`
- `eval/tool_eval.py`
- `eval/step_metrics.py`

## 6.7 本周交付物

- `run_agent.py`
- `eval/metrics.py`
- `data/eval.jsonl`

## 6.8 本周验收标准

- baseline 可以端到端跑完
- 能输出评估表
- 能从日志中定位基础错误类型

## 6.9 本周复盘问题

- baseline 主要错在 tool selection 还是 arg filling？
- schema error 多，还是执行错误多？
- 当前任务集是否足以区分各类失败？

---

# 7. Week 4 详细计划：SFT 基础训练

## 7.1 本周目标

用成功轨迹建立第一个可训练 baseline。

## 7.2 本周核心认知

SFT 的价值不在于“把模型调得更像聊天助手”，而在于：
- 学会稳定工具调用格式
- 学会常见参数模板
- 学会短程可执行轨迹

## 7.3 数据构造原则

SFT 样本应覆盖：
- 单步工具调用
- 两步到三步短程调用
- 参数从上下文补全
- 常见格式变体
- 少量失败恢复成功轨迹

## 7.4 数据规模建议

- 训练集：2k ~ 5k
- 验证集：200 ~ 500
- 测试集：与训练集任务模板保持一定分布差异

## 7.5 训练技术路线

建议优先：
- LoRA / QLoRA
- 7B 或更小基座模型
- 使用 LLaMA-Factory 或 TRL

## 7.6 训练完成后的重点观察指标

重点看以下指标是否显著提升：
- Schema Valid Rate
- Argument Exact Match
- Execution Success Rate
- 短程 Task Success Rate

## 本周参考项目与数据接入

### 推荐参考
- ToolACE：用于单步 / 短程 function calling SFT。
- ToolBench：用于多工具轨迹 SFT。
- xLAM-function-calling / xLAM-agent：用于观察 action model 的 SFT 数据组织。

### 数据格式建议

SFT 数据不要只保存最终答案，应保存可监督的动作轨迹：

```json
{
  "id": "tool_sft_0001",
  "messages": [
    {"role": "user", "content": "查询订单 48392 的物流状态，并给用户发邮件说明"},
    {"role": "assistant", "content": "<tool_call>{\"name\":\"get_order\",\"arguments\":{\"order_id\":\"48392\"}}</tool_call>"},
    {"role": "tool", "content": "{\"status\":\"shipped\",\"eta\":\"2026-07-12\"}"},
    {"role": "assistant", "content": "<tool_call>{\"name\":\"send_email\",\"arguments\":{\"subject\":\"订单物流状态\",\"body\":\"您的订单已发货，预计 2026-07-12 送达。\"}}</tool_call>"}
  ]
}
```

### 新增交付物
- `data/sft_tool_train.jsonl`
- `data/sft_tool_dev.jsonl`
- `train/sft_tool_lora.yaml`
- `eval/sft_before_after_report.md`

## 7.7 本周交付物

- `data/sft_train.jsonl`
- `train/sft_train.py`
- SFT 前后对比结果

## 7.8 本周验收标准

- SFT 模型相对 prompt baseline 有明确增益
- 参数格式和 schema 合法性显著变好
- 至少能稳定完成部分单步与短程任务

## 7.9 本周复盘问题

- SFT 提升主要体现在格式、参数还是工具选择？
- 哪类任务增益最明显？
- 哪类任务仍然明显失败？

---

# 8. Week 5 详细计划：DPO 偏好优化

## 8.1 本周目标

从“模仿正确轨迹”进入“显式压制错误决策”。

## 8.2 本周核心认知

DPO 不是为了泛化地“让模型更 aligned”，而是为了在工具场景中更清楚地比较：
- 正确动作 vs 错误动作
- 正确参数 vs 错误参数
- 正确恢复 vs 错误恢复

## 8.3 pair 类型设计

### 类型 A：tool selection pair
- chosen：选对工具
- rejected：选错工具

### 类型 B：argument pair
- chosen：参数完整且正确
- rejected：缺字段、字段错、值错、格式错

### 类型 C：recovery pair
- chosen：第一次失败后正确修复
- rejected：失败后继续错、放弃、胡编最终答案

## 8.4 rejected 样本构造方式

### 规则扰动
- 替换成相近工具
- 删除参数
- 修改字段名
- 打乱调用顺序

### 模型采样
- 让 base 或 SFT 模型对同一输入采样多个候选
- 从中筛选坏样本

### rollout 失败挖掘
- 用 SFT 模型真实跑环境
- 收集失败轨迹
- 人工或规则构造修复版本

## 8.5 本周重点评估问题

- DPO 是否降低 wrong-tool rate？
- DPO 是否降低 arg hallucination？
- DPO 是否减少无效调用？
- DPO 是否提升 recovery success？

## 本周参考项目与数据接入

### 推荐参考
- ToolBench：从模型 rollout 中挖掘失败轨迹，用于构造 rejected。
- OpenHands：参考真实 Agent trajectory 中的成功恢复与失败恢复。
- ToolACE-R / ToolACE-DEV：参考 self-refinement 与迭代数据增强思路。

### DPO / ORPO pair 构造建议

| Pair 类型 | chosen | rejected | 目标 |
|---|---|---|---|
| tool selection pair | 正确工具 | 相似但错误工具 | 降低 wrong-tool rate |
| argument pair | 参数完整且格式正确 | 缺字段、错字段、值幻觉 | 降低 arg error |
| step order pair | 先查信息再执行操作 | 未查清直接执行 | 降低 wrong step order |
| recovery pair | 根据 error 修复调用 | 重复错误或胡编答案 | 提升 recovery success |
| stop pair | 信息足够后停止 | 过度调用工具 | 控制成本和冗余步骤 |

### 新增交付物
- `data/dpo_tool_pairs.jsonl`
- `data/orpo_tool_pairs.jsonl`（可选）
- `train/dpo_tool.yaml`
- `eval/dpo_error_reduction_report.md`

## 8.6 本周交付物

- `data/dpo_train.jsonl`
- `train/dpo_train.py`
- SFT vs SFT+DPO 对比分析

## 8.7 本周验收标准

- 能清楚区分 DPO 的增益项
- 至少在某一类错误上明显优于 SFT
- 可以初步形成能力归因结论

## 8.8 本周复盘问题

- DPO 提升最明显的是哪一项能力？
- 当前 pair 是否过多偏向格式，而非决策？
- recovery pair 的比例是否足够？

---

# 9. Week 6 详细计划：建立 error taxonomy 与 step-level diagnosis

## 9.1 本周目标

建立系统化错误分析框架。

## 9.2 核心价值

从这周开始，项目从“会跑训练”进入“能做研究性分析”。

## 9.3 推荐错误分类

至少 8 类：
1. wrong tool
2. missing argument
3. wrong argument value
4. invalid schema
5. wrong step order
6. tool result misinterpretation
7. no recovery
8. hallucinated final answer

## 9.4 分析维度

### 按模型对比
- baseline
- SFT
- SFT + DPO

### 按任务类型对比
- 单步
- 多步
- 参数歧义
- 失败恢复

### 按步骤位置对比
- 第一步错误
- 中间步骤错误
- 恢复阶段错误

## 9.5 需要输出的分析结果

- 每类错误数量
- 每类错误占比
- 各模型在各类错误上的改善幅度
- 典型 bad cases

## 本周参考项目与数据接入

### 推荐参考
- OpenHands：学习真实 Agent trace 的日志粒度和恢复逻辑。
- SWE-Agent / SWE-bench Lite：学习代码执行环境中如何用测试结果做任务级 verifier。
- ToolBench：继续作为工具调用错误分类的基础数据源。

### 新增错误分类

在原 8 类基础上，进一步区分：

```text
tool planning error：选错工具或少调用必要工具
arg grounding error：参数没有从上下文或 observation 中正确落地
observation reading error：工具返回结果读错、漏读或误读
execution recovery error：工具报错后没有修复参数或换工具
premature stop：信息不足就停止
over-search / over-call：工具调用过多但没有增加信息增益
final synthesis error：工具轨迹正确，但最终回答整合错误
```

### 新增交付物
- `eval/trajectory_error_report.md`
- `logs/bad_cases/`
- `notes/sft_vs_dpo_gain_analysis.md`

## 9.6 本周交付物

- `eval/error_analysis.py`
- 一份 bad case report
- 一页项目中期结论摘要

## 9.7 本周验收标准

- 能明确说明 SFT 和 DPO 分别改善了什么
- 能说明仍然未解决的核心瓶颈是什么
- 能提出下一步数据或方法改进方向

## 9.8 本周复盘问题

- 当前最主要的剩余问题是局部错误，还是长程错误？
- recovery 失败是因为不会理解错误，还是不会重新规划？
- 是否已经有必要引入更复杂方法？

---

# 10. Week 7 详细计划：过渡到主动检索型 Agent

## 10.1 本周目标

在现有 Agent 框架上加入主动检索行为。

## 10.2 核心认知

主动检索型 Agent 的关键不是“会调用 search 工具”，而是：
- 是否需要检索
- 用什么 query 检索
- 何时 reformulate
- 何时停止
- 如何基于证据综合

## 10.3 最小环境设计

建议新增：
- 本地 corpus（200 ~ 1000 篇文档）
- 简单 BM25 或向量检索器
- `search(query, topk)`
- `open_doc(doc_id)`

## 10.4 新增任务类型

- 需要先搜索再回答的问题
- 需要多证据支撑的问题
- 需要 query reformulation 的问题
- 需要比较多篇文档后得出结论的问题

## 10.5 新增指标

- retrieval recall@k
- evidence coverage
- citation correctness
- search step count
- query reformulation success

## 本周参考项目与数据接入

### 推荐参考
- Search-R1：重点学习 search action、query reformulation、answer reward 与 GRPO 的结合方式。
- DeepResearcher：重点学习真实 web 环境下的 search / read / plan / write 多步研究型 Agent。
- LiteResearcher：重点学习如何用虚拟检索环境降低真实搜索 RL 的不稳定性和成本。

### 主动检索型 Agent 新动作

```text
search(query, topk)
open_doc(doc_id)
extract_evidence(doc_id, span)
reformulate_query(reason)
answer_with_citation(evidence_ids)
stop(reason)
```

### 新增评测指标

| 指标 | 说明 |
|---|---|
| search necessity accuracy | 是否在需要检索时检索，不需要时不检索 |
| query quality | query 是否覆盖关键实体、约束、时间范围 |
| retrieval recall@k | top-k 是否召回目标证据 |
| evidence coverage | 最终答案是否覆盖必要证据 |
| citation correctness | 引用是否支持结论 |
| stopping accuracy | 是否在信息足够时停止 |
| cost-aware score | 任务成功率与搜索次数 / token 成本的综合指标 |

### 新增交付物
- `env/retrieval_env.py`
- `data/search_agent_sft.jsonl`
- `data/search_agent_eval.jsonl`
- `eval/retrieval_metrics.py`

## 10.6 本周交付物

- `env/retrieval_env.py`
- `data/retrieval_eval.jsonl`
- 检索型 baseline 跑通结果

## 10.7 本周验收标准

- 能观察到 query 问题导致的失败
- 能区分 search failure 和 synthesis failure
- 能为后续 RL 化提供环境基础

## 10.8 本周复盘问题

- 失败是来自 query 质量，还是证据聚合能力？
- 哪类问题最需要多步 search policy？
- 当前任务是否已经体现出长程决策难点？

---

# 11. Week 8 详细计划：沉淀方法论并准备进入 RL 阶段

## 11.1 本周目标

将前 7 周的结果总结成个人 Agent 训练方法论。

## 11.2 需要沉淀的核心结论

### 结论一：工具编排型 Agent 是最佳入门环境
原因：
- 动作清晰
- 反馈明确
- 错误可诊断
- SFT / DPO 易于落地

### 结论二：主动检索型 Agent 才真正引入长程策略优化
原因：
- search policy 本质上是多步决策问题
- 停止策略、query reformulation、信息增益更接近 RL

### 结论三：训练方法的推荐顺序必须跟随环境复杂度变化
- 工具编排：SFT → DPO
- 主动检索：SFT → PPO / GRPO

## 11.3 输出内容建议

写一份阶段总结文档，至少包括：
1. Agent 统一抽象
2. 工具编排型实验结果
3. DPO 增益分析
4. 主动检索型新挑战
5. 是否进入 PPO / GRPO 的判断条件

## 11.4 进入 RL 阶段的判断条件

当且仅当以下条件基本满足时，再考虑 PPO / GRPO：
- evaluator 稳定
- error taxonomy 明确
- 主动检索环境已跑通
- reward 设计草案明确
- 当前瓶颈确实属于长程策略，而非局部格式问题

## 本周参考项目与数据接入

### 推荐参考
- rStar / rStar2-Agent：学习从 SFT 到多阶段 RL、工具执行、verifier、GRPO-RoC 的训练组织方式。
- DeepCoder 类项目：学习代码执行 reward、在线 rollout、测试驱动强化的基本模式。
- Search-R1 / DeepResearcher：作为主动检索型 GRPO / PPO 的参考入口。

### RL readiness checklist

进入 PPO / GRPO 前至少满足：

| 条件 | 最低要求 |
|---|---|
| evaluator 稳定 | 同一轨迹重复评估结果一致 |
| reward 可计算 | task success、evidence correctness 或 test pass 可自动判断 |
| rollout 可控 | 环境可复现，失败不会无限循环 |
| action space 收敛 | 工具集合、参数 schema、stop 动作固定 |
| error taxonomy 明确 | 已知道主要失败来自长程策略，而非格式问题 |
| 成本可接受 | 单次 rollout token、工具调用、执行时间可控 |

### 下一阶段 RL 方案草案

```text
SFT checkpoint
    ↓
rollout collection
    ↓
rule-based verifier / execution verifier
    ↓
reward shaping
    ↓
GRPO / PPO
    ↓
trajectory replay + error diagnosis
```

### 新增交付物
- `notes/rl_readiness_checklist.md`
- `notes/grpo_agent_training_plan.md`
- `notes/next_stage_backlog.md`

## 11.5 本周交付物

- `notes/final_report.md`
- `notes/method_map.md`
- 下一阶段研究 backlog

## 11.6 本周验收标准

- 能清楚解释三类 Agent 的区别
- 能清楚说明各类训练方法适用条件
- 能给出后续 PPO / GRPO 的进入时机

## 11.7 本周复盘问题

- 当前我是否已经掌握 Agent 训练的基础范式？
- 下一阶段更适合进入主动检索 RL，还是转向代码执行型 Agent？
- 当前系统中最值得继续打磨的是环境、数据还是方法？

---

# 12. 项目中的长期原则

## 12.1 先评测，后训练
没有 evaluator，不进入训练。

## 12.2 先单步清晰，再多步复杂
没有必要一开始就做开放环境。

## 12.3 先诊断错误，再讨论方法优劣
没有错误分类，就无法做能力归因。

## 12.4 先低成本高频迭代，再追求大规模训练
优先采用 LoRA / QLoRA、小模型、受控环境。

## 12.5 主动检索阶段再认真进入 RL
在工具编排阶段，先把 SFT / DPO 与评估体系立住。

---

# 13. 推荐项目目录结构

```text
agent_lab/
├── env/
│   ├── tools.py
│   ├── schemas.py
│   ├── executor.py
│   └── retrieval_env.py
├── data/
│   ├── sft_train.jsonl
│   ├── dpo_train.jsonl
│   ├── eval.jsonl
│   └── retrieval_eval.jsonl
├── train/
│   ├── sft_train.py
│   ├── dpo_train.py
│   └── configs/
├── eval/
│   ├── metrics.py
│   ├── error_analysis.py
│   └── run_eval.py
├── logs/
├── notes/
│   ├── problem_formulation.md
│   ├── weekly_review.md
│   ├── final_report.md
│   └── method_map.md
└── README.md
```

---

# 14. 每周固定复盘模板

每周建议统一回答以下问题：

1. 本周完成了什么？
2. 本周最关键的一个新认知是什么？
3. 当前最大的失败点是什么？
4. 这个失败更像数据问题、方法问题，还是评测问题？
5. 下周最优先解决什么？
6. 是否需要缩小问题范围？
7. 当前是否适合引入更复杂训练方法？

---

# 15. 最终目标状态

8 周结束后，应达到的状态不是“已经做出最强 Agent”，而是具备如下能力：

1. 能独立建模一个 Agent 训练问题。
2. 能搭建工具编排型 sandbox。
3. 能构造 SFT 与 DPO 数据。
4. 能设计 step-level evaluator。
5. 能进行系统化 bad case analysis。
6. 能理解主动检索型 Agent 为什么更需要策略优化。
7. 能判断何时应该进入 PPO / GRPO。

---

# 16. 项目后续扩展方向

在当前 8 周路线完成后，可继续扩展：

## 方向 A：主动检索型 Agent + RL
重点：
- query reformulation
- stopping policy
- evidence aggregation reward
- cost-aware retrieval policy

## 方向 B：代码执行型 Agent
重点：
- patch proposal
- test-driven repair
- verifier-guided optimization

## 方向 C：Memory-Augmented Agent
重点：
- 短期记忆
- 检索式长期记忆
- state compression

## 方向 D：Agent Evaluation Platform
重点：
- trace replay
- evaluator 平台化
- 多任务 benchmark
- 自动 bad case 聚类

---

# 17. 一句话项目定位

这是一个以**工具编排型 Agent 为入门环境**、以 **SFT → DPO** 为基础训练主线、以 **step-level diagnosis** 为核心能力、并逐步过渡到**主动检索型 Agent 与 RL 方法**的 Agent 训练项目。

---

# 18. 使用说明

后续在项目中使用本记忆文档时，建议遵循：
- 每周更新一次“已完成 / 未完成 / 新问题”
- 每次方法调整前先回看“项目方法论总纲”
- 每次进入新方向前先确认是否满足“进入 RL 阶段的判断条件”
- 每次出现性能变化时优先记录 error taxonomy 的变化，而不是只记录总分

本文件是该项目的长期基线文档。后续所有迭代，优先在本文件基础上追加，而不是推倒重写。

---

# 19. 开源 Agent 训练实战项目清单

本节用于记录后续学习和复现实验中最值得跟踪的开源项目。这里不追求一次性全部跑通，而是按训练阶段逐步吸收其数据格式、评估方法和训练组织方式。

## 19.1 ToolACE

### 定位
Tool / Function Calling 数据构造与 SFT 参考项目。

### 适合阶段
- Week 2：工具 schema 设计
- Week 3：function calling eval 构造
- Week 4：tool-use SFT

### 借鉴点
- API pool 构建
- function calling 样本合成
- 工具选择与参数生成
- rule-based + model-based 数据校验

### 本项目使用方式
不建议一开始完整复现 ToolACE pipeline。更合理做法是先抽象其数据结构，迁移到自己的工具 sandbox：

```text
user query
+ tool schema
+ expected tool call
+ expected arguments
+ execution result
```

---

## 19.2 ToolBench

### 定位
工具学习与多工具轨迹训练的经典参考项目。

### 适合阶段
- Week 2：多工具环境设计
- Week 3：baseline 与 evaluator
- Week 4：trajectory SFT
- Week 5：DPO rejected 样本挖掘

### 借鉴点
- 多工具调用轨迹
- API 调用链路
- tool-use 任务评估
- 失败轨迹可用于构造 preference pair

### 本项目使用方式
优先参考 ToolBench 的数据组织方式，而不是直接接入大量真实 API。先在本地构造小型可控版 ToolBench：

```text
10~20 个工具
100 条 eval
2k~5k 条 SFT
500~1000 条 DPO pair
```

---

## 19.3 xLAM

### 定位
Action Model / Tool-use Model 的模型与数据参考。

### 适合阶段
- Week 3：function calling baseline 对照
- Week 4：SFT 数据格式参考

### 借鉴点
- action model 训练数据组织
- 多工具调用泛化能力
- 与 Berkeley Function-Calling 类评测的关系

### 本项目使用方式
将 xLAM 作为“训练数据格式”和“能力边界”的参考，不作为第一阶段必须复现对象。

---

## 19.4 OpenHands

### 定位
真实工程型 Agent 系统与 trajectory 参考项目。

### 适合阶段
- Week 6：trajectory-level error diagnosis
- Week 7：长程 Agent 行为分析
- Week 8：进入代码执行型 Agent 或 RL 之前的系统参考

### 借鉴点
- Agent runtime
- 代码编辑、终端、浏览器等多工具 action
- 真实 action / observation 日志
- 失败恢复、任务拆解、执行反馈

### 本项目使用方式
不建议 Week 1 直接复现 OpenHands。更合理的是在 Week 6 之后将其作为“真实复杂 Agent”样板，用于校准自己的 toy sandbox 是否具备扩展价值。

---

## 19.5 SWE-Agent / SWE-bench

### 定位
代码执行型 Agent 与测试驱动评价的经典组合。

### 适合阶段
- Week 6：step-level 与 task-level 评价对照
- Week 8：RL readiness 判断
- 下一阶段：代码执行型 Agent 训练

### 借鉴点
- issue → patch → test 的闭环
- verifier 明确，reward 相对干净
- 长程代码定位、修改、测试、恢复

### 本项目使用方式
先跑 SWE-bench Lite 或构造本地 mini-SWE 环境。不要一开始直接追求完整 SWE-bench 成绩。

---

## 19.6 Search-R1

### 定位
Search Agent + GRPO / RL 的参考方向。

### 适合阶段
- Week 7：主动检索型 Agent
- Week 8：检索型 RL 方案设计

### 借鉴点
- search action 作为显式动作
- query reformulation
- answer reward
- 检索次数与效果的权衡

### 本项目使用方式
先复刻思想，不急于完整训练。可以先在本地 corpus 上构建：

```text
question → search → open_doc → evidence → answer → reward
```

---

## 19.7 DeepResearcher / LiteResearcher

### 定位
Deep Research Agent 的 RL 参考项目。

### 适合阶段
- Week 7：主动检索与证据聚合
- Week 8：真实 web / 虚拟 web 环境下 RL 的取舍判断

### 借鉴点
- search / read / plan / write 多步流程
- 真实 web 环境中的噪声、成本、不稳定性
- 虚拟检索环境降低 RL 成本
- 多证据交叉验证与自我纠偏

### 本项目使用方式
先在本地 corpus 上模拟 Deep Research，不直接进入真实 web RL。重点学习其 reward 和 rollout 设计。

---

## 19.8 rStar / rStar2-Agent

### 定位
Agentic RL、工具执行与 GRPO 类训练参考项目。

### 适合阶段
- Week 8：RL 方案设计
- 下一阶段：GRPO / PPO 实验

### 借鉴点
- SFT → 多阶段 RL
- code execution tool
- verifier / judge server
- GRPO 类算法
- rollout 成本控制

### 本项目使用方式
将 rStar 作为 RL 阶段的参考上限。当前 8 周内只需要理解其训练组织方式，不要求完整复现。

---

# 20. 推荐学习与复现实验顺序

## 20.1 最小可行路线

```text
Step 1：读 ToolACE / ToolBench 数据格式
Step 2：搭本地 toy tool sandbox
Step 3：构造 100 条 eval + 2k 条 SFT
Step 4：用 Qwen3-4B / 7B LoRA 跑 tool-use SFT
Step 5：从失败样本构造 500 条 DPO pair
Step 6：做 SFT vs SFT+DPO error taxonomy
Step 7：加入 search/open_doc 动作
Step 8：设计 GRPO / PPO readiness checklist
```

## 20.2 进阶路线

```text
ToolBench-style Tool Agent
    ↓
OpenHands-style Trajectory Agent
    ↓
SWE-Agent-style Code Execution Agent
    ↓
Search-R1 / DeepResearcher-style Search Agent
    ↓
rStar-style Agentic RL
```

## 20.3 与个人业务场景的迁移方向

本项目后续可以迁移到公安业务 Agent，但不建议一开始直接用真实业务数据做 RL。更稳妥的迁移方式是：

```text
通用 tool sandbox
    ↓
法律 / 案由检索 tool sandbox
    ↓
case-judge Agent 轨迹 SFT
    ↓
正确法条检索 vs 错误法条检索 DPO
    ↓
主动检索型案情判断 Agent
    ↓
带 verifier 的 RL 或 RLAIF
```

可优先设计如下业务工具：

```text
search_law(query, topk)
open_law(law_id)
search_case_rule(case_type, element)
summarize_case_facts(transcript)
judge_criminal_or_administrative(facts, evidence)
```

对应训练重点：
- tool selection：什么时候检索法条，什么时候总结案情。
- arg filling：检索 query 是否包含案由、金额、行为方式、结果、主体身份等关键约束。
- observation reading：模型是否正确读取法条阈值和构成要件。
- recovery：检索不到时是否改写 query 或切换规则库。
- final judgment：刑事 / 行政结论是否受证据和法条约束。

---

# 21. 增强版一句话定位

这是一个以 **ToolACE / ToolBench 风格工具调用数据** 为训练起点，以 **OpenHands / SWE-Agent 风格轨迹诊断** 为中期参照，以 **Search-R1 / DeepResearcher / rStar 风格 Agent RL** 为后续扩展方向的 8 周 Agent 训练实战路线。
