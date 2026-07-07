# Agent 训练问题形式化

## 1. 目标

本文档的目标不是简单解释 Agent 的组成部分，而是把“训练一个会使用工具的 Agent”形式化为一个可以落地的数据、训练和评测问题。

我们希望回答以下问题：

- 一条 Agent 训练样本应该如何表示？
- 模型在每一步看到的输入是什么？
- 模型在每一步要预测的输出是什么？
- 工具执行结果如何改变下一步状态？
- 训练时优化什么目标？
- 评测时如何判断单步 action 和完整 trajectory 的质量？

本文档关注的是 tool-using Agent，即模型需要在自然语言回答和工具调用之间做决策的场景。

## 2. 总体形式化

### 2.1 核心概念澄清

在讨论 Agent 训练之前，需要明确一个关键区分：**完整的 Agent 是一个工程系统**，而我们要训练的是这个系统中的**决策模块（policy）**。

```
完整的 Agent 系统 = 状态管理器 + 工具执行器 + 记忆系统 + 决策模块(policy)
                                                          ↑
                                                      这是训练目标
```

### 2.2 训练什么？学习什么？

我们要训练的是 Agent 系统中的**决策模块（policy）**，使其学会从当前状态到下一步动作的映射：

$$
\pi_\theta: s_t \to a_t
$$

其中：
- $s_t$：第 $t$ 步时模型的**输入状态**（模型推理时真实可见的信息）
- $a_t$：第 $t$ 步时模型的**输出动作**（如调用工具、给出最终答案、反问用户）

这本质上是**教模型如何像专家一样分步思考和执行**，类似思维链（CoT），但将内部思考过程外部化为可执行的动作序列。

### 2.3 为什么需要轨迹数据？

决策模块的训练需要**轨迹（trajectory）数据**，因为：

1. **学习状态→动作的映射**：需要知道在每个特定状态下应该采取什么动作
2. **学习多步决策逻辑**：需要看到专家如何一步步推进任务完成
3. **学习从错误中恢复**：需要看到专家如何处理工具失败、参数错误等情况

一条轨迹 $\tau$ 记录了完整任务执行过程：

$$
\tau = (s_0, a_0, o_1, s_1, a_1, o_2, \ldots, s_n, a_n)
$$

- $s_t$：模型第 $t$ 步看到的**输入状态**（由系统生成）
- $a_t$：模型第 $t$ 步输出的**动作**（决策模块的输出）
- $o_{t+1}$：动作执行后环境返回的**观察**（工具结果、错误信息等）
- $s_{t+1}$：系统根据 $(s_t, a_t, o_{t+1})$ 生成的**新状态**

### 2.4 训练目标形式化

给定：
- 任务分布 $D_{\text{task}}$：各种用户任务的集合
- 工具集合 $T$：可用的工具及其 schema
- 执行环境 $\text{Env}$：生成观察 $o_{t+1}$ 和执行状态转移 $s_t \rightarrow s_{t+1}$ 的系统
- 决策模块 $\pi_\theta$：要训练的参数化函数
- 评测器 $\text{Evaluator}$：判断轨迹是否成功的函数

训练目标是学习一个决策函数：

$$
\maximize_{\theta} \ \mathbb{E}_{\text{task} \sim D_{\text{task}}}[\text{Evaluator}(\tau_\theta, \text{task})]
$$

其中 $\tau_\theta$ 是通过策略 $\pi_\theta$ 与环境交互生成的轨迹，$\text{task}$ 是任务规范。

### 2.6 与思维链（Chain-of-Thought）的对比

理解 Agent 训练与思维链（CoT）训练的关系，有助于把握其核心思想：

| 维度 | 思维链（CoT）训练 | Agent 训练 |
|------|-----------------|-----------|
| **训练目标** | 学习在内部如何分步推理 | 学习在外部如何分步执行 |
| **训练数据** | (问题, 推理步骤, 答案) 三元组 | (状态, 动作, 观察, 新状态) 轨迹序列 |
| **模型输出** | 最终答案（可能附带推理文本） | 下一步动作（工具调用、最终回答等） |
| **执行方式** | 推理在模型内部完成，不可观察 | 动作在外部世界执行，可验证可干预 |
| **状态管理** | 隐含在模型的内部激活中 | 显式表示为 $s_t$，由系统维护 |
| **泛化能力** | 学会相似问题的推理模式 | 学会相似状态下的决策模式 |

#### 本质联系：外部化的思维链

Agent 训练可以看作是 **CoT 的外部化和可执行化**：

- **CoT**：模型内部默默思考："先查天气 → 温度适中 → 适合跑步" → 输出"适合跑步"
- **Agent**：模型将思考变为可执行动作：
  - 状态1（需要信息）→ 动作1（调用天气工具）
  - 状态2（已获取数据）→ 动作2（分析并判断）
  - 状态3（判断完成）→ 动作3（输出最终答案）

这种转变带来了关键优势：

1. **可验证性**：工具调用参数是否正确，可以实际执行验证
2. **可组合性**：可以复用工具、组合多个任务步骤
3. **可恢复性**：工具失败时可以重试、修正参数
4. **可解释性**：每个决策步骤都清晰可见，便于调试分析

#### 训练数据的深层相似性

尽管形式不同，但两者训练数据的本质都是 **"展示专家的思考/决策过程"**：

```python
# CoT 训练样本：展示思考过程
{
  "input": "查北京天气，看是否适合跑步",
  "reasoning": "先查北京天气 → 温度18-25度 → 晴天 → 适合跑步",
  "answer": "适合跑步"
}

# Agent 训练样本：展示执行过程  
{
  "state": "需要查询北京天气",
  "action": "调用天气工具(北京)",
  "next_state": "已获取天气数据(晴,18-25度)",
  "next_action": "判断适合跑步"
}
```

两者都旨在让模型**学会分步解决问题的模式**，只是 CoT 停留在文本推理层面，而 Agent 将其升级为可执行的动作序列。

这种训练让模型获得**持续思考和任务跟踪的能力**，而不仅仅是单次响应的能力。

## 3. Task 定义

一个 task 是用户希望 Agent 完成的目标。

任务可以表示为：

```json
{
  "task_id": "weather_001", 
  "user_query": "查询明天上海的天气，并告诉我是否适合户外跑步。",
  "available_tools": ["weather"],
  "success_criteria": [
    "调用天气工具查询上海明天天气",
    "根据天气结果判断是否适合户外跑步",
    "最终回答不能编造工具未返回的信息"
  ]
}
```

任务定义至少应该包含：

- `task_id`：任务唯一标识。
- `user_query`：用户原始输入。
- `available_tools`：该任务可使用的工具集合。
- `success_criteria`：任务成功条件。

更复杂的任务还可以包含：

- `constraints`：例如不能访问外部网络、必须先查 A 再查 B。
- `reference_answer`：参考最终答案。
- `reference_trajectory`：专家轨迹。
- `metadata`：任务类型、难度、领域、是否多工具等。

## 4. Tool 定义

一个工具 `tool` 可以表示为：

```json
{
  "name": "weather",
  "description": "查询指定地点和日期的天气。",
  "schema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string"
      },
      "date": {
        "type": "string"
      }
    },
    "required": ["location", "date"]
  }
}
```

工具定义至少包含：

- `name`：工具名称。
- `description`：工具用途。
- `schema`：参数结构、类型和必填字段。

在训练和评测中，工具 schema 是 action space 的一部分。模型不仅要知道可以调用哪个工具，还要生成符合 schema 的参数。

## 5. State 定义

`State` 是模型在第 `t` 步做决策时可见的全部信息。

可以表示为：

```json
{
  "task": {
    "user_query": "查询明天上海的天气，并告诉我是否适合户外跑步。"
  },
  "messages": [
    {
      "role": "user",
      "content": "查询明天上海的天气，并告诉我是否适合户外跑步。"
    }
  ],
  "tools": [
    {
      "name": "weather",
      "schema": {}
    }
  ],
  "history": [],
  "progress": {
    "step": 0,
    "finished": false,
    "known_facts": [],
    "open_requirements": [
      "需要查询上海明天天气",
      "需要判断是否适合跑步"
    ]
  }
}
```

State 中的核心字段：

- `task`：任务信息。
- `messages`：模型可见的对话上下文。
- `tools`：当前可用工具及 schema。
- `history`：已经发生过的 action 和 observation。
- `progress`：当前任务进展的结构化描述。
 

需要注意：

- `state` 不等于完整原始日志，而是模型做下一步决策时实际可见的输入。
- `progress` 可以来自人工标注、规则系统，也可以暂时只作为分析字段，不直接喂给模型。
  人工标注：专家写的“当前任务进展”
  规则系统：程序根据当前状态推导出的进度
  其他辅助模块：例如任务规划器、阶段追踪器
  如果你只是想用它做训练数据分析、评估、或生成样本，那么它可以“在数据里存在，但不作为模型输入”；如果目标是贴近真实运行时，就更要避免把这种辅助进度直接当成模型可见的 state
- 如果训练目标是贴近真实 Agent 运行时，state 应尽量匹配真实推理时的输入格式。

## 6. State 的两种表示

为了避免训练时引入推理时不可用的信息，需要区分两种 state。

### 6.1 Runtime State

`Runtime State` 是模型真实推理时能看到的状态。

它通常包括：

- system / developer / user messages。
- 历史 assistant messages。
- 历史 tool calls。
- 历史 tool results。
- 当前可用工具 schema。

示例：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "查询明天上海的天气，并告诉我是否适合户外跑步。"
    },
    {
      "role": "assistant",
      "tool_call": {
        "name": "weather",
        "arguments": {
          "location": "上海",
          "date": "明天"
        }
      }
    },
    {
      "role": "tool",
      "name": "weather",
      "content": {
        "temperature": "18-24C",
        "rain_probability": "20%",
        "wind": "light"
      }
    }
  ],
  "tools": [
    {
      "name": "weather",
      "schema": {}
    }
  ]
}
```

如果目标是训练一个真实可运行的 Agent，SFT 的 input 应该尽量使用 Runtime State。

### 6.2 Annotated State

`Annotated State` 是为了分析、标注、评测而额外添加的结构化状态。

它可以包括：

- `progress`
- `known_facts`
- `open_requirements`
- `expected_next_action`
- `failure_labels`

示例：

```json
{
  "progress": {
    "known_facts": [
      "上海明天天气已查询"
    ],
    "open_requirements": [
      "判断是否适合跑步"
    ]
  },
  "expected_next_action": {
    "type": "final_answer"
  }
}
```

Annotated State 的作用是帮助构造数据和评测，不一定应该直接喂给模型。

``` 注释： state  这里似懂非懂， 尤其是给模型看的  和不和给模型看的这块 ```

### 6.3 关键原则

训练时需要明确区分：

- `model_input_state`：模型真实看到的输入。
- `annotation_state`：标注和 evaluator 使用的辅助信息。

如果把 `expected_next_action`、`open_requirements` 这类强提示直接放进模型输入，模型可能学到的是读取标注，而不是从上下文中推理下一步 action。

因此，默认建议：

- SFT 输入使用 `model_input_state`。
- evaluator 使用 `annotation_state`。
- 数据分析可以同时保存两者。

### 6.4 Runtime Message 模板

第一版训练输入应尽量贴近真实模型推理格式。

推荐 runtime input：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an agent that can call tools when needed."
    },
    {
      "role": "user",
      "content": "查询明天上海的天气，并告诉我是否适合户外跑步。"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "name": "weather",
          "arguments": {
            "location": "上海",
            "date": "明天"
          }
        }
      ]
    },
    {
      "role": "tool",
      "name": "weather",
      "content": {
        "temperature": "18-24C",
        "rain_probability": "20%",
        "wind": "light"
      }
    }
  ],
  "tools": [
    {
      "name": "weather",
      "description": "查询指定地点和日期的天气。",
      "input_schema": {}
    }
  ]
}
```

训练时应明确：

- `messages` 和 `tools` 是模型输入。
- `annotation_state`、`labels`、`eval_result` 不是模型输入。
- 如果模型框架有原生 tool calling 格式，应优先使用原生格式，而不是把工具 schema 拼成普通文本。

### 6.5 字段使用边界

字段使用边界需要在数据层面固定，避免训练泄漏。

| 字段 | SFT input | SFT target | evaluator | analysis | 说明 |
|---|---:|---:|---:|---:|---|
| `messages` | yes | no | yes | yes | 模型真实上下文 |
| `tools` | yes | no | yes | yes | 可用工具 schema |
| `model_input_state` | yes | no | yes | yes | runtime state 容器 |
| `annotation_state` | no | no | yes | yes | 标注和评测辅助信息 |
| `expected_next_action` | no | no | yes | yes | 不能泄漏给模型 |
| `expert_action` | no | yes | yes | yes | SFT 目标 |
| `labels` | no | no | yes | yes | 成功失败标签 |
| `metadata` | no by default | no | yes | yes | 可用于过滤和分桶 |

## 7. Action 定义

`Action` 是模型在状态 $s_t$ 下输出的下一步行为。

当前可以把 action space 分成三类：

### 7.1 Tool Action

模型决定调用工具：

```json
{
  "type": "tool_call",
  "tool_name": "weather",
  "arguments": {
    "location": "上海",
    "date": "明天" 
  }
}
```

Tool action 的正确性包括：

- 工具名称是否正确。
- 参数字段是否完整。
- 参数类型是否符合 schema。
- 参数值是否符合用户意图和上下文。

### 7.2 Final Action

模型决定结束工具调用并回答用户：

```json
{
  "type": "final_answer",
  "content": "明天上海天气适合户外跑步，但建议避开降雨时段。"
}
```

Final action 的正确性包括：

- 是否在信息足够时才结束。
- 回答是否满足用户任务。
- 是否基于已有 observation。
- 是否包含幻觉内容。

### 7.3 Ask User Action

当用户输入不足以完成任务时，模型可以反问用户：

```json
{
  "type": "ask_user",
  "content": "你想查询哪个城市的天气？"
}
```

Ask user action 适用于：

- 缺少必要参数。
- 用户意图存在歧义。
- 工具调用前必须获得用户确认。

是否支持 `ask_user` 取决于具体训练任务设置。如果任务假设 Agent 必须自主完成，则可以暂时不纳入 action space。

### 7.4 Action 合法性与非法样例

Action evaluator 应先检查 action 是否可解析，再检查业务正确性。

| Action | 合法条件 | 非法样例 | Failure Type |
|---|---|---|---|
| `tool_call` | 包含 `tool_name` 和 `arguments`，且工具存在 | `{"type":"tool_call","arguments":{}}` | `invalid_action`、`missing_argument` |
| `tool_call` | `arguments` 通过对应 tool input schema | `{"location":123}` | `invalid_schema`、`wrong_argument_type` |
| `final_answer` | 包含非空 `content` | `{"type":"final_answer"}` | `invalid_action` |
| `ask_user` | 包含非空 `content`，且任务允许反问 | `{"type":"ask_user","content":""}` | `invalid_action` |
| any | `type` 在允许集合中 | `{"type":"search"}` | `invalid_action` |

### 7.5 Action 终止条件

不同 action 对 trajectory 的终止含义不同：

| Action | 默认是否终止 | 说明 |
|---|---:|---|
| `tool_call` | no | 进入工具执行，再返回 observation |
| `final_answer` | yes | 正常终止，但不等于任务成功 |
| `ask_user` | depends | 单轮任务中可终止，多轮任务中等待用户补充 |
| invalid action | depends | 可以终止，也可以允许一次格式修复 |

第一阶段建议：

- `final_answer` 总是终止。
- `ask_user` 在单轮任务中终止。
- invalid action 直接终止并标记失败。

## 8. Observation 定义

`Observation` 是 action 被执行后环境返回的信息。

如果 action 是 tool call，则 observation 可能是：

```json
{
  "type": "tool_result",
  "tool_name": "weather",
  "status": "success",
  "result": {
    "temperature": "18-24C",
    "rain_probability": "20%",
    "wind": "light"
  }
}
```

也可能是错误：

```json
{
  "type": "tool_error",
  "tool_name": "weather",
  "status": "schema_error",
  "error": {
    "code": "MISSING_REQUIRED_FIELD",
    "message": "Missing required field: location"
  }
}
```

Observation 类型包括：

- `tool_result`：工具正常返回。
- `tool_error`：工具执行失败。
- `schema_error`：参数未通过 schema 校验。
- `empty_result`：工具执行成功但结果为空。
- `system_error`：权限、网络、超时等系统错误。

Observation 会进入下一步 state，影响模型后续 action。

### 8.1 标准 Observation Error Codes

第一版可以使用以下错误码：

| Error Code | Observation Type | Retryable | 说明 |
|---|---|---:|---|
| `MISSING_REQUIRED_FIELD` | `schema_error` | yes | 缺少必填参数 |
| `WRONG_ARGUMENT_TYPE` | `schema_error` | yes | 参数类型错误 |
| `UNKNOWN_TOOL` | `schema_error` | no | 工具不存在 |
| `INVALID_ARGUMENT_VALUE` | `tool_error` | yes | 参数值不被工具接受 |
| `EMPTY_RESULT` | `empty_result` | depends | 查询成功但无结果 |
| `TIMEOUT` | `system_error` | yes | 工具执行超时 |
| `PERMISSION_DENIED` | `system_error` | no | 无权限 |
| `RATE_LIMITED` | `system_error` | yes | 频率限制 |
| `INTERNAL_ERROR` | `system_error` | yes | 工具内部错误 |

错误码的作用：

- 指导模型是否重试。
- 指导 transition 是否终止。
- 指导 evaluator 区分模型错误和环境错误。

## 9. Transition 定义

Transition 描述环境如何从当前状态转移到下一状态：

$$
s_{t+1} = \text{Transition}(s_t, a_t, o_{t+1})
$$

在工具调用场景中，transition 通常做以下事情：

- 将 action 记录到 history。
- 将 observation 记录到 history。
- 更新 messages 或上下文。
- 更新 progress，例如标记某个子目标已完成。
- 更新错误状态，例如记录失败次数、schema 错误类型。
- 判断任务是否已经完成或是否应该继续。

示例：

```json
{
  "history": [
    {
      "action": {
        "type": "tool_call",
        "tool_name": "weather",
        "arguments": {
          "location": "上海",
          "date": "明天"
        }
      },
      "observation": {
        "type": "tool_result",
        "status": "success"
      }
    }
  ],
  "progress": {
    "known_facts": [
      "上海明天天气已查询"
    ],
    "open_requirements": [
      "需要判断是否适合跑步"
    ]
  }
}
```

Transition 是 Agent 建模中非常关键的一环，因为它决定了模型在下一步究竟能看到什么。

## 10. Transition Rules

Transition rule 定义环境如何根据当前 state、模型 action 和 observation 更新下一步 state。

形式上：

$$
s_{t+1} = \text{Transition}(s_t, a_t, o_{t+1})
$$

工程上，transition 至少需要更新：

- `messages`：追加 assistant action 和 tool observation。
- `history`：保存结构化 action / observation。
- `annotation_state.progress`：更新已知事实、未完成要求和错误状态。
- `terminal_state`：判断任务是否结束。
- `labels`：如果出现失败，记录 failure type。

### 10.1 通用状态更新

每一步 action 之后，都应该生成一条 step record：

```json
{
  "step_index": 0,
  "model_input_state": {},
  "annotation_state": {},
  "action": {},
  "observation": {}
}
```

通用更新规则：

| 更新目标 | 更新规则 |
|---|---|
| `history` | 追加当前 step 的 action 和 observation |
| `messages` | 如果 action 是 tool call，追加 assistant tool call；如果有 observation，追加 tool message；如果 action 是 final answer，追加 assistant final message |
| `progress.step` | 自增 1 |
| `progress.known_facts` | 从成功 observation 中抽取可用事实 |
| `progress.open_requirements` | 根据 action 和 observation 移除已满足要求 |
| `progress.error_state` | 如果 observation 是错误，记录错误类型、工具名、是否可重试 |
| `terminal_state` | 如果 action 是 final answer、ask_user 或达到终止条件，则设置终止原因 |

### 10.2 Tool Call 成功

适用条件：

```text
a_t.type == "tool_call"
o_{t+1}.status == "success"
o_{t+1}.type == "tool_result"
```

转移规则：

| 字段 | 更新方式 |
|---|---|
| `messages` | 追加 assistant tool call 和 tool result |
| `history` | 记录 tool action 与 tool result |
| `known_facts` | 从 `observation.result` 抽取事实 |
| `open_requirements` | 移除已经由工具结果满足的需求 |
| `error_state` | 清空或保持为空 |
| `terminal_state` | 通常不终止，进入下一步由模型决定是否 final answer |
| `failure_types` | 不新增失败标签 |

示例：

```json
{
  "progress": {
    "known_facts": [
      "上海明天气温 18-24C",
      "上海明天降雨概率 20%"
    ],
    "open_requirements": [
      "判断是否适合跑步"
    ],
    "error_state": null
  }
}
```

### 10.3 Tool Call Schema Error

适用条件：

```text
a_t.type == "tool_call"
o_{t+1}.type == "schema_error"
```

转移规则：

| 字段 | 更新方式 |
|---|---|
| `messages` | 追加 assistant tool call 和 schema error tool message |
| `history` | 记录非法参数和 schema error |
| `known_facts` | 不新增事实 |
| `open_requirements` | 保持不变 |
| `error_state` | 记录 `schema_error`、缺失字段、是否可重试 |
| `terminal_state` | 通常不终止，允许模型修正参数后重试 |
| `failure_types` | step-level 标记 `invalid_schema`，如果缺字段则标记 `missing_argument` |

示例：

```json
{
  "progress": {
    "known_facts": [],
    "open_requirements": [
      "查询上海明天天气",
      "判断是否适合跑步"
    ],
    "error_state": {
      "type": "schema_error",
      "tool_name": "weather",
      "retryable": true,
      "missing_fields": ["location"]
    }
  }
}
```

### 10.4 Tool Call Empty Result

适用条件：

```text
a_t.type == "tool_call"
o_{t+1}.type == "empty_result"
```

转移规则：

| 字段 | 更新方式 |
|---|---|
| `messages` | 追加 assistant tool call 和 empty result observation |
| `history` | 记录 tool action 与 empty result |
| `known_facts` | 不新增事实，或记录“未查到结果” |
| `open_requirements` | 通常保持不变，除非任务允许“无结果”作为答案 |
| `error_state` | 记录 `empty_result` 和可重试信息 |
| `terminal_state` | 不立即终止，除非任务规则允许无结果终止 |
| `failure_types` | 不一定是模型失败；如果参数错误导致空结果，则后续 evaluator 标记 `wrong_argument_value` |

空结果不应自动等同于失败。它可能表示：

- 工具确实没有数据。
- 参数过窄。
- 参数值错误。
- 工具本身异常但未返回错误。

### 10.5 Tool Execution Error

适用条件：

```text
a_t.type == "tool_call"
o_{t+1}.type in ["tool_error", "system_error"]
```

转移规则：

| 字段 | 更新方式 |
|---|---|
| `messages` | 追加 assistant tool call 和 error observation |
| `history` | 记录工具错误 |
| `known_facts` | 不新增事实 |
| `open_requirements` | 保持不变 |
| `error_state` | 记录错误码、是否可重试、重试次数 |
| `terminal_state` | 如果不可恢复，设置为 `tool_error_unrecoverable`；否则继续 |
| `failure_types` | 如果是环境问题，不一定标记模型失败；如果模型选择错误工具导致错误，则标记 `wrong_tool` |

是否终止取决于：

- 错误是否 `retryable`。
- 是否还有替代工具。
- 是否超过最大重试次数。
- 用户任务是否允许部分完成。

### 10.6 Final Answer

适用条件：

```text
a_t.type == "final_answer"
```

转移规则：

| 字段 | 更新方式 |
|---|---|
| `messages` | 追加 assistant final answer |
| `history` | 记录 final action |
| `known_facts` | 不再新增事实 |
| `open_requirements` | 由 evaluator 判断是否全部满足 |
| `terminal_state` | 设置 `reason = final_answer` |
| `failure_types` | 如果仍有未满足需求，标记 `premature_final_answer`；如果缺少必要工具调用，标记 `missing_tool_call`；如果内容无依据，标记 `hallucinated_final_answer` |

Final answer 是 trajectory 的正常终止动作，但正常终止不等于任务成功。是否成功由 trajectory evaluator 判定。

### 10.7 Ask User

适用条件：

```text
a_t.type == "ask_user"
```

转移规则：

| 字段 | 更新方式 |
|---|---|
| `messages` | 追加 assistant ask user message |
| `history` | 记录 ask_user action |
| `open_requirements` | 保持未完成，等待用户补充 |
| `terminal_state` | 在单轮训练设置中可设置 `reason = ask_user`；在多轮环境中等待用户新输入 |
| `failure_types` | 如果用户信息充足却反问，标记 `unnecessary_ask_user`；如果确实缺少必要信息，不标记失败 |

是否允许 `ask_user` 取决于任务设置：

- 单轮闭环任务：`ask_user` 可以作为终止状态。
- 多轮交互任务：`ask_user` 后等待新的 user message，继续生成下一步 state。

### 10.8 Invalid Action

适用条件：

```text
action cannot be parsed
or action.type not in allowed_action_types
or required action fields are missing
```

转移规则：

| 字段 | 更新方式 |
|---|---|
| `messages` | 可记录原始非法输出，或不进入正式 message history |
| `history` | 记录 invalid action |
| `known_facts` | 不新增 |
| `open_requirements` | 保持不变 |
| `error_state` | 记录 parse/schema/action type 错误 |
| `terminal_state` | 可设置 `reason = invalid_action`，或允许一次格式修复 |
| `failure_types` | 标记 `invalid_action`，必要时标记 `invalid_schema` |

Invalid action 是 action 层面的失败，不应该进入普通正样本。

### 10.9 最大步数终止

为了避免无限循环，环境应设置 `max_steps`。

适用条件：

```text
progress.step >= max_steps
and terminal_state is not set
```

转移规则：

| 字段 | 更新方式 |
|---|---|
| `terminal_state.reason` | `max_steps_exceeded` |
| `terminal_state.success` | false |
| `failure_types` | 标记 `max_steps_exceeded`，如果存在重复调用可标记 `looping_tool_call` |

最大步数不是模型 action，而是环境终止条件。

### 10.10 Transition Decision Table

| 当前 action | Observation | 下一状态 | 是否终止 | 可能失败标签 |
|---|---|---|---:|---|
| `tool_call` | `tool_result/success` | 记录结果，更新 known facts 和 open requirements | no | none |
| `tool_call` | `schema_error` | 记录 schema error，等待模型修正 | no | `invalid_schema`、`missing_argument` |
| `tool_call` | `empty_result` | 记录空结果，等待重试、改参或解释无结果 | no by default | `wrong_argument_value` if caused by bad args |
| `tool_call` | `tool_error/retryable` | 记录错误，允许重试或替代工具 | no | depends on cause |
| `tool_call` | `tool_error/non_retryable` | 记录不可恢复错误 | yes if no alternative | `tool_error_unrecoverable` |
| `final_answer` | null | 设置终止状态，交给 evaluator 判定成功 | yes | `premature_final_answer`、`missing_tool_call`、`hallucinated_final_answer` |
| `ask_user` | null | 单轮任务中终止，多轮任务中等待用户输入 | depends | `unnecessary_ask_user` |
| invalid action | null/error | 记录非法动作 | yes or retry | `invalid_action` |
| any | max steps reached | 强制终止 | yes | `max_steps_exceeded` |

### 10.11 Transition 达到优秀的判断

本节达到优秀需要满足：

- 覆盖所有 action type。
- 覆盖主要 observation type。
- 明确每种情况下如何更新 `messages/history/progress/terminal_state/labels`。
- 区分模型失败、工具失败和环境失败。
- 能直接指导 `agent_state_machine.md` 和状态机代码实现。

当前本节已经满足以上条件，可以作为第一版状态机实现规格。

## 11. Trajectory 定义

一条完整 trajectory 是从用户任务开始，到 Agent 结束回答或失败终止的全过程。

可以表示为：

```json
{
  "task_id": "weather_001",
  "steps": [
    {
      "state": {},
      "action": {
        "type": "tool_call",
        "tool_name": "weather",
        "arguments": {
          "location": "上海",
          "date": "明天"
        }
      },
      "observation": {
        "type": "tool_result",
        "status": "success",
        "result": {}
      }
    },
    {
      "state": {},
      "action": {
        "type": "final_answer",
        "content": "..."
      },
      "observation": null
    }
  ],
  "label": {
    "success": true,
    "failure_types": []
  }
}
```

Trajectory 是训练和评测的基本单位。

对于 supervised fine-tuning，可以把每个 `(state, expert_action)` 拆成一个训练样本。

对于 reinforcement learning 或 rejection sampling，可以把整条 trajectory 的 evaluator score 作为筛选或优化信号。

## 12. 从 Trajectory 到训练样本

一条 trajectory 可以被拆成多个 step-level training examples。

例如一条轨迹：

```text
s_0 -> a_0 -> o_1 -> s_1 -> a_1
```

可以拆成两个 SFT 样本：

```json
[
  {
    "sample_id": "weather_001_step_0",
    "input": {
      "model_input_state": "s_0"
    },
    "target": {
      "action": "a_0"
    },
    "metadata": {
      "task_id": "weather_001",
      "step": 0,
      "action_type": "tool_call"
    }
  },
  {
    "sample_id": "weather_001_step_1",
    "input": {
      "model_input_state": "s_1"
    },
    "target": {
      "action": "a_1"
    },
    "metadata": {
      "task_id": "weather_001",
      "step": 1,
      "action_type": "final_answer"
    }
  }
]
```

这里最重要的是：

- `input.model_input_state` 只能包含模型推理时真实可见的信息。
- `target.action` 是希望模型学习的专家动作。
- `metadata` 可以保存任务类型、错误类型、难度等信息，但默认不作为模型输入。

如果一条 trajectory 是失败轨迹，也可以用于训练，但用途不同：

- 用于 evaluator 训练：学习识别失败类型。
- 用于 preference data：和成功轨迹组成正负样本对。
- 用于 targeted repair：把失败 state 对应到修正后的正确 action。

失败轨迹不应该直接当作普通 SFT 正样本，否则模型会模仿错误行为。

## 13. Canonical Schemas

为了让数据构造、训练和评测能够对齐，需要定义一组 canonical schemas。

这些 schema 的目标不是覆盖所有未来情况，而是给第一版实现提供稳定的数据契约。

### 13.1 Task Schema

`task` 描述用户希望 Agent 完成的目标。

必填字段：

```json
{
  "task_id": "weather_001",
  "task_type": "single_tool",
  "user_query": "查询明天上海的天气，并告诉我是否适合户外跑步。",
  "available_tools": ["weather"],
  "success_criteria": [
    "must_call_tool:weather",
    "must_use_observation",
    "must_answer_running_advice"
  ]
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `task_id` | string | yes | 任务唯一 ID |
| `task_type` | string | yes | 任务类型，例如 `no_tool`、`single_tool`、`multi_tool`、`recovery` |
| `user_query` | string | yes | 用户原始请求 |
| `available_tools` | string[] | yes | 当前任务可用工具 |
| `success_criteria` | string[] | yes | 任务成功条件，建议使用可解析标签 |
| `constraints` | string[] | no | 额外约束，例如禁止调用某工具、必须先查某信息 |
| `reference_answer` | string | no | 参考最终答案 |
| `reference_trajectory_id` | string | no | 对应专家轨迹 ID |
| `metadata` | object | no | 领域、难度、来源等辅助信息 |

### 13.2 Tool Schema

`tool` 描述工具的输入、输出和错误结构。

```json
{
  "name": "weather",
  "description": "查询指定地点和日期的天气。",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string"
      },
      "date": {
        "type": "string"
      }
    },
    "required": ["location", "date"],
    "additionalProperties": false
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "temperature": {
        "type": "string"
      },
      "rain_probability": {
        "type": "string"
      },
      "wind": {
        "type": "string"
      }
    },
    "required": ["temperature", "rain_probability"]
  },
  "error_schema": {
    "type": "object",
    "properties": {
      "code": {
        "type": "string"
      },
      "message": {
        "type": "string"
      },
      "retryable": {
        "type": "boolean"
      }
    },
    "required": ["code", "message", "retryable"]
  }
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `name` | string | yes | 工具名称，也是 action 中的 `tool_name` |
| `description` | string | yes | 工具用途说明 |
| `input_schema` | object | yes | 工具参数 JSON schema |
| `output_schema` | object | yes | 工具成功返回结果 schema |
| `error_schema` | object | yes | 工具错误返回 schema |
| `side_effects` | string[] | no | 是否有写文件、发请求、下单等副作用 |
| `metadata` | object | no | 版本、owner、mock 信息等 |

### 13.3 Action Schema

`action` 是模型要预测的目标。

`tool_call`：

```json
{
  "type": "tool_call",
  "tool_name": "weather",
  "arguments": {
    "location": "上海",
    "date": "明天"
  }
}
```

`final_answer`：

```json
{
  "type": "final_answer",
  "content": "明天上海整体适合户外跑步。"
}
```

`ask_user`：

```json
{
  "type": "ask_user",
  "content": "你想查询哪个城市的天气？"
}
```

字段说明：

| 字段 | 类型 | 必填 | 适用 action | 说明 |
|---|---|---:|---|---|
| `type` | string | yes | all | `tool_call`、`final_answer` 或 `ask_user` |
| `tool_name` | string | conditional | tool_call | 要调用的工具名称 |
| `arguments` | object | conditional | tool_call | 工具参数，必须满足对应 `input_schema` |
| `content` | string | conditional | final_answer / ask_user | 最终回答或反问用户的内容 |

合法性约束：

- 当 `type=tool_call` 时，必须包含 `tool_name` 和 `arguments`。
- 当 `type=final_answer` 时，必须包含 `content`，且不应包含 `tool_name`。
- 当 `type=ask_user` 时，必须包含 `content`，且内容应该针对缺失信息。

### 13.4 Observation Schema

`observation` 是环境执行 action 后返回的信息。

成功结果：

```json
{
  "type": "tool_result",
  "tool_name": "weather",
  "status": "success",
  "result": {
    "temperature": "18-24C",
    "rain_probability": "20%",
    "wind": "light"
  }
}
```

错误结果：

```json
{
  "type": "schema_error",
  "tool_name": "weather",
  "status": "failed",
  "error": {
    "code": "MISSING_REQUIRED_FIELD",
    "message": "Missing required field: location",
    "retryable": true
  }
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `type` | string | yes | `tool_result`、`schema_error`、`tool_error`、`empty_result`、`system_error` |
| `tool_name` | string | conditional | 对 tool action 必填 |
| `status` | string | yes | `success` 或 `failed` |
| `result` | object | conditional | 工具成功时返回 |
| `error` | object | conditional | 工具失败或校验失败时返回 |
| `raw` | object | no | 原始工具返回，供 debug 使用 |

### 13.5 Trajectory Schema

`trajectory` 是一次完整任务执行过程。

```json
{
  "trajectory_id": "traj_weather_001_gold",
  "task_id": "weather_001",
  "source": "human_annotated",
  "steps": [
    {
      "step_index": 0,
      "model_input_state": {},
      "annotation_state": {},
      "action": {
        "type": "tool_call",
        "tool_name": "weather",
        "arguments": {
          "location": "上海",
          "date": "明天"
        }
      },
      "observation": {
        "type": "tool_result",
        "tool_name": "weather",
        "status": "success",
        "result": {}
      }
    },
    {
      "step_index": 1,
      "model_input_state": {},
      "annotation_state": {},
      "action": {
        "type": "final_answer",
        "content": "..."
      },
      "observation": null
    }
  ],
  "terminal_state": {
    "reason": "final_answer",
    "success": true
  },
  "labels": {
    "success": true,
    "failure_types": []
  }
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `trajectory_id` | string | yes | 轨迹唯一 ID |
| `task_id` | string | yes | 对应任务 ID |
| `source` | string | yes | `human_annotated`、`agent_log`、`model_generated`、`simulated` |
| `steps` | object[] | yes | 按顺序保存每一步 |
| `steps[].step_index` | number | yes | 从 0 开始 |
| `steps[].model_input_state` | object | yes | 模型真实输入 |
| `steps[].annotation_state` | object | no | 标注/evaluator 辅助信息 |
| `steps[].action` | object | yes | 当前步动作 |
| `steps[].observation` | object/null | yes | tool action 后有 observation；final action 后可为 null |
| `terminal_state` | object | yes | 终止原因和终止状态 |
| `labels` | object | yes | 成功标签和失败类型 |

`terminal_state.reason` 可选值：

- `final_answer`
- `ask_user`
- `max_steps_exceeded`
- `tool_error_unrecoverable`
- `invalid_action`
- `manual_stop`

### 13.6 SFT Sample Schema

`sft_sample` 是从 trajectory step 拆出来的训练样本。

```json
{
  "sample_id": "weather_001_step_0",
  "task_id": "weather_001",
  "trajectory_id": "traj_weather_001_gold",
  "step_index": 0,
  "input": {
    "model_input_state": {}
  },
  "target": {
    "action": {
      "type": "tool_call",
      "tool_name": "weather",
      "arguments": {
        "location": "上海",
        "date": "明天"
      }
    }
  },
  "metadata": {
    "task_type": "single_tool",
    "action_type": "tool_call",
    "source": "human_annotated"
  }
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `sample_id` | string | yes | 样本唯一 ID |
| `task_id` | string | yes | 对应任务 |
| `trajectory_id` | string | yes | 来源轨迹 |
| `step_index` | number | yes | 来源轨迹中的 step |
| `input.model_input_state` | object | yes | 模型输入 |
| `target.action` | object | yes | 专家动作 |
| `metadata` | object | no | 只用于分析和过滤，默认不喂给模型 |

### 13.7 Eval Result Schema

`eval_result` 保存 evaluator 对 step 或 trajectory 的判断。

```json
{
  "eval_id": "eval_weather_001",
  "task_id": "weather_001",
  "trajectory_id": "traj_weather_001_model_a",
  "level": "trajectory",
  "score": 1.0,
  "metrics": {
    "tool_selection_accuracy": 1.0,
    "schema_valid_rate": 1.0,
    "execution_success_rate": 1.0,
    "task_success": true,
    "hallucination": false
  },
  "failure_types": [],
  "details": [
    {
      "step_index": 0,
      "score": 1.0,
      "failure_types": []
    }
  ]
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `eval_id` | string | yes | 评测结果唯一 ID |
| `task_id` | string | yes | 对应任务 |
| `trajectory_id` | string | yes | 被评测轨迹 |
| `level` | string | yes | `step` 或 `trajectory` |
| `score` | number | yes | 总分，范围建议为 0 到 1 |
| `metrics` | object | yes | 各项指标 |
| `failure_types` | string[] | yes | 聚合后的失败标签 |
| `details` | object[] | no | step-level 明细 |

### 13.8 字段使用边界

为了避免训练泄漏，需要明确字段使用范围：

| 字段类别 | 可进入模型输入 | 可用于 target | 可用于 evaluator | 可用于分析 |
|---|---:|---:|---:|---:|
| `model_input_state` | yes | no | yes | yes |
| `annotation_state` | no | no | yes | yes |
| `action` in expert trajectory | no | yes | yes | yes |
| `observation` | yes, if already occurred | no | yes | yes |
| `labels` | no | no | yes | yes |
| `metadata` | no by default | no | yes | yes |
| `eval_result` | no | no | no | yes |

这一组 schema 达到第一版可实现要求：数据构造脚本、SFT 样本导出脚本和 deterministic evaluator 可以共享同一套字段约定。

## 14. Policy 与训练目标

Agent policy 可以写成：

$$
\pi_\theta(a_t | s_t)
$$

也就是模型根据当前 state 预测下一步 action。

根据数据和训练方式不同，可以有几种目标。

### 14.1 Imitation Learning / SFT

如果有专家轨迹 $\tau^*$，可以训练模型模仿专家 action：

$$
\minimize\ -\log \pi_\theta(a_t^* | s_t)
$$

训练样本是：

```json
{
  "input": {
    "state": {}
  },
  "target": {
    "action": {}
  }
}
```

适合场景：

- 有人工标注的正确工具调用。
- 有可验证的专家轨迹。
- 希望模型先学会基本工具使用格式和决策模式。

### 14.2 Rejection Sampling

如果可以让模型生成多条 trajectory，并用 evaluator 评分，则可以保留高质量样本：

$$
\begin{aligned}
&\text{sample } \tau_1, \tau_2, \ldots, \tau_k \text{ from } \pi_\theta \\
&\text{keep } \tau_i \text{ if } \text{Evaluator}(\tau_i) \geq \text{threshold}
\end{aligned}
$$

适合场景：

- 人工标注成本高。
- evaluator 比较可靠。
- 需要扩大训练数据。

### 14.3 Reinforcement Learning

如果 evaluator 可以提供 reward，可以优化期望回报：

$$
\maximize\ \mathbb{E}[R(\tau)]
$$

其中 reward 可以来自：

- schema 是否合法。
- 工具是否执行成功。
- 参数是否正确。
- 最终任务是否完成。
- 是否避免幻觉。

RL 更依赖 evaluator 的稳定性。如果 evaluator 不可靠，模型可能学到投机行为。

## 15. Feedback 与 Reward 设计

Feedback 是训练或评测时对 action / trajectory 的质量信号。

可以分成两层：

### 15.1 Step-level Feedback

用于判断单步 action 是否正确。

示例：

```json
{
  "step": 0,
  "tool_correct": true,
  "schema_valid": true,
  "arguments_correct": true,
  "error_types": []
}
```

适合评估：

- 工具选择。
- 参数结构。
- 参数值。
- schema 合法性。

### 15.2 Trajectory-level Feedback

用于判断完整任务是否成功。

示例：

```json
{
  "task_success": true,
  "final_answer_grounded": true,
  "unnecessary_tool_calls": 0,
  "recovered_from_errors": true,
  "failure_types": []
}
```

适合评估：

- 最终任务是否完成。
- 调用顺序是否合理。
- 是否出现幻觉。
- 是否能从错误 observation 中恢复。

### 15.3 Reward Table

如果进入 rejection sampling 或 RL 阶段，需要把 evaluator 输出转成 reward。

第一版 reward 可以按可解释组件构造：

| Reward Component | 来源 | 建议范围 | 说明 |
|---|---|---:|---|
| `tool_selection_reward` | step evaluator | 0/1 | 工具是否选对 |
| `schema_reward` | schema validator | 0/1 | tool call 是否合法 |
| `argument_reward` | reference action / semantic match | 0-1 | 参数是否正确 |
| `execution_reward` | environment | 0/1 | 工具是否成功执行 |
| `recovery_reward` | recovery evaluator | 0-1 | 错误后是否正确恢复 |
| `groundedness_reward` | groundedness evaluator | 0-1 | final answer 是否基于 observation |
| `task_success_reward` | trajectory evaluator | 0/1 | 任务是否最终完成 |

可以先使用简单组合：

$$
\begin{aligned}
R(\tau) =&\ 0.40 \times \text{task\_success\_reward} \\
        &+ 0.20 \times \text{average\_step\_reward} \\
        &+ 0.20 \times \text{groundedness\_reward} \\
        &+ 0.10 \times \text{recovery\_reward} \\
        &+ 0.10 \times \text{efficiency\_reward}
\end{aligned}
$$

其中：

$$
\text{efficiency\_reward} = \min\left(\frac{\text{reference\_steps}}{\text{model\_steps}}, 1.0\right)
$$

这可以惩罚无意义的重复调用，但不应惩罚必要的多步推理。

### 15.4 不同训练阶段如何使用 Feedback

| 训练阶段 | 使用的数据 | 使用的 feedback | 目标 |
|---|---|---|---|
| SFT cold start | expert trajectories | `expert_action` | 学会基本 action 格式和工具决策 |
| SFT repair | failed state + corrected action | failure label + corrected action | 学会从错误状态修正 |
| Rejection sampling | model generated trajectories | trajectory score | 筛选高质量自生成样本 |
| Preference training | success/failure trajectory pairs | pairwise preference | 偏好更短、更准、更 grounded 的轨迹 |
| RL | online rollouts | reward | 优化任务成功率和恢复能力 |

第一阶段建议只做：

- SFT cold start。
- deterministic evaluator。
- failure report。

不要过早进入 RL。原因是 evaluator 尚未覆盖所有语义情况时，RL 容易放大 evaluator 漏洞。

## 16. Evaluator 定义

Evaluator 应该被定义成函数，而不仅是指标列表。

### 16.1 Step Evaluator

$$
E_{\text{step}}(s_t, a_t, \text{reference\_action}_t) \to \text{step\_score}
$$

输入：

- 当前 state。
- 模型 action。
- 参考 action，或规则定义的正确 action。

输出：

```json
{
  "score": 1.0,
  "tool_selection": "correct",
  "schema_valid": true,
  "argument_match": "exact",
  "failure_types": []
}
```

### 16.2 Trajectory Evaluator

$$
E_{\text{traj}}(\tau, \text{task\_spec}) \to \text{traj\_score}
$$

输入：

输入：

- 完整 trajectory。
- task spec，包括 success criteria。

输出：

```json
{
  "score": 1.0,
  "task_success": true,
  "step_success_rate": 1.0,
  "execution_success_rate": 1.0,
  "hallucination": false,
  "failure_types": []
}
```

第一版 evaluator 指标：

- `tool_selection_accuracy`
- `schema_valid_rate`
- `argument_exact_match`
- `argument_semantic_match`
- `execution_success_rate`
- `recovery_success_rate`
- `task_success_rate`
- `hallucination_rate`

## 17. Evaluator 的可实现规则

Evaluator 应优先使用确定性规则，只有在规则无法覆盖时再引入模型判断。

### 17.1 可以用规则直接判断的部分

这些指标通常可以稳定实现：

- `schema_valid`：使用 JSON schema validator 判断。
- `missing_argument`：检查 required fields。
- `wrong_argument_type`：检查字段类型。
- `tool_execution_success`：检查工具返回 status。
- `empty_result`：检查 result 是否为空。
- `unnecessary_tool_call`：对无工具任务检查是否产生 tool action。

示例：

```text
if action.type == "tool_call":
    schema_valid = validate(action.arguments, tool.schema)
```

### 17.2 需要参考答案或标注判断的部分

这些指标依赖 reference action 或 task spec：

- `tool_selection_accuracy`
- `argument_exact_match`
- `wrong_order`
- `missing_tool_call`
- `task_success`

示例：

```text
tool_selection_accuracy = action.tool_name == reference_action.tool_name
argument_exact_match = action.arguments == reference_action.arguments
```

### 17.3 需要语义判断的部分

这些指标难以只靠规则完成：

- `argument_semantic_match`
- `final_answer_grounded`
- `hallucinated_final_answer`
- `task_success` 在开放式任务中的判断。

可选实现方式：

- 人工标注。
- LLM judge。
- 规则加 LLM judge。
- 把最终答案转成结构化 claim 后逐条验证。

如果使用 LLM judge，需要记录 judge prompt、judge model、temperature 和输出理由，否则结果难以复现。

### 17.4 推荐优先级

第一版 evaluator 可以按以下优先级实现：

1. 先实现 schema 和工具执行相关规则。
2. 再实现 reference action 的 exact match。
3. 再实现 trajectory-level success criteria。
4. 最后处理 semantic match 和 hallucination。

这样可以先得到一个稳定、可复现的 evaluator baseline，再逐步覆盖更复杂的语义判断。

## 18. Failure Decision Table

Failure decision table 用来把模型行为映射成稳定的失败标签。

它的目标是让 evaluator 输出不仅有分数，还有可解释诊断。

### 18.1 判定输入

Evaluator 判定 failure type 时，至少需要以下输入：

```json
{
  "task": {},
  "tool_specs": [],
  "reference_trajectory": {},
  "model_trajectory": {},
  "eval_config": {
    "max_steps": 5,
    "allow_ask_user": true,
    "allow_semantic_argument_match": false
  }
}
```

不同失败类型需要的输入不同：

- schema 类错误只需要 `action` 和 `tool_specs`。
- exact match 类错误需要 `reference_action`。
- 顺序类错误需要完整 `reference_trajectory`。
- groundedness 和 hallucination 需要 `observation` 与 final answer。

### 18.2 Step-level Failure Rules

| Failure Type | 触发条件 | 所需输入 | 可规则判断 | 严重程度 | 说明 |
|---|---|---|---:|---|---|
| `invalid_action` | action 无法解析，或 `type` 不在允许集合中 | model action | yes | high | action 层面的格式失败 |
| `wrong_tool` | `action.type=tool_call`，但 `tool_name != reference.tool_name` | model action, reference action | yes | high | 工具选择错误 |
| `missing_tool_call` | reference 需要 tool call，但模型输出 `final_answer` 或 `ask_user` | model action, reference action | yes | high | 应查工具却直接回答或反问 |
| `unnecessary_tool_call` | reference 不需要工具，但模型输出 `tool_call` | model action, reference action/task spec | yes | medium | 多余工具调用 |
| `missing_argument` | required field 缺失 | model action, tool input schema | yes | high | 参数结构错误 |
| `wrong_argument_type` | 参数类型不符合 schema | model action, tool input schema | yes | high | 参数类型错误 |
| `invalid_schema` | 参数无法通过工具 input schema | model action, tool input schema | yes | high | 包含缺字段、错类型、多余字段等 |
| `wrong_argument_value` | 参数值与 reference 不一致，且不满足语义等价 | model action, reference action | partly | high | exact match 可规则判断，semantic match 可能需要 judge |
| `unnecessary_ask_user` | 信息充足时模型仍反问用户 | model action, task spec/reference action | partly | medium | 需要判断任务信息是否充足 |
| `premature_final_answer` | 仍有未满足需求时输出 final answer | model action, annotation state/task criteria | partly | high | 需要 open requirements 或 reference trajectory |
| `ignored_observation` | final answer 或下一步 action 没有使用关键 observation | model trajectory, observations | partly | high | 通常需要语义判断 |
| `looping_tool_call` | 重复调用同一工具和同一参数，且没有新信息 | model trajectory | yes | medium | 常见于失败恢复不当 |
| `max_steps_exceeded` | 超过环境最大步数仍未终止 | model trajectory, eval config | yes | high | 环境强制终止 |

### 18.3 Trajectory-level Failure Rules

| Failure Type | 触发条件 | 所需输入 | 可规则判断 | 严重程度 | 说明 |
|---|---|---|---:|---|---|
| `wrong_order` | 工具调用顺序违反 reference 或任务依赖 | model trajectory, reference trajectory/task dependency | yes/partly | high | 多工具任务中常见 |
| `missing_required_step` | 缺少完成任务所需的关键步骤 | model trajectory, success criteria | partly | high | 例如没查天气却给建议 |
| `poor_recovery` | 出现可恢复错误后，模型没有修正、重试或换工具 | model trajectory, observations | partly | medium/high | 依赖 recovery 规则 |
| `tool_error_unrecoverable` | 工具不可恢复错误导致任务失败 | model trajectory, observations | yes | medium | 不一定归因于模型 |
| `hallucinated_final_answer` | final answer 包含 observation 或 reference 中不存在的关键事实 | final answer, observations/reference | partly | high | 需要 groundedness 判断 |
| `incomplete_final_answer` | final answer 没有覆盖 success criteria | final answer, task spec | partly | high | 开放式任务可能需要 judge |
| `contradict_observation` | final answer 与工具 observation 明显矛盾 | final answer, observations | partly | high | 可用规则或 claim checking |
| `task_failed` | success criteria 未全部满足 | full trajectory, task spec | partly | high | 轨迹级最终失败 |

### 18.4 Failure 优先级

同一步可能触发多个 failure type。为了输出稳定诊断，需要定义优先级。

推荐优先级：

1. `invalid_action`
2. `invalid_schema`
3. `wrong_tool`
4. `missing_tool_call`
5. `missing_argument`
6. `wrong_argument_type`
7. `wrong_argument_value`
8. `wrong_order`
9. `premature_final_answer`
10. `hallucinated_final_answer`
11. `ignored_observation`
12. `poor_recovery`
13. `unnecessary_tool_call`
14. `unnecessary_ask_user`

优先级的作用：

- 用于选择 primary failure type。
- 避免同一个错误被重复归因。
- 保持 failure report 稳定。

但 evaluator 仍应保留多标签能力：

```json
{
  "primary_failure_type": "invalid_schema",
  "failure_types": [
    "invalid_schema",
    "missing_argument"
  ]
}
```

### 18.5 Score Aggregation

Evaluator 可以先输出 step-level scores，再聚合成 trajectory-level score。

Step score 示例：

```json
{
  "step_index": 0,
  "score": 0.75,
  "metrics": {
    "tool_correct": true,
    "schema_valid": true,
    "arguments_correct": false,
    "execution_success": true
  },
  "failure_types": [
    "wrong_argument_value"
  ]
}
```

第一版 step score 可以使用简单加权：

$$
\begin{aligned}
\text{step\_score} =&\ 0.30 \times \text{tool\_correct} \\
                  &+ 0.25 \times \text{schema\_valid} \\
                  &+ 0.25 \times \text{arguments\_correct} \\
                  &+ 0.20 \times \text{execution\_success}
\end{aligned}
$$

Final answer step 可以使用：

$$
\begin{aligned}
\text{final\_score} =&\ 0.40 \times \text{task\_requirements\_satisfied} \\
                   &+ 0.30 \times \text{grounded\_in\_observation} \\
                   &+ 0.20 \times \text{no\_contradiction} \\
                   &+ 0.10 \times \text{answer\_completeness}
\end{aligned}
$$

Trajectory score 可以使用：

$$
\begin{aligned}
\text{trajectory\_score} =&\ 0.60 \times \text{task\_success} \\
                        &+ 0.25 \times \text{average\_step\_score} \\
                        &+ 0.15 \times \text{recovery\_score}
\end{aligned}
$$

第一版也可以更保守：只输出 pass/fail 和 failure types，不强行使用连续分数。

推荐第一阶段：

- deterministic evaluator 输出 pass/fail。
- 同时输出 step-level metrics。
- 暂不把 LLM judge 分数混入总分。
- 等规则稳定后，再引入连续分数或 reward。

### 18.6 Evaluation Report

一次评测应该输出可聚合报告：

```json
{
  "run_id": "eval_run_001",
  "num_tasks": 100,
  "metrics": {
    "task_success_rate": 0.82,
    "tool_selection_accuracy": 0.91,
    "schema_valid_rate": 0.96,
    "argument_exact_match": 0.84,
    "execution_success_rate": 0.88,
    "hallucination_rate": 0.07
  },
  "failure_breakdown": {
    "wrong_tool": 5,
    "missing_argument": 3,
    "wrong_argument_value": 12,
    "premature_final_answer": 6,
    "hallucinated_final_answer": 7
  }
}
```

这个 report 的作用：

- 判断模型整体能力。
- 找出最需要补数据的失败类型。
- 比较不同训练版本。
- 指导下一轮 targeted data construction。

### 18.7 Evaluator Rules 达到优秀的判断

本节达到优秀需要满足：

- 每个主要 failure type 都有触发条件。
- 明确哪些 failure 可规则判断，哪些需要 reference，哪些需要语义判断。
- 支持 primary failure type 和 multi-label failure types。
- 有 step score、trajectory score 和 evaluation report 的聚合方式。
- 能直接指导 deterministic evaluator baseline 实现。

当前本节已经满足第一版 deterministic evaluator 的实现要求。语义类 failure 仍需要在 groundedness 章节继续细化。

## 19. Grounded Final Answer

Grounded final answer 用来判断模型最终回答是否基于已有 observation，而不是凭空编造。

在 tool-using Agent 中，最终回答的质量不能只看语言是否流畅，还要看它是否满足：

- 使用了必要工具结果。
- 没有加入 observation 中不存在的关键事实。
- 没有与 observation 矛盾。
- 覆盖了用户任务的核心要求。

### 19.1 核心定义

给定：

```text
O = {o_1, o_2, ..., o_k}
```

表示当前 trajectory 中所有可用 observations。

模型最终回答：

```text
y = final_answer
```

Groundedness 评测目标是判断：

```text
Every task-relevant factual claim in y is supported by O or task context.
```

也就是说，最终回答里的关键事实要么来自工具 observation，要么来自用户输入或任务上下文。

### 19.2 Claim 分类

可以先把 final answer 中的内容分成四类：

| Claim Type | 说明 | 是否需要 observation 支持 | 示例 |
|---|---|---:|---|
| `observed_fact` | 来自工具结果的事实 | yes | “明天上海降雨概率 20%” |
| `derived_judgment` | 基于 observation 推导出的判断 | yes | “整体适合户外跑步” |
| `user_context` | 来自用户输入的事实 | no, 但需与用户输入一致 | “你想在上海跑步” |
| `generic_advice` | 通用建议 | no, 但不能与 observation 冲突 | “出门前再确认实时天气” |

需要重点检查的是：

- `observed_fact` 是否真的出现在 observation 中。
- `derived_judgment` 是否能由 observation 合理推出。
- `generic_advice` 是否引入了新的具体事实。

### 19.3 Groundedness Labels

Evaluator 可以为 final answer 输出以下标签：

```json
{
  "grounded_in_observation": true,
  "uses_required_observation": true,
  "has_unsupported_claim": false,
  "contradicts_observation": false,
  "covers_task_requirements": true,
  "failure_types": []
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `grounded_in_observation` | boolean | 关键事实是否由 observation 或 task context 支持 |
| `uses_required_observation` | boolean | 是否使用了完成任务所必需的 observation |
| `has_unsupported_claim` | boolean | 是否出现无依据关键事实 |
| `contradicts_observation` | boolean | 是否和 observation 明显矛盾 |
| `covers_task_requirements` | boolean | 是否覆盖 success criteria |
| `failure_types` | string[] | 对应失败标签 |

### 19.4 可规则判断的 Groundedness

第一版可以先处理结构化 observation 中的 exact match。

适用场景：

- observation 是结构化 JSON。
- final answer 中包含可抽取的数值、地点、日期、状态。
- success criteria 可以映射到具体字段。

示例 observation：

```json
{
  "temperature": "18-24C",
  "rain_probability": "20%",
  "wind": "light"
}
```

final answer：

```text
明天上海气温 18-24C，降雨概率 20%，风力较小，整体适合户外跑步。
```

规则判断：

| 检查项 | 规则 | 结果 |
|---|---|---|
| 温度 | final answer 中的 `18-24C` 与 observation 一致 | pass |
| 降雨概率 | final answer 中的 `20%` 与 observation 一致 | pass |
| 风 | “风力较小” 与 `wind=light` 语义一致 | pass, 可用映射表 |
| 跑步建议 | 基于低降雨、温和气温、微风推导 | pass |

### 19.5 Unsupported Claim

如果 final answer 中出现 observation 和任务上下文都不支持的关键事实，则标记：

```text
has_unsupported_claim = true
failure_types includes hallucinated_final_answer
```

示例：

Observation：

```json
{
  "temperature": "18-24C",
  "rain_probability": "20%"
}
```

Final answer：

```text
明天上海空气质量优良，紫外线很弱，非常适合跑步。
```

判定：

| Claim | 判定 |
|---|---|
| “空气质量优良” | unsupported |
| “紫外线很弱” | unsupported |
| “非常适合跑步” | partially supported, 但程度可能过强 |

### 19.6 Contradiction

如果 final answer 和 observation 明显冲突，则标记：

```text
contradicts_observation = true
failure_types includes contradict_observation
```

示例：

Observation：

```json
{
  "rain_probability": "90%",
  "wind": "strong"
}
```

Final answer：

```text
明天基本不会下雨，风力较小，很适合户外跑步。
```

判定：

| Claim | Observation | 结果 |
|---|---|---|
| “基本不会下雨” | `rain_probability=90%` | contradiction |
| “风力较小” | `wind=strong` | contradiction |
| “很适合户外跑步” | 高降雨 + 强风 | unsupported/contradiction |

### 19.7 Derived Judgment

有些回答不是直接复述 observation，而是基于 observation 做判断。

例如：

```text
temperature=18-24C
rain_probability=20%
wind=light
=> suitable_for_running = true
```

这类 derived judgment 需要定义领域规则或 judge prompt。

第一版可以采用简单规则：

```text
if rain_probability <= 30%
and wind in ["light", "moderate"]
and temperature is within comfortable range:
    running_advice = "suitable"
else:
    running_advice = "not_suitable_or_use_caution"
```

对于不同任务域，需要不同的 derivation rules。

### 19.8 Claim Checking 流程

更通用的 groundedness evaluator 可以分三步：

1. 从 final answer 中抽取 task-relevant claims。
2. 对每个 claim 判断支持关系：`supported`、`unsupported`、`contradicted`、`not_checkable`。
3. 聚合成 final answer groundedness score。

输出示例：

```json
{
  "claims": [
    {
      "text": "明天上海气温 18-24C",
      "type": "observed_fact",
      "status": "supported",
      "evidence": "observation.weather.temperature"
    },
    {
      "text": "空气质量优良",
      "type": "observed_fact",
      "status": "unsupported",
      "evidence": null
    }
  ],
  "grounded_in_observation": false,
  "failure_types": [
    "hallucinated_final_answer"
  ]
}
```

### 19.9 第一阶段实现建议

第一阶段不必直接实现完整 LLM judge。

推荐顺序：

1. 对结构化字段做 exact match。
2. 对常见枚举值做映射，例如 `light -> 风力较小`。
3. 对 success criteria 做覆盖检查。
4. 对明显矛盾做规则检查。
5. 最后再引入 LLM judge 处理开放式 claim。

第一阶段可以先输出三类结果：

- `grounded`
- `unsupported_claim`
- `contradiction`

这足够支撑 hallucination rate 的初版统计。

### 19.10 LLM Judge 使用边界

当使用 LLM judge 判断 groundedness 时，必须固定：

- judge model。
- judge prompt。
- temperature。
- 输入 observation。
- 输入 final answer。
- 输出 JSON schema。

Judge 输出建议：

```json
{
  "grounded_in_observation": true,
  "unsupported_claims": [],
  "contradictions": [],
  "reason": "Final answer only uses weather facts returned by the tool."
}
```

LLM judge 不应直接替代规则 evaluator，而应作为规则无法覆盖时的补充。

### 19.11 Groundedness 达到优秀的判断

本节达到优秀需要满足：

- 定义 grounded final answer 的判定目标。
- 区分 observed fact、derived judgment、user context 和 generic advice。
- 明确 unsupported claim 和 contradiction 的触发条件。
- 给出 claim checking 流程和输出 schema。
- 给出第一阶段可实现方案和 LLM judge 使用边界。

当前本节已经达到优秀，可以指导 hallucination 和 final answer groundedness 的第一版 evaluator 实现。

## 20. Multi-tool Dependency

多工具任务是 tool-using Agent 的核心场景之一。

单工具任务只需要判断“是否调用正确工具、参数是否正确”，而多工具任务还需要建模：

- 工具之间是否有依赖关系。
- 后续工具参数是否来自前序 observation。
- 多个工具调用是否允许交换顺序。
- 某个工具失败后是否可以使用替代工具。

### 20.1 多工具任务表示

多工具 task 可以增加 `tool_plan_spec`：

```json
{
  "task_id": "travel_001",
  "task_type": "multi_tool_sequential",
  "user_query": "帮我查明天从上海到北京的航班，并根据北京天气建议是否需要带伞。",
  "available_tools": ["flight_search", "weather"],
  "tool_plan_spec": {
    "nodes": [
      {
        "id": "search_flight",
        "tool_name": "flight_search",
        "required": true
      },
      {
        "id": "check_weather",
        "tool_name": "weather",
        "required": true
      }
    ],
    "edges": [
      {
        "from": "search_flight",
        "to": "check_weather",
        "type": "context_dependency",
        "description": "目的地城市来自航班查询任务或用户输入"
      }
    ]
  },
  "success_criteria": [
    "must_call_tool:flight_search",
    "must_call_tool:weather",
    "must_answer_umbrella_advice"
  ]
}
```

这里的 `tool_plan_spec` 不是必须暴露给模型。它主要用于标注、evaluator 和任务生成。

### 20.2 Dependency Graph

多工具依赖可以表示为 DAG：

```text
G = (V, E)
```

其中：

- `V` 是工具调用节点。
- `E` 是调用依赖边。

边类型可以包括：

| Edge Type | 说明 | 示例 |
|---|---|---|
| `data_dependency` | 后一个工具参数来自前一个工具结果 | 先查订单 ID，再查物流 |
| `context_dependency` | 后一个工具使用前一步确定的上下文 | 先查航班目的地，再查目的地天气 |
| `validation_dependency` | 后一个工具用于验证前一个结果 | 查网页后再查数据库确认 |
| `fallback_dependency` | 前一个工具失败后尝试替代工具 | 搜索 API 失败后用浏览器 |

### 20.3 参数绑定

多工具任务的关键是参数如何从 observation 派生。

可以用 `argument_bindings` 表示：

```json
{
  "argument_bindings": [
    {
      "target_step": "track_package",
      "target_argument": "tracking_id",
      "source_step": "find_order",
      "source_path": "result.tracking_id",
      "transform": "identity"
    }
  ]
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `target_step` | 使用该参数的工具节点 |
| `target_argument` | 目标工具参数名 |
| `source_step` | 参数来源工具节点 |
| `source_path` | 从 observation 中取值的路径 |
| `transform` | 可选转换，例如日期格式化、单位转换、字符串清洗 |

Evaluator 可以用这个绑定判断：

- 后续参数是否真的来自前序 observation。
- 模型是否编造了中间参数。
- 参数转换是否合理。

### 20.4 顺序约束与等价顺序

不是所有多工具任务都有严格顺序。

可以把顺序分成三类：

| 类型 | 说明 | 评测方式 |
|---|---|---|
| strict order | 必须按指定顺序调用 | exact sequence match |
| partial order | 只要求满足依赖关系 | DAG topological validity |
| unordered | 多个工具互不依赖 | set match |

示例：

```text
strict: login -> fetch_profile -> update_profile
partial: fetch_weather and fetch_calendar can happen in any order, then final_answer
unordered: 查询 A 股票价格和 B 股票价格
```

多工具 evaluator 不应该把所有顺序差异都判错。只要满足 DAG 依赖的拓扑序，就可以认为顺序有效。

### 20.5 多工具 Trajectory Evaluator

多工具 trajectory 可以增加以下指标：

| Metric | 说明 |
|---|---|
| `required_tools_covered` | 是否调用了所有必需工具 |
| `extra_tools_count` | 是否调用了多余工具 |
| `dependency_valid` | 调用顺序是否满足依赖图 |
| `argument_binding_valid` | 后续参数是否正确来自前序 observation |
| `parallel_order_equivalent` | 是否属于允许的等价顺序 |
| `multi_step_task_success` | 多工具任务是否完成 |

判定示例：

```text
dependency_valid = all(
  position(edge.from) < position(edge.to)
  for edge in dependency_edges
)
```

对于 unordered 工具集合：

```text
tool_set_match = set(model_tools) == set(reference_tools)
```

### 20.6 多工具失败类型

多工具任务新增 failure types：

| Failure Type | 触发条件 |
|---|---|
| `missing_required_tool` | 缺少必须调用的工具 |
| `extra_tool_call` | 调用了任务不需要的工具 |
| `dependency_violation` | 工具调用顺序违反依赖图 |
| `invalid_argument_binding` | 后续参数没有正确来自前序 observation |
| `lost_intermediate_result` | 工具返回了中间结果，但后续没有使用 |
| `wrong_parallel_order_judgment` | 把等价顺序误判为错误，属于 evaluator 问题 |

### 20.7 多工具达到优秀的判断

本节达到优秀需要满足：

- 能表示多工具任务的依赖图。
- 能区分 strict order、partial order 和 unordered。
- 能表示参数从前序 observation 到后续 action 的绑定。
- 能定义多工具 evaluator 指标和失败类型。
- 能避免把合法等价顺序误判为错误。

当前本节已经达到优秀，可以支撑多工具 trajectory 建模和 evaluator baseline。

## 21. Recovery Behavior

Recovery behavior 描述 Agent 遇到错误 observation 后，如何重试、修正、换工具、反问用户或停止。

它衡量的是 Agent 是否能从失败状态中恢复，而不是一次性永远不犯错。

### 21.1 Recovery State

当 observation 是错误或空结果时，transition 应更新 `recovery_state`：

```json
{
  "recovery_state": {
    "active": true,
    "error_type": "schema_error",
    "tool_name": "weather",
    "retryable": true,
    "attempt_count": 1,
    "max_attempts": 3,
    "last_failed_action": {
      "type": "tool_call",
      "tool_name": "weather",
      "arguments": {
        "date": "明天"
      }
    },
    "suggested_fix": {
      "missing_fields": ["location"]
    }
  }
}
```

Recovery state 不一定进入模型输入。若真实运行时工具错误会返回给模型，则错误 message 可以进入 runtime state；结构化 `suggested_fix` 是否进入模型输入需要按实验设定决定。

### 21.2 Recovery Action Types

遇到错误后，模型可以采取以下恢复动作：

| Recovery Action | 说明 | 示例 |
|---|---|---|
| `retry_same_tool_fixed_args` | 修正参数后重试同一工具 | 补上缺失的 `location` |
| `retry_same_tool_same_args` | 原参数重试 | 网络超时后重试 |
| `switch_tool` | 换替代工具 | 搜索 API 失败后使用浏览器 |
| `ask_user` | 缺少必要信息时反问用户 | 缺少城市 |
| `final_answer_with_limitation` | 无法完成时说明限制 | 工具不可用，不能确认实时天气 |
| `stop_failure` | 不可恢复时终止 | 权限不足且无替代工具 |

### 21.3 Recovery Policy

第一版可以使用规则定义合理恢复策略：

| Error Type | Retryable | 合理动作 | 不合理动作 |
|---|---:|---|---|
| `schema_error` | yes | 修正参数后重试 | 原样重试、直接编造答案 |
| `missing_argument` | yes | 从上下文补参数；无法补则 ask user | 忽略缺失字段 |
| `wrong_argument_type` | yes | 转换类型后重试 | 换无关工具 |
| `empty_result` | depends | 放宽参数、解释无结果、换工具 | 编造结果 |
| `timeout` | yes | 原参数重试或稍后重试 | 立即判任务成功 |
| `permission_denied` | no | 说明无法访问或请求授权 | 反复重试 |
| `rate_limited` | yes | 等待、降频、说明限制 | 高频重复调用 |
| `tool_error_non_retryable` | no | 换工具或说明无法完成 | 同样参数无限重试 |

### 21.4 Recovery Evaluator

Recovery evaluator 判断错误后的下一步 action 是否合理。

输入：

```json
{
  "error_observation": {},
  "next_action": {},
  "task": {},
  "tool_specs": [],
  "recovery_state": {}
}
```

输出：

```json
{
  "recovery_success": true,
  "recovery_type": "retry_same_tool_fixed_args",
  "attempt_count": 2,
  "failure_types": []
}
```

指标：

| Metric | 说明 |
|---|---|
| `recovery_success_rate` | 错误后采取合理恢复动作的比例 |
| `retry_fix_rate` | schema/参数错误后正确修参重试的比例 |
| `unrecoverable_stop_accuracy` | 不可恢复错误时是否正确停止或说明限制 |
| `loop_rate` | 是否出现无意义重复调用 |
| `max_attempts_exceeded_rate` | 是否超过最大尝试次数 |

### 21.5 Recovery Failure Types

| Failure Type | 触发条件 |
|---|---|
| `poor_recovery` | 错误后没有采取合理恢复动作 |
| `retry_without_fix` | 参数错误后原样重试 |
| `over_retry` | 超过最大重试次数 |
| `premature_abort` | 可恢复错误却直接放弃 |
| `hallucinate_after_error` | 工具失败后编造结果 |
| `wrong_fallback_tool` | 选择了不合适的替代工具 |
| `missing_user_clarification` | 缺少用户信息时没有反问 |

### 21.6 最大尝试次数

每个工具或错误类型应设置 `max_attempts`。

推荐默认值：

| 场景 | max attempts |
|---|---:|
| schema error | 2 |
| timeout | 2 |
| rate limited | 1 or defer |
| empty result | 2 |
| permission denied | 0 |
| non-retryable tool error | 0 |

超过最大次数后，模型应：

- 换工具，如果存在替代工具。
- 反问用户，如果缺少必要信息。
- 说明无法完成，并避免编造结果。

### 21.7 Recovery 达到优秀的判断

本节达到优秀需要满足：

- 有明确 recovery state。
- 有错误类型到合理恢复动作的策略表。
- 有最大尝试次数。
- 有 recovery evaluator 输入输出和指标。
- 有 recovery failure types。
- 能区分“工具不可用导致失败”和“模型恢复策略错误”。

当前本节已经达到优秀，可以支撑错误恢复任务的数据构造和 evaluator baseline。

## 22. Failure Taxonomy

为了训练和评测可解释，需要给失败类型建立稳定分类。

当前失败类型包括：

- `invalid_action`：action 无法解析或 action type 不合法。
- `wrong_tool`：选择了错误工具。
- `missing_tool_call`：应该调用工具但直接回答。
- `unnecessary_tool_call`：不需要工具时调用了工具。
- `missing_required_tool`：多工具任务中缺少必须调用的工具。
- `extra_tool_call`：调用了任务不需要的额外工具。
- `missing_argument`：缺少必填参数。
- `wrong_argument_type`：参数类型错误。
- `wrong_argument_value`：参数值错误。
- `invalid_argument_binding`：多工具任务中后续参数没有正确来自前序 observation。
- `invalid_schema`：tool call 不符合 schema。
- `wrong_order`：多工具调用顺序错误。
- `dependency_violation`：工具调用违反依赖图。
- `lost_intermediate_result`：工具返回了中间结果，但后续没有使用。
- `ignored_observation`：没有正确使用工具返回结果。
- `poor_recovery`：失败后没有正确修正或重试。
- `retry_without_fix`：参数错误后原样重试。
- `over_retry`：超过最大重试次数。
- `premature_abort`：可恢复错误却直接放弃。
- `hallucinate_after_error`：工具失败后编造结果。
- `premature_final_answer`：信息不足时提前回答。
- `hallucinated_final_answer`：最终答案包含无依据内容。
- `contradict_observation`：最终答案与 observation 矛盾。
- `incomplete_final_answer`：最终答案没有覆盖任务成功条件。
- `max_steps_exceeded`：超过最大步数仍未完成。

Failure taxonomy 的作用：

- 用于分析模型薄弱点。
- 用于构造 targeted training data。
- 用于 evaluator 输出可解释诊断。

## 23. 数据构造方式

训练数据可以来自多个来源。

### 23.1 人工标注专家轨迹

人工为任务写出正确 trajectory。

优点：

- 质量高。
- 可控性强。
- 适合冷启动。

缺点：

- 成本高。
- 覆盖面有限。

### 23.2 真实 Agent 日志

从实际运行日志中抽取 state/action/observation。

优点：

- 真实分布。
- 包含大量边界情况。

缺点：

- 需要清洗。
- 可能包含错误轨迹。
- 需要额外标注成功和失败原因。

### 23.3 模型自生成轨迹

让模型尝试完成任务，再用 evaluator 筛选。

优点：

- 扩展快。
- 可以覆盖大量任务变体。

缺点：

- 依赖 evaluator 质量。
- 容易引入系统性偏差。

### 23.4 环境模拟

构造可控任务环境，自动生成任务、工具结果和正确答案。

优点：

- 可自动生成大量数据。
- 成功条件清晰。

缺点：

- 和真实任务分布可能有差距。

### 23.5 数据验收标准

无论数据来自人工、日志、模型生成还是模拟环境，进入训练集前都需要验收。

| 检查项 | 标准 |
|---|---|
| schema valid | task、trajectory、action、observation 都符合 canonical schema |
| tool executable | trajectory 中的 tool action 能被 mock 或真实 executor 执行 |
| label complete | 每条 trajectory 有 success label 和 failure types |
| no leakage | SFT input 中不包含 `expected_next_action`、labels 或 evaluator 结果 |
| reference consistent | reference action 与 success criteria 一致 |
| reproducible | 数据来源、生成方式、工具版本可追踪 |
| balanced coverage | 不同 task type、action type、observation type 和 failure type 有基本覆盖 |

### 23.6 数据清洗流程

推荐清洗流程：

1. schema validation。
2. tool name 和 tool schema 对齐。
3. action 和 observation 顺序检查。
4. 去除泄漏字段。
5. 标注或修正 failure types。
6. 将成功轨迹导出为 SFT 正样本。
7. 将失败轨迹导出为 evaluator、preference 或 repair 数据。
8. 生成数据质量报告。

数据质量报告示例：

```json
{
  "num_tasks": 100,
  "num_trajectories": 120,
  "schema_valid_rate": 0.99,
  "label_coverage": 1.0,
  "task_type_distribution": {
    "no_tool": 10,
    "single_tool": 50,
    "multi_tool": 25,
    "recovery": 15
  },
  "rejected_samples": 4,
  "rejection_reasons": {
    "schema_invalid": 2,
    "label_missing": 1,
    "leakage_detected": 1
  }
}
```

### 23.7 数据来源达到优秀的判断

本节达到优秀需要满足：

- 明确每种数据来源的优缺点。
- 定义训练集准入标准。
- 定义清洗流程。
- 能区分成功轨迹、失败轨迹、repair 数据和 evaluator 数据的用途。
- 有数据质量报告格式。

当前本节已经达到优秀，可以指导第一版数据构造和清洗。

## 24. 任务类型划分

不同任务类型需要不同的 trajectory 和 evaluator。

第一版可以按以下维度划分：

- 单工具任务：只需要调用一个工具。
- 多工具任务：需要多个工具组合。
- 顺序依赖任务：后一个工具参数依赖前一个工具结果。
- 可并行任务：多个工具调用顺序不唯一。
- 需要反问任务：用户输入缺少必要信息。
- 错误恢复任务：工具失败后需要重试或修正。
- 无工具任务：模型应该直接回答，不调用工具。

这个划分可以帮助我们构造覆盖面更完整的训练集和测试集。

### 24.1 推荐任务占比

第一版数据集可以按以下比例构造：

| Task Type | 推荐占比 | 目的 |
|---|---:|---|
| `no_tool` | 10% | 学会不滥用工具 |
| `single_tool` | 35% | 学会基础工具选择和参数填写 |
| `single_tool_with_final` | 20% | 学会基于 observation 回答 |
| `multi_tool_sequential` | 15% | 学会顺序依赖 |
| `multi_tool_unordered` | 5% | 学会等价顺序 |
| `ask_user` | 5% | 学会信息不足时反问 |
| `recovery` | 10% | 学会错误恢复 |

### 24.2 每类任务最小样例

| Task Type | 用户任务 | 期望能力 |
|---|---|---|
| `no_tool` | “把这句话改得更正式” | 不调用工具，直接回答 |
| `single_tool` | “查明天上海天气” | 正确调用 weather |
| `single_tool_with_final` | “查天气并判断是否适合跑步” | 工具结果到最终判断 |
| `multi_tool_sequential` | “查订单，再查物流” | 后一步参数来自前一步 observation |
| `multi_tool_unordered` | “查 A 和 B 两只股票价格” | 顺序可交换 |
| `ask_user` | “帮我查天气” | 缺少地点时反问 |
| `recovery` | “查上海天气”，首次缺 location | schema error 后修正 |

### 24.3 覆盖标准

任务集达到优秀需要满足：

- 覆盖所有 action type。
- 覆盖所有主要 observation type。
- 覆盖主要 failure types。
- 包含成功轨迹和失败轨迹。
- 包含至少一类多工具顺序依赖和一类等价顺序任务。
- 包含至少一类错误恢复任务。

## 25. 最小完整样例

### 25.1 Task

```json
{
  "task_id": "weather_001",
  "user_query": "查询明天上海的天气，并告诉我是否适合户外跑步。",
  "available_tools": ["weather"],
  "success_criteria": [
    "必须查询上海明天天气",
    "必须根据工具结果判断是否适合跑步",
    "最终回答必须基于 observation"
  ]
}
```

### 25.2 Step 0

State:

```json
{
  "user_query": "查询明天上海的天气，并告诉我是否适合户外跑步。",
  "tools": ["weather"],
  "history": [],
  "progress": {
    "open_requirements": [
      "查询上海明天天气",
      "判断是否适合跑步"
    ]
  }
}
```

Action:

```json
{
  "type": "tool_call",
  "tool_name": "weather",
  "arguments": {
    "location": "上海",
    "date": "明天"
  }
}
```

Observation:

```json
{
  "type": "tool_result",
  "tool_name": "weather",
  "status": "success",
  "result": {
    "temperature": "18-24C",
    "rain_probability": "20%",
    "wind": "light"
  }
}
```

### 25.3 Step 1

State:

```json
{
  "user_query": "查询明天上海的天气，并告诉我是否适合户外跑步。",
  "history": [
    {
      "action_type": "tool_call",
      "tool_name": "weather",
      "status": "success"
    }
  ],
  "known_facts": [
    "上海明天气温 18-24C",
    "降雨概率 20%",
    "微风"
  ],
  "progress": {
    "open_requirements": [
      "判断是否适合跑步"
    ]
  }
}
```

Action:

```json
{
  "type": "final_answer",
  "content": "明天上海气温 18-24C，降雨概率 20%，风力较小，整体适合户外跑步。建议出门前再确认实时天气，并避开可能降雨的时段。"
}
```

### 25.4 Evaluator Output

```json
{
  "task_success": true,
  "tool_selection_accuracy": 1.0,
  "schema_valid_rate": 1.0,
  "execution_success_rate": 1.0,
  "hallucination": false,
  "failure_types": []
}
```

## 26. 最小可执行版本

为了让这套 formulation 尽快落地，可以先做一个最小可执行版本。

### 26.1 第一阶段只覆盖单工具任务

第一阶段任务约束：

- 每个任务最多需要一次工具调用。
- 工具调用成功后，下一步应该 final answer。
- 暂不处理多工具依赖。
- 暂不处理复杂错误恢复。
- 暂不处理开放式 LLM judge。

这样可以先验证最核心链路：

```text
user_query -> tool_call -> observation -> final_answer
```

### 26.2 第一阶段数据格式

每条任务保存一条完整 trajectory：

```json
{
  "task_id": "weather_001",
  "task_type": "single_tool",
  "model_input_states": [
    {}
  ],
  "expert_actions": [
    {}
  ],
  "observations": [
    {}
  ],
  "labels": {
    "success": true,
    "failure_types": []
  }
}
```

再从 trajectory 拆出 step-level SFT 样本：

```json
{
  "sample_id": "weather_001_step_0",
  "input": {},
  "target": {
    "type": "tool_call",
    "tool_name": "weather",
    "arguments": {
      "location": "上海",
      "date": "明天"
    }
  },
  "metadata": {
    "task_type": "single_tool",
    "step": 0
  }
}
```

### 26.3 第一阶段 evaluator

第一阶段 evaluator 只实现确定性规则：

- tool name 是否 exact match。
- arguments 是否 exact match。
- tool call 是否 schema valid。
- 工具是否 execution success。
- final answer 是否在 tool result 之后出现。

暂时不自动判断：

- argument semantic match。
- final answer 是否表达得足够好。
- 是否存在细粒度幻觉。

### 26.4 第一阶段产出

第一阶段完成后，应该能得到：

- 一套 task specs。
- 一套 expert trajectories。
- 一套 step-level SFT samples。
- 一个 deterministic evaluator。
- 一份 failure report，统计 wrong tool、missing argument、invalid schema 等错误。

如果这一步跑通，再扩展到多工具、错误恢复、反问用户和语义评测。

### 26.5 第一阶段数量目标

第一阶段建议目标：

| 项目 | 建议数量 |
|---|---:|
| task specs | 50-100 |
| expert trajectories | 50-100 |
| SFT step samples | 100-200 |
| held-out eval tasks | 20-30 |
| failure cases | 20+ |

### 26.6 第一阶段验收阈值

第一阶段完成后，至少应该能报告：

| Metric | 建议阈值 |
|---|---:|
| schema valid rate | >= 0.98 |
| tool selection accuracy | >= 0.85 |
| argument exact match | >= 0.75 |
| execution success rate | >= 0.80 |
| grounded final answer rate | >= 0.80 |
| task success rate | >= 0.70 |

这些阈值不是最终目标，而是判断 pipeline 是否跑通的 baseline。

如果达不到阈值，需要根据 failure report 决定补哪类数据，而不是盲目扩大数据量。

## 27. 建模覆盖矩阵与优秀标准

为了判断本文档是否真正支撑 Agent 训练建模，需要把“写得好不好”转化为可检查标准。

下表用于持续评估本文档的完成度。每完善一项，都应该更新对应行的状态、依据和下一步动作。

| 模块 | 需要回答的核心问题 | 当前回答情况 | 达到优秀的标准 | 当前依据 | 下一步动作 |
|---|---|---|---|---|---|
| Task 定义 | 一个任务如何被唯一、稳定、可复现地描述？ | 达到优秀 | 有统一 task schema，明确必填/可选字段，能表达任务约束、成功条件、工具范围和任务类型 | 已补 `Task Schema`，覆盖 `task_id/task_type/user_query/available_tools/success_criteria/constraints/metadata` | 后续随真实任务补充 success criteria 标签集合 |
| Tool 定义 | 工具如何进入 action space？schema 如何约束模型输出？ | 达到优秀 | 有统一 tool schema，包含名称、描述、JSON schema、返回格式、错误类型和执行约束 | 已补 `Tool Schema`，包含 `input_schema/output_schema/error_schema/side_effects` | 后续随具体工具补充真实 output schema |
| Runtime State | 模型推理时真实看见什么？ | 达到优秀 | 明确 message 格式、tool calls、tool results、可用工具 schema，并避免泄漏标注信息 | 已补 runtime message 模板，并明确 `messages/tools` 是模型输入 | 后续按具体训练框架适配原生 tool calling 格式 |
| Annotated State | 哪些信息只用于标注、分析和评测？ | 达到优秀 | 明确哪些字段不能喂给模型，哪些字段只供 evaluator 使用 | 已补字段使用边界表，区分 SFT input、target、evaluator、analysis | 后续在导出脚本中强制校验 no leakage |
| Action Space | 模型可以输出哪些动作？每类动作结构是什么？ | 达到优秀 | 所有 action type 都有 schema、合法性条件、适用场景和终止条件 | 已补 Action Schema、非法样例和终止条件 | 后续随框架适配 action 序列化方式 |
| Observation Space | 环境会返回哪些观察？错误如何表示？ | 达到优秀 | 正常结果、空结果、schema 错误、执行错误、系统错误都有统一结构 | 已补 Observation Schema 和标准错误码表 | 后续按真实工具扩展错误码 |
| Transition Rules | `s_t, a_t, o_{t+1}` 如何生成 `s_{t+1}`？ | 达到优秀 | 每种 action/observation 组合都有状态转移规则，能指导状态机实现 | 已补 `Transition Rules`，覆盖 tool success、schema error、empty result、tool error、final answer、ask user、invalid action、max steps | 后续可直接迁移到 `agent_state_machine.md` |
| Trajectory Schema | 一条完整轨迹如何保存？ | 达到优秀 | 有 canonical trajectory schema，能同时支持成功轨迹、失败轨迹、部分轨迹和多步轨迹 | 已补 `Trajectory Schema`，包含 `trajectory_id/source/steps/terminal_state/labels` | 后续可按工程需要补 `timestamps` |
| SFT Sample Schema | 如何从 trajectory 切成训练样本？ | 达到优秀 | 明确 input/target/metadata，说明失败轨迹如何用于训练或过滤 | 已补 `SFT Sample Schema`，并明确 `metadata` 默认不进入模型输入 | 后续补导出脚本时再校验字段 |
| Feedback / Reward | feedback 如何变成训练信号？ | 达到优秀 | 明确 step feedback、trajectory feedback、reward 组成和适用训练方法 | 已补 reward table、reward 组合公式和不同训练阶段使用方式 | 第一阶段建议只使用 SFT + deterministic evaluator |
| Evaluator Function | evaluator 输入输出是什么？ | 达到优秀 | `E_step` 和 `E_traj` 有完整输入、输出、评分字段、失败类型和聚合方式 | 已补 `Eval Result Schema`、step score、trajectory score 和 evaluation report | 后续可按实验偏好调整权重 |
| Evaluator Rules | 每个 failure type 如何触发？ | 达到优秀 | 有 decision table：条件、触发标签、严重程度、是否可规则判断 | 已补 `Failure Decision Table`，覆盖 step-level 和 trajectory-level failure rules | 语义类规则在 groundedness 章节继续细化 |
| Failure Taxonomy | 失败类型是否稳定、互斥或可组合？ | 达到优秀 | 每个 failure type 有定义、触发条件、例子和优先级 | 已补 primary failure type、多标签策略和优先级 | 后续随真实错误扩展标签 |
| Data Source | 训练数据从哪里来？如何保证质量？ | 达到优秀 | 明确人工标注、日志、模型生成、模拟环境的进入标准和清洗规则 | 已补数据验收标准、清洗流程和数据质量报告 | 后续用真实数据填充分布 |
| Task Taxonomy | 任务类型是否覆盖 Agent 核心能力？ | 达到优秀 | 覆盖无工具、单工具、多工具、反问、错误恢复、并行、顺序依赖，并有样例比例 | 已补推荐任务占比、每类任务最小样例和覆盖标准 | 后续按项目目标调整比例 |
| Minimal Experiment | 第一阶段如何开工？ | 达到优秀 | 明确任务范围、数据格式、evaluator、产出物和验收指标 | 已补数量目标和验收阈值 | 后续根据实际 baseline 调整阈值 |
| Grounded Final Answer | 如何判断最终回答是否基于 observation？ | 达到优秀 | 有 claim extraction 或 reference-based 判断方法，并能标注 hallucination | 已补 `Grounded Final Answer`，包含 claim 分类、unsupported claim、contradiction、derived judgment、claim checking 和 LLM judge 边界 | 后续按任务域补具体 derivation rules |
| Multi-tool Dependency | 多工具顺序和依赖如何建模？ | 达到优秀 | 能表达工具依赖图、等价顺序、参数从 observation 派生 | 已补 dependency graph、argument binding、顺序类型、多工具 evaluator 和失败类型 | 后续在 `agent_state_machine.md` 中展开状态图 |
| Recovery Behavior | 工具失败后如何重试、修正或停止？ | 达到优秀 | 有错误恢复状态、重试策略、最大尝试次数和 evaluator 规则 | 已补 recovery state、策略表、最大尝试次数、recovery evaluator 和失败类型 | 后续在状态机文档中展开恢复路径 |

当前总体判断：

- 本文档已经达到“优秀实现规格”的第一版标准。
- 依据是：核心对象、canonical schemas、transition rules、failure decision table、groundedness、多工具依赖、错误恢复、数据质量、任务覆盖和最小实验验收都已经有可执行定义。
- 读者现在可以据此开始写数据构造脚本、SFT 样本导出脚本、deterministic evaluator baseline 和第一版状态机。
- 后续工作不再是补建模骨架，而是根据具体工具、真实任务和实验结果调整 schema、阈值、任务比例和 evaluator 权重。

## 28. 后续需要按项目确认的问题

这份文档已经给出了 Agent 训练问题的第一版优秀实现规格。剩下的问题不再是建模缺口，而是需要结合具体项目目标和工具环境确认的实验选择：

- 第一阶段具体选择哪些工具和任务域。
- 每个任务类型实际采集多少条 expert trajectory。
- reference trajectory 由人工标注、自动生成，还是二者结合。
- argument semantic match 在具体领域中使用规则、embedding，还是 LLM judge。
- groundedness 的领域推导规则如何细化。
- evaluator 权重和验收阈值是否需要根据 baseline 结果调整。
- 真实工具的 output schema 和 error code 是否需要扩展。
