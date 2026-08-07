# ToolBench 术语表

本文档用于集中记录 ToolBench 相关术语、概念边界、构建方式和实现示例。术语按独立条目组织，便于持续补充、交叉引用和全文检索。

## 文档结构

- [术语索引](#术语索引) - 所有术语的快速查找表
- [核心概念](#核心概念) - Tool、Trajectory、Action、Observation
- [算法与方法](#算法与方法) - DFSDT、ToolEval
- [附录](#附录) - 新增术语模板

## 使用指南

1. 使用目录或术语索引快速定位需要的概念
2. 每个术语条目包含定义、构成要素、示例和使用方式
3. 相关术语通过链接互相引用
4. 新增术语时使用附录中的模板并更新索引

## 术语索引

| 术语 | 分类 | 一句话说明 |
| --- | --- | --- |
| [Tool（Agent Tool）](#toolagent-tool) | Agent 能力与运行时 | 暴露给 Agent、由模型选择并由 Runtime 执行的能力接口 |
| [轨迹（Trajectory）](#轨迹trajectory) | Agent 运行、观测与评估 | Agent 为完成一次任务而产生的、按时间排序的状态、决策、动作与反馈序列 |
| [动作（Action）](#动作action) | Agent 决策与执行 | Agent 在当前状态下选择并提交给 Runtime 或环境执行的一项操作 |
| [观察（Observation）](#观察observation) | Agent 感知与状态更新 | Runtime 或环境在处理 Action 后返回给 Agent 的可感知反馈 |
| [深度优先搜索决策树（DFSDT）](#深度优先搜索决策树dfsdt) | ToolBench 轨迹生成与推理 | 通过深度优先扩展、失败回溯和分支探索，为工具调用任务搜索有效解决路径的方法 |
| [ToolEval](#tooleval) | ToolBench 自动评估 | 使用模型评估任务完成情况和成对轨迹偏好，并汇总 Pass Rate 与 Win Rate 的评估框架 |

---

## 阅读与维护约定

- 每个术语使用二级标题：`## 中文名（English Name）`。
- 每个条目优先包含：概览、定义、构建要素、示例、使用或暴露方式、生命周期、参考资料。
- 不适用的章节可以省略，但"一句话定义"和"相关术语"应保留。
- 新增术语时，先在术语索引中增加一行，再在文末追加完整条目。
- 名称、字段和代码标识使用英文原名；解释性内容使用中文。

---

## 核心概念

### Tool（Agent Tool）

| 属性 | 内容 |
| --- | --- |
| 英文名 | Tool / Agent Tool |
| 中文名 | 工具 / Agent 工具 |
| 分类 | Agent 能力与运行时 |
| 相关术语 | [Agent](#)、[Function](#)、[API](#)、[Tool Call](#)、[MCP](#)、[Runtime](#) |
| 一句话定义 | **暴露给 Agent、由模型根据任务选择、再由 Runtime 执行的一项外部能力** |

#### 定义

Tool 是 Agent 可选择调用的一项外部能力。它通过结构化契约告诉模型”能够做什么、何时使用、需要哪些参数”，并通过运行时执行器完成真正的数据查询或操作。

Tool 不存在于模型权重中，也不等同于普通函数或 API。它通常由两部分组成：

1. **模型可见的 Tool 契约**：名称、用途描述、输入参数 Schema 等。
2. **运行时可执行的 Tool 实现**：本地函数、远程 API、数据库操作、MCP Server、平台内置能力或另一个 Agent。

可以概括为：

```text
Tool = 模型可见的调用契约 + Runtime 可执行的实现 + 必要的运行治理
```

相关概念的区别如下：

| 概念 | 本质 | 谁决定调用 | 典型执行位置 |
| --- | --- | --- | --- |
| 普通函数 | 一段确定的程序逻辑 | 应用代码 | 当前进程 |
| API | 跨进程或跨系统的服务接口 | 调用方代码 | 远程服务或独立进程 |
| Tool | 暴露给模型的能力接口 | 模型根据任务选择，Runtime 执行 | 本地、远程或平台托管环境 |
| Agent | 模型、指令、Tools、状态和执行循环的组合 | Agent 规划并推进任务 | Agent Runtime |

Tool 的底层实现可以是普通函数，也可以由普通函数继续调用外部 API。因此，这些概念属于不同层次，而不是互斥关系：

```text
Agent
  -> 选择并生成 Tool Call
Tool 契约
  -> Runtime 查找执行器
本地函数
  -> 可选：调用外部 API、数据库或其他服务
Tool Result
  -> 返回 Agent 继续推理
```

#### 构建要素

##### 核心要素

| 要素 | 面向对象 | 必需性 | 作用 | 示例或建议 |
| --- | --- | --- | --- | --- |
| `type` | 模型 | 接口通常要求 | 标识工具类型 | `function`、`web_search`、`mcp` |
| `name` | 模型与 Runtime | 必需 | Tool 的稳定唯一标识，也是执行器分发键 | 使用动词开头，如 `get_weather` |
| `description` | 模型 | 强烈建议 | 说明能力、适用条件和禁用条件，帮助模型正确选用 Tool | 写清”何时用、何时不用、是否有副作用” |
| `parameters` / `input_schema` | 模型与 Runtime | Function Tool 必需 | 用 JSON Schema 定义输入字段、类型和约束 | 定义 `properties`、`required`、`enum` 等 |
| `strict` | 模型 | 可选但建议 | 要求模型严格遵循输入 Schema | OpenAI Function Tool 可设置为 `true` |
| Handler / Executor | Runtime | 本地 Tool 必需 | 执行真正的业务逻辑 | Python 函数、HTTP Client、数据库操作等 |
| Tool Registry | Runtime | 多 Tool 场景必需 | 将 Tool 名称映射到对应执行器 | `{“get_weather”: get_weather}` |

##### 生产环境要素

| 要素 | 作用 | 建议 |
| --- | --- | --- |
| 输出契约 | 让 Agent 稳定理解执行结果 | 返回结构化对象，并保持字段和单位一致 |
| 错误契约 | 区分参数错误、业务错误和系统错误 | 使用稳定错误码，并提供可安全展示的信息 |
| 身份与上下文 | 确认工具代表哪个用户或租户执行 | 认证信息由 Runtime 注入，不要让模型生成密钥 |
| 权限与审批 | 防止模型直接执行高风险动作 | 付款、删除、发布等操作应设置授权或人工确认 |
| 超时与重试 | 避免执行无限等待或无节制重试 | 设置明确超时、重试次数和可重试错误范围 |
| 幂等性 | 防止重试造成重复写入 | 写操作使用幂等键或业务唯一键 |
| 可观测性 | 支持调试、审计和评估 | 记录调用名称、参数摘要、结果、耗时和错误 |
| 数据最小化 | 限制敏感信息进入模型上下文 | Tool Result 只返回完成任务所需的信息 |

#### 构建示例

下面以”查询城市天气”为例。一个完整的本地 Function Tool 包含 Tool 契约、执行函数和执行器注册表。

##### 1. 定义模型可见的 Tool 契约

```python
weather_tool = {
    “type”: “function”,
    “name”: “get_weather”,
    “description”: (
        “查询指定城市的实时天气。”
        “仅在用户询问当前天气时使用；不要用于历史天气统计。”
    ),
    “parameters”: {
        “type”: “object”,
        “properties”: {
            “city”: {
                “type”: “string”,
                “description”: “城市名称，例如上海”
            },
            “unit”: {
                “type”: “string”,
                “enum”: [“celsius”, “fahrenheit”],
                “description”: “温度单位”
            }
        },
        “required”: [“city”, “unit”],
        “additionalProperties”: False
    },
    “strict”: True
}
```

##### 2. 实现运行时执行器

```python
def get_weather(city: str, unit: str) -> dict:
    “””真正执行天气查询；内部也可以调用第三方天气 API。”””
    return {
        “ok”: True,
        “city”: city,
        “temperature”: 25,
        “unit”: unit,
        “condition”: “sunny”
    }
```

##### 3. 注册执行器

```python
TOOL_HANDLERS = {
    “get_weather”: get_weather
}
```

##### 4. 分发 Tool Call

```python
def execute_tool_call(tool_name: str, arguments: dict) -> dict:
    handler = TOOL_HANDLERS.get(tool_name)

    if handler is None:
        return {
            “ok”: False,
            “error”: {
                “code”: “TOOL_NOT_FOUND”,
                “message”: f”Unknown tool: {tool_name}”
            }
        }

    try:
        return handler(**arguments)
    except TypeError as exc:
        return {
            “ok”: False,
            “error”: {
                “code”: “INVALID_ARGUMENTS”,
                “message”: str(exc)
            }
        }
    except Exception:
        return {
            “ok”: False,
            “error”: {
                “code”: “TOOL_EXECUTION_FAILED”,
                “message”: “Tool execution failed”
            }
        }
```

这里的职责分工是：

- `weather_tool` 供模型理解和选择。
- `get_weather` 完成实际业务逻辑。
- `TOOL_HANDLERS` 供 Runtime 根据 Tool 名称定位执行器。
- `execute_tool_call` 负责参数传递、执行和错误标准化。

#### 暴露与注册方式

“注册 Tool”的本质，是将 Tool 放入 Agent 当前可用的工具集合。只有定义普通函数但未暴露给 Agent，模型无法发现或调用它。

| 暴露方式 | Tool 定义位于哪里 | 执行器位于哪里 | 是否需要自行分发 |
| --- | --- | --- | --- |
| Agents SDK | `Agent(tools=[...])` | SDK 包装的本地函数或集成 | 通常由 SDK 负责 |
| Responses API | 请求的 `tools` 数组 | 应用服务器 | Function Tool 需要 |
| MCP | MCP Server 的工具清单 | MCP Server | 由 MCP Client/Server 协议处理 |
| 平台内置 Tool | 请求的 `tools` 数组，仅声明类型和配置 | 平台托管环境 | 通常不需要 |
| Agent as Tool | 父 Agent 的 `tools` 集合 | 子 Agent Runtime | 通常由 Agent SDK 负责 |

##### 方式一：通过 Agents SDK 暴露

```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    “””查询指定城市的实时天气。”””
    return f”{city}：晴，25°C”

agent = Agent(
    name=”weather_agent”,
    instructions=”帮助用户查询实时天气。”,
    tools=[get_weather]
)

result = Runner.run_sync(agent, “上海今天天气怎么样？”)
```

其中：

- `@function_tool` 将普通函数包装为模型可调用的 Tool。
- `tools=[get_weather]` 将 Tool 注册到当前 Agent。
- SDK 负责 Tool Call 循环和执行器调用。

##### 方式二：通过 Responses API 暴露

```python
response = client.responses.create(
    model=”<model>”,
    input=”上海今天天气怎么样？”,
    tools=[weather_tool]
)
```

这种方式将 Tool 契约随请求提供给模型。应用需要读取模型返回的 Tool Call，根据名称调用 `TOOL_HANDLERS` 中的执行器，再把 Tool Result 返回给模型继续生成答案。

##### 方式三：通过 MCP 暴露

```text
Agent Runtime
  -> 连接 MCP Server
  -> 发现 MCP Server 提供的 Tool
  -> 将 Tool 契约暴露给模型
  -> 把模型的 Tool Call 发送给 MCP Server 执行
```

MCP 模式下，Tool 的定义与实现注册在 MCP Server 中。Agent Runtime 作为 MCP Client 发现并调用这些工具，适合跨应用共享 Tool 或连接外部系统。

##### 方式四：使用平台内置 Tool

```python
response = client.responses.create(
    model=”<model>”,
    input=”搜索今天的重要科技新闻”,
    tools=[{“type”: “web_search”}]
)
```

平台内置 Tool 只需要在请求中声明类型和相关配置，执行器由平台提供，应用通常不需要维护本地 Handler。

#### 调用生命周期

一个典型的 Tool Calling 生命周期如下：

1. 应用将 Tool 契约暴露给 Agent。
2. Agent 根据用户请求决定是否使用 Tool。
3. 模型生成包含 Tool 名称和参数的 Tool Call。
4. Runtime 校验参数、权限和审批状态。
5. Runtime 查找 Handler，或将调用路由到托管工具/MCP Server。
6. 执行器返回 Tool Result。
7. Runtime 将 Tool Result 交还模型。
8. 模型生成最终答案，或者继续发起新的 Tool Call。

核心原则：

```text
函数存在 != Agent 可以调用
函数被包装为 Tool，并暴露到当前 Agent 的工具集合中 = Agent 可以选择调用
```

#### 参考资料

- [OpenAI：Function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenAI：Using tools](https://developers.openai.com/api/docs/guides/tools)
- [OpenAI：Agents SDK](https://developers.openai.com/api/docs/guides/agents)

---

## 轨迹（Trajectory）

| 属性 | 内容 |
| --- | --- |
| 英文名 | Trajectory |
| 中文名 | 轨迹 / 执行轨迹 |
| 分类 | Agent 运行、观测与评估 |
| 相关术语 | Agent、State、Step、Action、Observation、Tool Call、Tool Result、Trace、Episode、Rollout |
| 一句话定义 | Agent 为完成一次任务而产生的、按时间排序的状态、决策、动作与环境反馈序列 |

### 定义

Trajectory 描述 Agent 从接收任务到结束执行所经历的完整过程。它不仅记录最终答案，还保留中间发生了什么，例如模型看到的上下文、做出的决策、发起的 Tool Call、Tool Result、状态变化、错误与重试。

可以抽象为：

```text
τ = (s₀, a₀, o₁, s₁, a₁, o₂, ..., sₙ)
```

其中：

- `τ` 表示一条 Trajectory。
- `sᵢ`（State）表示第 `i` 步时 Agent 可用的状态或上下文。
- `aᵢ`（Action）表示 Agent 在该状态下选择的动作，例如生成文本、调用 Tool 或结束任务。
- `oᵢ₊₁`（Observation）表示动作执行后从环境得到的反馈，例如 Tool Result、错误或用户回复。

在工程系统中，Trajectory 通常表现为一个按时间排序的 Step 列表，而不一定显式存储数学表达式中的每个状态。某个时刻的 State 可以由初始输入和此前所有 Step 重建，也可以通过快照直接保存。

Trajectory 的边界由一次任务执行或评估样本决定。同一个会话可以包含多条 Trajectory；一次 Trajectory 也可能跨越多个模型调用和 Tool Call。

#### 与相近概念的区别

| 概念 | 关注点 | 与 Trajectory 的关系 |
| --- | --- | --- |
| Conversation History | 用户与 Agent 的消息内容 | 可能是 Trajectory 的输入和组成部分，但通常不包含完整的工具执行、状态变化与运行元数据 |
| Log | 单个组件产生的运行记录 | 多个 Log 可以用于重建 Trajectory，但普通 Log 不一定具有统一步骤结构或任务边界 |
| Trace | 跨组件调用的可观测性结构，通常由 Span 组成 | Trace 偏向系统调用链与性能诊断；Trajectory 偏向 Agent 的决策、动作及其效果，两者可以互相引用 |
| Episode | 从环境重置到终止的一次完整交互 | 在强化学习或仿真环境中常与 Trajectory 近义；Episode 更强调环境定义的回合边界 |
| Rollout | 使用某个策略实际采样得到的一次执行 | Rollout 强调“采样过程或样本”；其产物通常是一条 Trajectory |
| Final Answer | Agent 最终输出 | 只是 Trajectory 的终止结果，不能单独说明任务是如何完成的 |

### 构成要素

#### 顶层字段

| 字段 | 必需性 | 作用 | 示例或建议 |
| --- | --- | --- | --- |
| `trajectory_id` | 必需 | 唯一标识一次执行 | 使用 UUID 或平台生成的稳定 ID |
| `task` / `input` | 必需 | 记录任务目标和初始输入 | 保存结构化输入；敏感字段应脱敏 |
| `steps` | 必需 | 按时间顺序保存决策、动作与反馈 | 每个 Step 使用稳定的 `step_id` 和 `type` |
| `status` | 必需 | 表示执行结果 | `completed`、`failed`、`cancelled`、`max_steps_reached` |
| `output` | 建议 | 保存最终答案或结构化结果 | 与终止 Step 保持一致 |
| `started_at` / `ended_at` | 建议 | 确定时间范围并计算总耗时 | 使用带时区的 ISO 8601 时间 |
| `agent` / `model` | 建议 | 标识产生轨迹的 Agent、策略或模型版本 | 记录稳定版本号，便于回归比较 |
| `usage` / `cost` | 建议 | 汇总 Token、Tool 和费用消耗 | 同时保留各 Step 明细，避免只有总数 |
| `trace_id` | 可选 | 关联底层可观测性 Trace | 不要默认将 `trace_id` 与 `trajectory_id` 视为同一标识 |
| `labels` / `metadata` | 可选 | 保存数据集、实验和运行环境信息 | 仅保存检索与分析所需字段 |
| `evaluation` | 可选 | 保存成功率、得分和失败原因 | 区分规则评分、模型评分和人工评分 |

#### Step 字段

| 字段 | 必需性 | 作用 | 示例或建议 |
| --- | --- | --- | --- |
| `step_id` | 必需 | 唯一标识一个步骤 | 在单条 Trajectory 内稳定且可排序 |
| `type` | 必需 | 区分步骤类型 | `model`、`tool_call`、`tool_result`、`user_input`、`handoff`、`error` |
| `timestamp` | 建议 | 记录步骤发生时间 | 使用带时区的 ISO 8601 时间 |
| `parent_step_id` | 可选 | 表示并行、嵌套或因果关系 | 仅靠数组顺序不足以表示并发结构时使用 |
| `input` | 按类型必需 | 保存该步骤收到的输入 | Tool Step 中保存经过校验的参数 |
| `output` | 按类型必需 | 保存该步骤产生的输出 | 大对象可存引用、摘要和内容哈希 |
| `state_delta` | 可选 | 记录该步骤引起的状态变化 | 比重复保存完整 State 更节省空间 |
| `duration_ms` | 建议 | 支持延迟分析 | 采用统一单位 |
| `error` | 失败时必需 | 记录稳定错误码和安全错误信息 | 避免写入密钥、完整堆栈或敏感数据 |

### 示例

下面是一条经过简化的天气查询 Trajectory。它展示模型决策、Tool Call、Tool Result 和最终答案之间的顺序关系：

```json
{
  "trajectory_id": "traj_01JXYZ",
  "task": {
    "user_input": "上海今天天气怎么样？"
  },
  "agent": {
    "name": "weather_agent",
    "version": "1.2.0"
  },
  "steps": [
    {
      "step_id": "step_1",
      "type": "model",
      "output": {
        "decision": "call_tool",
        "tool_name": "get_weather"
      }
    },
    {
      "step_id": "step_2",
      "type": "tool_call",
      "input": {
        "name": "get_weather",
        "arguments": {
          "city": "上海",
          "unit": "celsius"
        }
      }
    },
    {
      "step_id": "step_3",
      "type": "tool_result",
      "output": {
        "ok": true,
        "temperature": 25,
        "unit": "celsius",
        "condition": "sunny"
      }
    },
    {
      "step_id": "step_4",
      "type": "model",
      "output": {
        "decision": "finish",
        "text": "上海今天晴，气温约 25°C。"
      }
    }
  ],
  "status": "completed",
  "output": "上海今天晴，气温约 25°C。"
}
```

为了保护安全与隐私，生产环境中的 Trajectory 不应直接保存密钥、认证头、完整思维链或不必要的个人信息。模型内部推理可以用简短的决策摘要、结构化动作和可验证结果代替。

### 使用方式

Trajectory 常用于以下场景：

| 场景 | 使用方式 |
| --- | --- |
| 调试 | 定位 Agent 在哪一步选择了错误 Tool、生成了错误参数或误解了 Tool Result |
| 评估 | 对完整执行过程计算任务成功率、步骤数、Tool 正确率、延迟和成本 |
| 回归测试 | 比较不同 Prompt、模型、Tool 版本产生的 Trajectory，发现行为退化 |
| 训练与优化 | 将高质量 Trajectory 用作示例、偏好数据或策略优化数据 |
| 审计 | 还原高风险动作的发起者、输入、审批、执行结果和时间 |
| 数据分析 | 聚合失败模式、重复调用、无效步骤和高成本路径 |

对 Trajectory 进行评估时，不应只判断最终答案。一个答案即使正确，也可能经过危险、低效或不可复现的路径；一个失败答案也可能暴露 Tool 描述、参数校验或运行环境中的具体缺陷。常见评估维度包括：

- **结果正确性**：最终输出是否完成任务。
- **过程正确性**：每一步决策、Tool 选择和参数是否合理。
- **效率**：是否存在冗余步骤、重复 Tool Call 或不必要的 Token 消耗。
- **安全与合规**：是否正确执行授权、审批、数据最小化和敏感信息处理。
- **鲁棒性**：面对 Tool 错误、超时或缺失信息时，是否能够合理恢复或停止。

### 生命周期

一条典型 Trajectory 的生命周期如下：

1. Runtime 创建 `trajectory_id`，记录任务输入、Agent 版本和运行配置。
2. Agent 读取当前 State，并生成下一步 Action。
3. Runtime 执行 Action；如果是 Tool Call，则完成参数校验、授权和工具调用。
4. Runtime 将 Tool Result、错误或其他环境反馈记录为 Observation。
5. 系统更新 State，并追加新的 Step；必要时重复步骤 2 至 4。
6. Agent 产生最终答案、明确失败，或触发取消、超时、最大步数等终止条件。
7. Runtime 写入终止状态、最终输出、资源用量和关联的 `trace_id`。
8. 评估器或人工审核者为 Trajectory 添加评分、标签和失败原因。
9. 数据进入调试、回归分析或训练流程，并按照数据保留策略归档或删除。

核心原则：

```text
只保存最终答案，无法解释 Agent 如何到达结果。
保存结构化 Trajectory，才能对 Agent 的结果、过程、效率和安全性进行联合评估。
```

---

## 动作（Action）

| 属性 | 内容 |
| --- | --- |
| 英文名 | Action |
| 中文名 | 动作 |
| 分类 | Agent 决策与执行 |
| 相关术语 | Agent、Policy、State、Decision、Step、Tool Call、Observation、Trajectory |
| 一句话定义 | Agent 在当前状态下选择并提交给 Runtime 或环境执行的一项操作 |

### 定义

Action 是 Agent 面对当前 State 时选择的下一项可执行操作。它把模型的判断转化为能够被 Runtime 或外部环境解释和处理的结构化指令，例如调用 Tool、向用户回复、请求补充信息、把任务移交给另一个 Agent，或者结束当前任务。

在最简化的 Agent 模型中，可以表示为：

```text
aₜ ~ π(· | sₜ)
sₜ --aₜ--> environment --oₜ₊₁--> sₜ₊₁
```

其中：

- `sₜ`（State）表示第 `t` 步时 Agent 可用的状态。
- `π`（Policy）表示 Agent 根据 State 选择 Action 的策略。
- `aₜ`（Action）表示被选中的动作。
- `oₜ₊₁`（Observation）表示环境执行或处理 Action 后返回的反馈。

Action 强调“Agent 要环境做什么”，而不是模型内部如何得出该选择。生产系统通常只记录简短的决策摘要、Action 类型、参数和执行结果，不应把模型的完整内部推理当作 Action 保存。

#### Action 的常见类型

| 类型 | 含义 | 示例 |
| --- | --- | --- |
| `tool_call` | 请求 Runtime 调用一个 Tool | 调用 `get_weather(city="上海")` |
| `respond` | 向用户或调用方输出内容 | 返回问题答案或进度说明 |
| `request_input` | 请求用户或外部系统补充必要信息 | 询问缺少的订单号 |
| `handoff` | 将任务或控制权移交给其他 Agent | 转交给退款处理 Agent |
| `delegate` | 创建或指派一个可独立执行的子任务 | 并行分析多个数据源 |
| `wait` | 等待异步操作、事件或外部状态变化 | 等待长时间任务完成 |
| `finish` | 明确结束当前执行 | 返回成功、失败或取消状态 |

具体系统可以定义不同的 Action 集合。一个 Action 只有在当前环境的 Action Space 中合法、参数满足契约且通过权限校验时，才是可执行动作。

#### 与相近概念的区别

| 概念 | 本质 | 与 Action 的关系 |
| --- | --- | --- |
| Decision | Agent 对下一步行为作出的选择 | Decision 是选择过程或结果；Action 是提交给 Runtime 或环境执行的具体表示 |
| Tool Call | 调用 Tool 的结构化请求 | Tool Call 是一种 Action，但 Action 还可以是回复、等待、移交或结束任务 |
| Step | Trajectory 中的一个记录单元 | 一个 Step 可以记录 Action，也可以记录 Observation、状态更新或错误 |
| Observation | 环境对 Action 的反馈 | Action 是 Agent 的输出，Observation 是执行后返回给 Agent 的输入 |
| Command | 面向某个执行器的具体命令 | Command 常是 Action 经 Runtime 校验、解析或编译后的底层执行形式 |
| Event | 系统中已经发生的事实 | Action 表达意图；Action 被接受、拒绝或执行后可以产生一个或多个 Event |

### 构成要素

| 字段 | 必需性 | 作用 | 示例或建议 |
| --- | --- | --- | --- |
| `action_id` | 必需 | 唯一标识该动作 | 使用稳定 ID，支持重试、审计和结果关联 |
| `type` | 必需 | 标识动作类型并决定处理方式 | `tool_call`、`respond`、`request_input`、`handoff`、`wait`、`finish` |
| `name` | 按类型必需 | 指定 Tool、目标 Agent 或具体操作 | Tool Call 中使用已注册的 Tool 名称 |
| `arguments` / `payload` | 按类型必需 | 提供执行动作所需的结构化参数 | 使用 Schema 校验，避免自由文本拼接命令 |
| `target` | 可选 | 指定动作接收方 | 用户、Tool、Agent、队列或外部服务 |
| `reason_summary` | 可选 | 简要说明选择该动作的可观察理由 | 使用简短、可审计摘要，不记录完整内部推理 |
| `idempotency_key` | 写操作建议 | 防止超时或重试造成重复副作用 | 付款、创建、发送、发布等操作应提供 |
| `timeout_ms` | 可选 | 限制动作的最长执行时间 | 按 Tool 或动作类型设置合理上限 |
| `approval` | 高风险操作必需 | 表示授权或人工审批要求 | 区分 `required`、`approved`、`rejected` |
| `metadata` | 可选 | 关联 Trajectory、Trace 和实验信息 | 保存 `trajectory_id`、`step_id`、`trace_id` 等引用 |

Action 的输入契约应尽量满足以下要求：

- **类型明确**：Runtime 能根据 `type` 选择唯一的校验器和执行路径。
- **参数结构化**：使用 JSON Schema 或等价契约描述字段、类型、枚举和必需性。
- **边界清晰**：定义前置条件、权限、超时、重试规则与可能的副作用。
- **结果可关联**：通过 `action_id` 将 Action 与后续 Observation、错误和审计记录关联。
- **内容最小化**：仅携带执行所需数据，避免传播密钥或无关个人信息。

### 示例

下面是一个 Tool Call 类型的 Action：

```json
{
  "action_id": "act_01JXYZ",
  "type": "tool_call",
  "name": "get_weather",
  "arguments": {
    "city": "上海",
    "unit": "celsius"
  },
  "timeout_ms": 5000,
  "metadata": {
    "trajectory_id": "traj_01JXYZ",
    "step_id": "step_2"
  }
}
```

Runtime 对该 Action 完成校验和执行后，可以产生如下 Observation：

```json
{
  "observation_id": "obs_01JXYZ",
  "action_id": "act_01JXYZ",
  "type": "tool_result",
  "status": "succeeded",
  "output": {
    "temperature": 25,
    "unit": "celsius",
    "condition": "sunny"
  }
}
```

两者通过 `action_id` 关联。Action 记录 Agent 提交的操作意图，Observation 记录环境实际返回的结果；不能因为 Action 已生成，就假定其已经成功执行。

### 执行与治理

Runtime 接收到 Action 后，通常需要依次完成以下处理：

| 阶段 | 处理内容 | 失败时的典型结果 |
| --- | --- | --- |
| 解析 | 识别 `type`、目标和参数 | 返回 `INVALID_ACTION` |
| Schema 校验 | 检查字段类型、必需参数和枚举 | 返回 `INVALID_ARGUMENTS` |
| 能力检查 | 确认 Tool 或目标 Agent 当前可用 | 返回 `ACTION_NOT_AVAILABLE` |
| 权限与审批 | 校验身份、权限和人工确认状态 | 返回 `PERMISSION_DENIED` 或进入等待审批状态 |
| 策略检查 | 检查安全、合规和业务限制 | 拒绝执行并记录策略原因 |
| 执行 | 将 Action 路由给对应 Handler 或环境 | 返回执行结果、超时或系统错误 |
| 结果标准化 | 将底层响应转换为 Observation | 返回统一状态、输出和错误结构 |
| 记录 | 写入 Trajectory、Trace、用量和审计信息 | 记录失败不应掩盖动作本身的执行状态 |

高风险 Action 应特别考虑：

- 执行前展示目标、关键参数和潜在影响。
- 把“准备动作”和“执行动作”拆开，确保审批针对最终参数。
- 使用幂等键避免重试引起重复付款、发送或创建。
- 明确区分可重试错误与永久错误，并限制自动重试次数。
- 对删除、覆盖、发布等操作优先提供预览、软删除或补偿机制。

### 生命周期

一个典型 Action 的生命周期如下：

1. Agent 根据当前 State 和可用 Action Space 选择下一项操作。
2. Agent 生成带类型和参数的结构化 Action。
3. Runtime 解析 Action，并校验 Schema、能力、权限和审批状态。
4. Runtime 接受、拒绝或暂缓该 Action。
5. 对于被接受的 Action，Runtime 将其路由到对应执行器或环境。
6. 执行器完成操作，或返回业务错误、系统错误、取消或超时。
7. Runtime 将结果标准化为 Observation，并通过 `action_id` 建立关联。
8. Action 与 Observation 被写入 Trajectory，Agent 据此更新 State 并决定下一步。

核心原则：

```text
Action 表达“计划执行什么”，Observation 表达“实际发生了什么”。
生成 Action != Action 已获批准 != Action 已执行成功。
```

---

### 观察（Observation）

| 属性 | 内容 |
| --- | --- |
| 英文名 | Observation |
| 中文名 | 观察 / 观测 |
| 常见拼写错误 | Obeservation ❌ → Observation ✅ |
| 分类 | Agent 感知与状态更新 |
| 相关术语 | [Agent](#)、[Environment](#)、[Action](#动作action)、[Tool Result](#)、[Event](#)、[State](#)、[Step](#)、[Trajectory](#轨迹trajectory) |
| 一句话定义 | **Runtime 或环境在处理 Action 后返回给 Agent、用于更新状态和决定下一步的可感知反馈** |

### 定义

Observation 是 Agent 能够从 Runtime、Tool、用户或外部环境中接收到的反馈。它说明某个 Action 被如何处理、环境发生了什么变化，或者 Agent 下一步可以依据哪些新信息继续行动。

在典型的 Agent 交互循环中：

```text
State(sₜ)
  -> Agent 选择 Action(aₜ)
  -> Runtime / Environment 处理动作
  -> 返回 Observation(oₜ₊₁)
  -> 更新为 State(sₜ₊₁)
```

可以简化表示为：

```text
sₜ₊₁ = update(sₜ, aₜ, oₜ₊₁)
```

Observation 是 Agent “可见的环境反馈”，不一定等于环境的完整真实状态。Runtime 通常会对底层结果进行选择、裁剪、脱敏、标准化或汇总，再将适合进入 Agent 上下文的信息作为 Observation 返回。

例如，数据库可能返回数千行原始记录，但 Agent 实际收到的 Observation 只包含匹配数量、前若干条结果和分页标记。因此应区分：

```text
环境真实状态 != 底层原始响应 != Agent 可见的 Observation
```

#### Observation 的常见来源与类型

| 类型 | 来源 | 表达的反馈 | 示例 |
| --- | --- | --- | --- |
| `tool_result` | Tool / Runtime | Tool Call 的成功结果 | 天气数据、搜索结果、文件内容 |
| `tool_error` | Tool / Runtime | Tool 调用失败或被拒绝 | 参数错误、权限不足、超时 |
| `user_input` | 用户 | 新指令、澄清或审批结果 | 补充订单号、确认执行删除 |
| `environment_event` | 外部环境 | 与任务相关的状态变化 | 作业完成、文件生成、付款状态更新 |
| `agent_message` | 其他 Agent | 委派任务的进度或结果 | 子 Agent 返回分析结论 |
| `system_feedback` | Runtime / 策略系统 | 执行限制或治理结果 | 达到最大步数、内容被策略拒绝 |
| `time_signal` | 时钟 / 调度器 | 等待结束或截止时间到达 | 重试窗口开启、定时任务触发 |

#### 与相近概念的区别

| 概念 | 本质 | 与 Observation 的关系 |
| --- | --- | --- |
| Action | Agent 提交给环境执行的操作 | Action 是 Agent 的输出；Observation 是环境返回给 Agent 的输入 |
| Tool Result | Tool 执行产生的结果 | Tool Result 是 Observation 的常见来源；Runtime 标准化、裁剪后才形成 Agent 可见的 Observation |
| Event | 已经发生的事实或状态变化通知 | Event 可以触发 Observation；并非所有系统 Event 都需要暴露给 Agent |
| State | Agent 在某一步可用于决策的信息集合 | Observation 是新增反馈，State 是结合历史、记忆和 Observation 后形成的决策上下文 |
| Step | Trajectory 中的记录单元 | Observation 可以单独占一个 Step，也可以与对应 Action 记录在同一 Step 中 |
| Message | Agent 系统中的通信载体 | Observation 可以用 Message 表示，但 Message 也可能承载指令、Action 或普通回复 |
| Log | 供运维或开发者查看的运行记录 | Log 不一定对 Agent 可见；只有进入 Agent 决策上下文的反馈才属于 Observation |

### 构成要素

| 字段 | 必需性 | 作用 | 示例或建议 |
| --- | --- | --- | --- |
| `observation_id` | 必需 | 唯一标识一条 Observation | 使用稳定 ID，便于追踪和去重 |
| `type` | 必需 | 标识反馈类型并决定解析方式 | `tool_result`、`tool_error`、`user_input`、`environment_event` |
| `status` | 建议 | 表示对应动作或事件的处理状态 | `succeeded`、`failed`、`rejected`、`timed_out`、`pending` |
| `action_id` | 由 Action 触发时必需 | 关联产生该反馈的 Action | 一个 Action 可以产生零个、一个或多个 Observation |
| `source` | 建议 | 标识反馈来源 | Tool 名称、用户 ID、Agent 名称或系统组件 |
| `output` / `payload` | 按类型必需 | 保存 Agent 可使用的结构化反馈 | 使用稳定 Schema，避免把数据嵌入不可解析文本 |
| `error` | 失败时必需 | 保存标准化错误信息 | 包含稳定错误码、可安全展示的信息及是否可重试 |
| `timestamp` | 建议 | 记录反馈发生或接收时间 | 使用带时区的 ISO 8601 时间 |
| `sequence` | 并发或流式场景建议 | 保持同一来源内的顺序 | 不要仅依赖到达时间推断因果关系 |
| `is_partial` | 流式场景可选 | 表示结果是否尚未结束 | 最终片段应明确标记完成状态 |
| `content_ref` | 大结果场景可选 | 引用未直接放入上下文的完整内容 | 同时保存摘要、内容类型、大小和完整性哈希 |
| `metadata` | 可选 | 关联 Trajectory、Trace、版本和用量 | 避免放入 Agent 不需要的内部信息 |

Observation 契约应尽量满足以下要求：

- **来源明确**：Agent 和审计系统能够判断反馈来自哪个组件或参与者。
- **结果可关联**：由 Action 触发时，通过 `action_id` 保持因果关系。
- **结构稳定**：相同类型的成功与错误响应采用一致字段和单位。
- **事实优先**：描述实际结果，不把计划、猜测或未执行的意图写成已发生事实。
- **数据最小化**：只暴露下一步决策所需的信息，并移除密钥和无关敏感数据。
- **保真可追溯**：如果进行了裁剪或汇总，应明确标记，并在必要时保留原始结果引用。

### 示例

下面是一条与 Tool Call Action 对应的成功 Observation：

```json
{
  "observation_id": "obs_01JXYZ",
  "type": "tool_result",
  "status": "succeeded",
  "action_id": "act_01JXYZ",
  "source": {
    "type": "tool",
    "name": "get_weather"
  },
  "output": {
    "city": "上海",
    "temperature": 25,
    "unit": "celsius",
    "condition": "sunny"
  },
  "timestamp": "2026-08-06T10:30:00+08:00",
  "metadata": {
    "trajectory_id": "traj_01JXYZ",
    "step_id": "step_3"
  }
}
```

失败 Observation 应提供稳定错误码和恢复信息，而不是伪造正常结果：

```json
{
  "observation_id": "obs_01JXYA",
  "type": "tool_error",
  "status": "timed_out",
  "action_id": "act_01JXYZ",
  "source": {
    "type": "tool",
    "name": "get_weather"
  },
  "error": {
    "code": "TOOL_TIMEOUT",
    "message": "天气服务未在 5 秒内返回结果",
    "retryable": true
  }
}
```

Agent 收到第二条 Observation 后，可以选择有限次数重试、换用其他 Tool、向用户说明暂时不可用，或安全结束任务。它不应把超时解释成“天气数据不存在”。

### 处理与治理

Runtime 把底层结果转换为 Observation 时，通常需要完成以下处理：

| 阶段 | 处理内容 | 关键要求 |
| --- | --- | --- |
| 接收 | 获取 Tool、用户、Agent 或环境返回的原始结果 | 保留来源、接收时间和关联标识 |
| 验证 | 检查格式、Schema、完整性和来源可信度 | 不把未验证的外部内容视为系统指令 |
| 标准化 | 统一状态、错误码、字段名称和单位 | 成功、失败、拒绝、取消和超时应可区分 |
| 安全处理 | 移除密钥、认证信息和不必要的个人数据 | 外部内容可能包含提示注入或恶意载荷 |
| 裁剪与汇总 | 控制结果大小，使其适合模型上下文 | 标记截断、分页、摘要或原始内容引用 |
| 关联 | 连接 Action、Step、Trajectory 和 Trace | 使用 ID 表示因果关系，不只依赖时间顺序 |
| 投递 | 将 Observation 加入 Agent 的下一轮输入 | 清楚区分可信系统数据与不可信外部内容 |
| 记录 | 保存评估、调试和审计所需信息 | Agent 可见内容与受限原始记录可以分层保存 |

来自网页、邮件、文档、用户或第三方 Tool 的 Observation 都可能包含不可信指令。Runtime 应把它们作为数据处理，不应允许其覆盖系统指令、绕过权限或自行触发新的高风险 Action。

### 生命周期

一条典型 Observation 的生命周期如下：

1. Action 被执行、用户提供新输入，或环境产生与任务相关的 Event。
2. Runtime 接收底层结果，并记录来源、时间和关联的 `action_id`。
3. Runtime 验证结果格式、来源可信度和完整性。
4. Runtime 对结果进行标准化、脱敏、裁剪或摘要。
5. 系统创建带有稳定类型和状态的 Observation。
6. Observation 被写入 Trajectory，并投递给 Agent。
7. Agent 将 Observation 与已有历史、记忆和任务目标结合，更新当前 State。
8. Agent 根据新 State 选择下一项 Action，或结束任务。
9. Observation 按照审计与数据保留策略归档、限制访问或删除。

核心原则：

```text
Action 表达操作意图，Observation 记录环境反馈。
Tool Result 经过验证和治理后，才适合作为 Agent 可见的 Observation。
Observation 是 Agent 对环境的有限视图，不等于环境的完整真实状态。
```

---

## 深度优先搜索决策树（DFSDT）

| 属性 | 内容 |
| --- | --- |
| 英文名 | Depth-First Search based Decision Tree |
| 缩写 | DFSDT |
| 中文名 | 深度优先搜索决策树 |
| 分类 | ToolBench 轨迹生成与推理 |
| 相关术语 | ToolBench、ToolLLM、DFS、Decision Tree、Trajectory、Action、Observation、Tool Call、ReAct、Backtracking |
| 一句话定义 | 通过深度优先扩展、失败回溯和多分支探索，为工具调用任务搜索有效解决路径并生成训练轨迹的方法 |

### 定义

DFSDT 是 ToolBench 在答案标注和工具调用推理中使用的搜索方法，全称为 **Depth-First Search based Decision Tree**。它把一次任务的候选解决过程组织成树：节点记录模型生成的 Action、Action Input、Observation 或终止结果；从根节点到叶节点的一条路径构成一条候选 Trajectory。

DFSDT 不把模型第一次生成的线性推理当作唯一方案。当当前路径遇到无效 Tool、错误参数、失败 Observation、主动放弃或其他终止条件时，搜索可以回溯到先前节点，让模型在已知旧候选和反馈的基础上生成不同分支，再继续向深处探索。

可以抽象为：

```text
任务输入
  -> 生成候选 Action
  -> 执行 Tool 并获得 Observation
  -> 沿当前分支继续向深处扩展
  -> 成功：保存根到成功叶节点的 Trajectory
  -> 失败：剪枝并回溯，探索不同 Action 或 Action Input
```

这里的“Decision Tree”是由 LLM 动态生成的工具调用搜索树，不是传统机器学习中用于分类或回归、通过特征阈值训练得到的决策树模型。

#### 搜索树中的典型节点

| 节点或信息 | 含义 | 在路径中的作用 |
| --- | --- | --- |
| Root | 当前任务及初始上下文 | 所有候选解决路径的起点 |
| Thought / Reasoning Summary | 对下一步的简短规划或理由 | 为 Action 选择提供可观察上下文 |
| Action | 选择的 Tool 或终止操作 | 决定下一步执行方向 |
| Action Input | 传给 Tool 的参数 | 与 Action 一起形成 Tool Call |
| Observation | Tool 或环境返回的结果 | 更新状态，并影响后续分支选择 |
| Give Answer | 成功终止节点 | 表示找到候选有效答案 |
| Give Up / Pruned | 放弃或剪枝节点 | 表示当前分支不再继续扩展 |

#### 与相近方法和概念的区别

| 概念 | 核心方式 | 与 DFSDT 的区别 |
| --- | --- | --- |
| Chain-of-Thought（CoT） | 沿单条推理链连续生成 | 通常不显式维护搜索树，也缺少系统化分支探索和回溯 |
| ReAct | 交替生成 Reasoning、Action 和 Observation | 通常沿一条在线轨迹推进；DFSDT 在此类交互步骤之上增加树搜索与失败回溯 |
| 普通 DFS | 按深度优先顺序遍历既定图或树 | DFSDT 的候选节点由 LLM 动态生成，执行 Tool 后才获得新的环境反馈 |
| Beam Search | 同时保留并比较多个高分候选 | ToolBench 实现中的 DFSDT 按深度优先方式递归扩展，不依赖每层额外的 LLM 候选排序 |
| 分类/回归决策树 | 从训练数据学习特征划分规则 | DFSDT 的树是一次任务中的搜索过程，不是预测模型 |
| Trajectory | 一次具体执行路径 | DFSDT 是搜索方法；搜索树中的一条根到叶路径是一条候选 Trajectory |

### 构成要素

| 要素 | ToolBench 实现中的对应内容 | 作用 |
| --- | --- | --- |
| 任务与 Tool 集合 | System/User Prompt、Function 定义、`io_func` | 定义根节点状态和可用 Action Space |
| 搜索树 | `my_tree`、`tree_node` | 保存节点、父子关系、Observation 和终止状态 |
| 节点生成器 | LLM | 根据当前路径和 Prompt 生成下一步 Action 或答案 |
| 环境执行器 | `io_func.step(...)` | 执行 Tool Call 并返回 Observation 与状态码 |
| 深度优先控制器 | `DFS_tree_search.DFS(...)` | 优先扩展当前分支，在终止或失败后回溯 |
| 多样性提示 | `DIVERSITY_PROMPT` | 告知模型已有候选，促使其生成不同分支 |
| 剪枝与终止规则 | 深度、状态码、终止节点、放弃节点 | 限制无效路径并决定何时停止扩展 |
| 搜索预算 | 最大深度、分支数、最大查询次数、目标答案数 | 控制成本、时延和搜索范围 |
| 输出转换 | 搜索树、候选链、训练消息、最终答案 | 生成可分析或可用于训练的轨迹数据 |

#### 关键搜索参数

| 参数 | ToolBench 实现中的名称 | 含义 |
| --- | --- | --- |
| 最大路径深度 | `single_chain_max_step` | 单条候选路径允许扩展的最大深度 |
| 每节点候选数 | `tree_beam_size` | 一个节点在一轮中生成的子分支数量 |
| 查询预算 | `max_query_count` | 搜索允许使用的最大 LLM 查询次数 |
| 目标答案数 | `answer` | 找到指定数量的成功终止节点后停止 |
| 候选过滤 | `with_filter` | 官方实现中 `False` 对应 DFSDT；`True` 会额外比较并排序候选 |

ToolBench 官方命令和数据处理中常使用 `DFS_woFilter_w2` 表示 DFSDT 配置，其中 `woFilter` 对应不执行同层候选的额外 LLM 排序；具体参数语义应以当前代码版本为准。

### 示例

假设任务是：“查询上海明天的天气，并推荐是否适合户外跑步。”一次简化的 DFSDT 搜索可以表示为：

```text
Root: 查询天气并给出跑步建议
├─ Action: get_historical_weather          [错误分支]
│  └─ Observation: 不支持未来日期
│     └─ Pruned -> 回溯到 Root
├─ Action: get_weather_forecast            [继续深入]
│  └─ Observation: 上海明天 27°C，雷阵雨
│     ├─ Action: search_running_guideline   [冗余/失败分支]
│     │  └─ Observation: 服务超时
│     │     └─ Pruned -> 回溯到天气结果节点
│     └─ Action: give_answer                [成功分支]
│        └─ Answer: 不建议户外跑步，可选择室内训练
└─ Action: give_up                          [其他候选分支]
```

最终用于标注或回答的路径可以是：

```text
Root
  -> get_weather_forecast
  -> Observation(27°C, 雷阵雨)
  -> give_answer
```

被剪枝的错误分支仍可保留在完整搜索树中，用于调试、分析失败模式或构造偏好数据；训练数据是否包含这些分支，取决于具体预处理策略。

### 搜索流程

一轮典型 DFSDT 搜索如下：

1. 使用任务描述、用户输入和 Tool 定义创建根节点。
2. LLM 根据从根到当前节点的消息与 Observation 生成下一步 Action。
3. 如果 Action 是 Tool Call，Runtime 执行 Tool 并把结果记录为 Observation 节点。
4. 如果当前节点仍可扩展，算法继续沿该分支递归向下搜索。
5. 如果同一节点已存在候选分支，使用多样性提示要求模型避免重复旧候选。
6. 如果达到成功答案、主动放弃、错误剪枝、最大深度或查询预算，则停止当前分支。
7. 搜索根据回溯长度返回祖先节点，继续探索尚未尝试的候选分支。
8. 找到目标数量的成功答案，或耗尽搜索预算后结束。
9. 输出搜索树、有效候选路径、最终答案、训练消息及查询/Token 统计。

简化伪代码如下：

```python
def dfsdt(node, budget):
    if is_terminal(node) or budget.exhausted():
        return evaluate_terminal(node)

    for _ in range(budget.branch_width):
        action = llm_generate_action(
            path=node.path,
            previous_candidates=node.children
        )
        observation = execute_or_finish(action)
        child = append_child(node, action, observation)

        result = dfsdt(child, budget.consume())
        if result.should_stop_search:
            return result

    return backtrack()
```

这段伪代码只表达核心思想，不等同于 ToolBench 的完整实现。官方实现还处理节点类型、状态码、回溯长度、回调、候选答案、消息转换和用量统计。

### 优势与局限

| 方面 | 说明 |
| --- | --- |
| 多路径探索 | 可以在第一条方案失败后尝试其他 Tool、参数或调用顺序 |
| 错误利用 | 将 Tool 错误和环境反馈用于后续分支生成，而不是立即结束整个任务 |
| 轨迹标注 | 能为复杂单工具和多工具任务搜索可用于训练的解决路径 |
| 可解释与可调试 | 完整树展示成功路径、失败分支、回溯位置和 Tool Observation |
| 成本 | 多分支会增加 LLM 查询、Tool 调用、Token、延迟和存储开销 |
| 非完备性 | 受搜索预算、模型候选质量和 Tool 可用性限制，未找到答案不等于答案不存在 |
| 副作用风险 | 若搜索分支直接执行写操作，回溯不能自动撤销已经发生的外部副作用 |
| 数据与安全 | Tool Observation 可能包含敏感数据或提示注入，需要在进入后续 Prompt 前治理 |

在生产环境使用 DFSDT 时，应优先在只读、仿真或可回滚环境中探索。对于付款、发送、删除、发布等有副作用的 Action，需要使用审批、幂等键、事务、补偿机制或“规划与执行分离”，不能把搜索树回溯误认为现实环境也已回滚。

### 生命周期

一次 DFSDT 任务的生命周期如下：

1. 准备任务、候选 Tool、Prompt、搜索预算和终止规则。
2. 创建搜索树根节点，并初始化 Tool 环境状态。
3. 从根节点开始生成 Action，执行 Tool 并记录 Observation。
4. 沿当前分支进行深度优先扩展。
5. 在失败、剪枝或终止节点处回溯，并使用已有候选促进新分支生成。
6. 找到有效答案，或因深度、查询次数等预算耗尽而结束。
7. 从成功叶节点回溯到根节点，提取候选解决 Trajectory。
8. 保存搜索树、答案、训练消息和资源用量。
9. 对结果执行有效性过滤、ToolEval 或人工评估，再进入训练或测试数据集。

核心原则：

```text
DFSDT 是搜索方法，搜索树是过程数据，根到叶路径是候选 Trajectory。
回溯改变的是搜索控制流，不会自动撤销外部 Tool 已产生的现实副作用。
```

### 参考资料

- [ToolLLM：Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789)
- [OpenBMB：ToolBench 官方仓库](https://github.com/OpenBMB/ToolBench)
- [ToolBench：DFSDT 官方实现](https://github.com/OpenBMB/ToolBench/blob/master/toolbench/inference/Algorithms/DFS.py)

---

## ToolEval

| 属性 | 内容 |
| --- | --- |
| 英文名 | ToolEval |
| 中文名 | 工具调用自动评估框架 |
| 分类 | ToolBench 自动评估 |
| 相关术语 | ToolBench、ToolLLM、Trajectory、Action Sequence、Pass Rate、Preference、Win Rate、LLM-as-a-Judge、Human Evaluation |
| 一句话定义 | 使用模型评估任务完成情况和成对工具调用轨迹偏好，并汇总 Pass Rate 与 Win Rate 的 ToolBench 自动评估框架 |

### 定义

ToolEval 是 ToolBench 为降低人工评估成本而设计的机器自动评估框架。它不只比较最终答案文本，还结合任务指令、可用 Tool、候选答案的中间 Action Sequence 和最终步骤，判断 Agent 是否完成任务，以及两个候选方案中哪一个更好。

ToolEval 包含两个核心评估方向：

```text
单答案评估：Task + Tools + Candidate Trajectory
  -> 是否成功完成任务
  -> 汇总为 Pass Rate

成对偏好评估：Task + Tools + Reference Trajectory + Candidate Trajectory
  -> 哪个答案/动作序列更好
  -> 汇总为 Win Rate
```

ToolEval 属于 **LLM-as-a-Judge** 评估：评估器模型依据预定义准则和 Prompt 对开放式工具调用结果作判断。它不是 Tool 本身，也不是负责执行 Agent 的 Runtime；它消费已经生成的预测或 Trajectory，并产出评估标签和聚合指标。

### 核心指标

#### Pass Rate

Pass Rate（通过率）衡量模型在规定的调用或运行预算内成功完成测试指令的比例：

```text
Pass Rate = 成功完成的任务数 / 参与评估的任务总数
```

单个样本的判断目标是“任务是否被解决”，而不是输出是否与某个参考答案逐字一致。对于存在多个有效 Tool 调用顺序或多个合理答案的任务，这种判断比 Exact Match 更适合，但也会受到评估 Prompt 和评估器模型能力影响。

#### Preference 与 Win Rate

Preference（偏好）是对同一任务的两条候选答案或 Action Sequence 进行成对比较，判断哪一个更符合预定义的优质答案标准。ToolEval 可以对同一答案对执行多次采样，再聚合偏好结果以提高稳定性。

当候选模型与固定参考模型比较时，Win Rate（胜率）可以概括为：

```text
Win Rate = 候选方案被评估器判为优于参考方案的样本数 / 有效比较样本总数
```

具体实现可能对双方成功状态、平局、无效样本和多次采样结果进行额外处理，因此复现实验时应使用同一 ToolEval 版本、配置和汇总脚本，不能只根据上述简式自行计算后直接比较榜单结果。

#### 指标关系

| 指标 | 评估对象 | 是否需要参考模型 | 回答的问题 |
| --- | --- | --- | --- |
| Pass Rate | 单个候选答案/轨迹 | 否 | 这个任务是否成功完成 |
| Preference | 同一任务的两个候选答案/轨迹 | 是 | 两个方案中哪一个更好 |
| Win Rate | 一组 Preference 结果 | 是 | 候选模型相对参考模型有多大比例更优 |

高 Pass Rate 不必然对应高 Win Rate：两个模型都可能完成任务，但其中一个模型的调用路径更正确、简洁或完整。Win Rate 也不是模型的绝对分数；更换参考模型、评估器、Prompt 或数据集后，数值不可直接横向比较。

### 评估输入与输出

#### 样本输入

| 字段 | 作用 | 说明 |
| --- | --- | --- |
| `query` / instruction | 定义需要完成的任务 | 必须与被测模型实际接收的任务一致 |
| `available_tools` | 给出任务允许使用的 Tool | 用于判断 Tool 选择和调用是否合法 |
| `answer_details` / intermediate steps | 保存中间 Action、Action Input 和 Observation | 支持过程级判断，不应只保留最终答案 |
| `final_answer` / final step | 保存最终结果或结束状态 | 用于判断任务是否完整回答 |
| `method` | 标识推理方法 | 例如 CoT、ReAct、DFSDT；预处理时可能移除该字段以减少偏差 |
| `query_id` | 关联测试样本 | 用于对齐参考模型、候选模型和测试子集 |

#### 评估输出

| 输出 | 粒度 | 作用 |
| --- | --- | --- |
| Pass/Fail 标签 | 单样本 | 表示候选方案是否解决任务 |
| Preference Label | 答案对 | 表示参考方案、候选方案或平局等比较结果 |
| 评估原始响应 | 单次 Judge 调用 | 支持解析检查、重试和审计 |
| Pass Rate | 模型 × 测试集 | 汇总单样本通过标签 |
| Win Rate | 候选模型 × 参考模型 × 测试集 | 汇总成对偏好标签 |

ToolBench 官方转换格式将任务、可用 Tool 和答案对象组织在一起。不同推理方法的原始输出需要先转换成统一 Answer Format，才能交给相同评估流程比较。

### 示例

假设测试任务是：“查询上海天气并给出是否适合跑步的建议。”两个候选结果如下：

```text
Candidate A
  -> 调用 get_weather(city="上海")
  -> Observation: 27°C，雷阵雨
  -> 回答：不建议户外跑步，可改为室内训练

Candidate B
  -> 调用 get_historical_weather(city="上海")
  -> Observation: 返回上个月平均天气
  -> 回答：天气温暖，适合跑步
```

ToolEval 可以执行两类判断：

```json
{
  "pass_rate_evaluation": {
    "candidate_a": "pass",
    "candidate_b": "fail"
  },
  "preference_evaluation": {
    "preferred": "candidate_a"
  }
}
```

该 JSON 仅用于解释概念，不是 ToolEval 官方脚本的原始输出 Schema。实际格式应以所使用版本的评估器配置和结果文件为准。

### 评估流程

一次典型的 ToolEval 评估流程如下：

1. 选择 ToolBench 测试子集，并固定测试样本 ID。
2. 使用被测模型和推理方法生成每个任务的答案与中间步骤。
3. 将不同方法的原始预测转换为 ToolEval 统一 Answer Format。
4. 配置评估器模型、Prompt、采样次数、线程数和输出目录。
5. 对每条候选答案执行任务完成判断，保存 Pass/Fail 结果。
6. 按测试子集聚合单样本标签，计算 Pass Rate。
7. 选择固定参考模型，把参考答案与候选答案按 `query_id` 对齐。
8. 对答案对执行 Preference 评估；必要时多次采样并聚合结果。
9. 计算候选模型相对参考模型的 Win Rate。
10. 保存逐样本结果、聚合指标和配置，以支持复现、误差分析和人工抽查。

官方实现的偏好评估接口接收 `query`、`available_tools` 和候选 `answers`，并支持 `multisample` 与 `sample_n`。实现还会随机调整候选答案顺序，以降低固定展示位置带来的偏差。

### 与相近评估方式的区别

| 评估方式 | 主要依据 | 与 ToolEval 的区别 |
| --- | --- | --- |
| Exact Match | 候选文本是否与标准答案完全一致 | 简单确定，但难以覆盖开放式答案和多条有效工具路径 |
| Function Call Match | Tool 名称和参数是否匹配参考调用 | 适合明确单步任务，但可能错误惩罚不同却有效的调用方案 |
| Execution-based Evaluation | 在环境中执行并检查最终状态 | 结果更直接，但需要稳定、可重置且可验证的环境 |
| Human Evaluation | 人工阅读并判断答案或轨迹 | 通常更灵活，但成本高、速度慢，也需要一致的标注规范 |
| ToolEval | 模型根据任务、Tool 和轨迹进行成功及偏好判断 | 可规模化处理开放式工具任务，但存在 Judge 偏差与非确定性 |
| ToolBench Leaderboard | 展示模型与方法的汇总结果 | Leaderboard 使用评估结果；ToolEval 是产生部分指标的评估框架 |

### 可靠性与局限

ToolBench 官方报告中，ToolEval 的 ChatGPT 评估器与人工标注在 Pass Rate 上达到 **87.1%** 一致率，在 Win Rate 上达到 **80.3%** 一致率。这是特定样本、模型、Prompt 和评估版本下的验证结果，不代表所有新数据集或新 Tool 场景都具有相同一致性。

| 风险 | 表现 | 建议 |
| --- | --- | --- |
| Judge 偏差 | 偏好某种写作风格、长度、模型家族或推理模式 | 隐藏模型身份，随机交换答案顺序，并抽样人工复核 |
| 非确定性 | 同一答案对多次评估结果不同 | 多次采样、记录原始标签并使用稳定聚合规则 |
| Prompt 敏感 | 修改评估准则或模板导致分数变化 | 对 Prompt 和配置做版本控制，不混合不同版本结果 |
| 位置偏差 | 总是偏好先展示或后展示的答案 | 随机化候选顺序，并正确映射回原候选 |
| 输入截断 | 长 Trajectory 的关键步骤未进入 Judge 上下文 | 记录截断策略，保留关键 Action、Observation 和最终结果 |
| 环境不稳定 | API 失效或结果随时间变化影响任务完成 | 固定环境快照、缓存结果或单独标记基础设施失败 |
| 参考依赖 | Win Rate 随参考模型变化 | 报告参考模型名称、版本和推理方法 |
| 指标误读 | 把 Win Rate 当成绝对能力或把 Judge 标签当作事实 | 同时报告 Pass Rate、成本、延迟及人工抽查结果 |

### 复现要求

为了让 ToolEval 结果可比较，应至少保存：

- ToolBench 数据版本、测试子集和 `query_id` 列表。
- 被测模型名称、版本、Prompt、Tool 定义与推理方法。
- 原始预测、转换后的 Answer Format 和转换脚本版本。
- ToolEval 代码版本、Evaluator 名称、Judge 模型及配置文件。
- 评估 Prompt、采样次数、温度、并发和重试规则。
- Pass Rate 的单样本标签及 Preference 的逐次采样标签。
- Win Rate 使用的参考模型及其推理方法。
- 失败、超时、无效输出、平局和缺失样本的处理规则。

核心原则：

```text
Pass Rate 回答“能否完成”，Win Rate 回答“相对谁更好”。
ToolEval 的自动标签是评估器判断，不等同于无误差的客观真值。
没有固定数据、参考模型、Judge 和配置的分数，不具备严格可比性。
```

### 参考资料

- [ToolLLM：Facilitating Large Language Models to Master 16000+ Real-world APIs](https://arxiv.org/abs/2307.16789)
- [OpenBMB：ToolBench 官方仓库](https://github.com/OpenBMB/ToolBench)
- [OpenBMB：ToolEval 官方说明与实现](https://github.com/OpenBMB/ToolBench/tree/master/toolbench/tooleval)

---

## 附录：新增术语模板

新增术语时，可复制以下结构，并同步更新文首的术语索引：

```markdown
## 中文名（English Name）

| 属性 | 内容 |
| --- | --- |
| 英文名 | 术语英文名 |
| 中文名 | 术语中文名 |
| 分类 | 所属分类 |
| 相关术语 | 术语 A、术语 B |
| 一句话定义 | 用一句话说明术语的本质和用途 |

### 定义

说明术语是什么、不是什么，以及它与相关概念的边界。

### 构建要素

使用表格说明组成部分、必需性、作用和建议。

### 示例

提供一个最小、完整且可以理解的示例。

### 使用方式

说明该概念如何接入系统或参与运行流程。

### 生命周期

按顺序说明主要运行步骤；不适用时可省略。

### 参考资料

- [资料名称](https://example.com)
```
