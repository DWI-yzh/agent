# ToolBench 术语表

本文档用于集中记录 ToolBench 相关术语、概念边界、构建方式和实现示例。术语按独立条目组织，便于持续补充、交叉引用和全文检索。

## 阅读与维护约定

- 每个术语使用二级标题：`## 中文名（English Name）`。
- 每个条目优先包含：概览、定义、构建要素、示例、使用或暴露方式、生命周期、参考资料。
- 不适用的章节可以省略，但“一句话定义”和“相关术语”应保留。
- 新增术语时，先在术语索引中增加一行，再在文末追加完整条目。
- 名称、字段和代码标识使用英文原名；解释性内容使用中文。

## 术语索引

| 术语 | 分类 | 一句话说明 |
| --- | --- | --- |
| [Tool（Agent Tool）](#toolagent-tool) | Agent 能力与运行时 | 暴露给 Agent、由模型选择并由 Runtime 执行的能力接口 |

---

## Tool（Agent Tool）

| 属性 | 内容 |
| --- | --- |
| 英文名 | Tool / Agent Tool |
| 中文名 | 工具 / Agent 工具 |
| 分类 | Agent 能力与运行时 |
| 相关术语 | Agent、Function、API、Tool Call、MCP、Runtime |
| 一句话定义 | 暴露给 Agent、由模型根据任务选择、再由 Runtime 执行的一项外部能力 |

### 定义

Tool 是 Agent 可选择调用的一项外部能力。它通过结构化契约告诉模型“能够做什么、何时使用、需要哪些参数”，并通过运行时执行器完成真正的数据查询或操作。

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

### 构建要素

#### 核心要素

| 要素 | 面向对象 | 必需性 | 作用 | 示例或建议 |
| --- | --- | --- | --- | --- |
| `type` | 模型 | 接口通常要求 | 标识工具类型 | `function`、`web_search`、`mcp` |
| `name` | 模型与 Runtime | 必需 | Tool 的稳定唯一标识，也是执行器分发键 | 使用动词开头，如 `get_weather` |
| `description` | 模型 | 强烈建议 | 说明能力、适用条件和禁用条件，帮助模型正确选用 Tool | 写清“何时用、何时不用、是否有副作用” |
| `parameters` / `input_schema` | 模型与 Runtime | Function Tool 必需 | 用 JSON Schema 定义输入字段、类型和约束 | 定义 `properties`、`required`、`enum` 等 |
| `strict` | 模型 | 可选但建议 | 要求模型严格遵循输入 Schema | OpenAI Function Tool 可设置为 `true` |
| Handler / Executor | Runtime | 本地 Tool 必需 | 执行真正的业务逻辑 | Python 函数、HTTP Client、数据库操作等 |
| Tool Registry | Runtime | 多 Tool 场景必需 | 将 Tool 名称映射到对应执行器 | `{"get_weather": get_weather}` |

#### 生产环境要素

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

### 构建示例

下面以“查询城市天气”为例。一个完整的本地 Function Tool 包含 Tool 契约、执行函数和执行器注册表。

#### 1. 定义模型可见的 Tool 契约

```python
weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": (
        "查询指定城市的实时天气。"
        "仅在用户询问当前天气时使用；不要用于历史天气统计。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，例如上海"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "温度单位"
            }
        },
        "required": ["city", "unit"],
        "additionalProperties": False
    },
    "strict": True
}
```

#### 2. 实现运行时执行器

```python
def get_weather(city: str, unit: str) -> dict:
    """真正执行天气查询；内部也可以调用第三方天气 API。"""
    return {
        "ok": True,
        "city": city,
        "temperature": 25,
        "unit": unit,
        "condition": "sunny"
    }
```

#### 3. 注册执行器

```python
TOOL_HANDLERS = {
    "get_weather": get_weather
}
```

#### 4. 分发 Tool Call

```python
def execute_tool_call(tool_name: str, arguments: dict) -> dict:
    handler = TOOL_HANDLERS.get(tool_name)

    if handler is None:
        return {
            "ok": False,
            "error": {
                "code": "TOOL_NOT_FOUND",
                "message": f"Unknown tool: {tool_name}"
            }
        }

    try:
        return handler(**arguments)
    except TypeError as exc:
        return {
            "ok": False,
            "error": {
                "code": "INVALID_ARGUMENTS",
                "message": str(exc)
            }
        }
    except Exception:
        return {
            "ok": False,
            "error": {
                "code": "TOOL_EXECUTION_FAILED",
                "message": "Tool execution failed"
            }
        }
```

这里的职责分工是：

- `weather_tool` 供模型理解和选择。
- `get_weather` 完成实际业务逻辑。
- `TOOL_HANDLERS` 供 Runtime 根据 Tool 名称定位执行器。
- `execute_tool_call` 负责参数传递、执行和错误标准化。

### 暴露与注册方式

“注册 Tool”的本质，是将 Tool 放入 Agent 当前可用的工具集合。只有定义普通函数但未暴露给 Agent，模型无法发现或调用它。

| 暴露方式 | Tool 定义位于哪里 | 执行器位于哪里 | 是否需要自行分发 |
| --- | --- | --- | --- |
| Agents SDK | `Agent(tools=[...])` | SDK 包装的本地函数或集成 | 通常由 SDK 负责 |
| Responses API | 请求的 `tools` 数组 | 应用服务器 | Function Tool 需要 |
| MCP | MCP Server 的工具清单 | MCP Server | 由 MCP Client/Server 协议处理 |
| 平台内置 Tool | 请求的 `tools` 数组，仅声明类型和配置 | 平台托管环境 | 通常不需要 |
| Agent as Tool | 父 Agent 的 `tools` 集合 | 子 Agent Runtime | 通常由 Agent SDK 负责 |

#### 方式一：通过 Agents SDK 暴露

```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。"""
    return f"{city}：晴，25°C"

agent = Agent(
    name="weather_agent",
    instructions="帮助用户查询实时天气。",
    tools=[get_weather]
)

result = Runner.run_sync(agent, "上海今天天气怎么样？")
```

其中：

- `@function_tool` 将普通函数包装为模型可调用的 Tool。
- `tools=[get_weather]` 将 Tool 注册到当前 Agent。
- SDK 负责 Tool Call 循环和执行器调用。

#### 方式二：通过 Responses API 暴露

```python
response = client.responses.create(
    model="<model>",
    input="上海今天天气怎么样？",
    tools=[weather_tool]
)
```

这种方式将 Tool 契约随请求提供给模型。应用需要读取模型返回的 Tool Call，根据名称调用 `TOOL_HANDLERS` 中的执行器，再把 Tool Result 返回给模型继续生成答案。

#### 方式三：通过 MCP 暴露

```text
Agent Runtime
  -> 连接 MCP Server
  -> 发现 MCP Server 提供的 Tool
  -> 将 Tool 契约暴露给模型
  -> 把模型的 Tool Call 发送给 MCP Server 执行
```

MCP 模式下，Tool 的定义与实现注册在 MCP Server 中。Agent Runtime 作为 MCP Client 发现并调用这些工具，适合跨应用共享 Tool 或连接外部系统。

#### 方式四：使用平台内置 Tool

```python
response = client.responses.create(
    model="<model>",
    input="搜索今天的重要科技新闻",
    tools=[{"type": "web_search"}]
)
```

平台内置 Tool 只需要在请求中声明类型和相关配置，执行器由平台提供，应用通常不需要维护本地 Handler。

### 调用生命周期

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

### 参考资料

- [OpenAI：Function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenAI：Using tools](https://developers.openai.com/api/docs/guides/tools)
- [OpenAI：Agents SDK](https://developers.openai.com/api/docs/guides/agents)

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
