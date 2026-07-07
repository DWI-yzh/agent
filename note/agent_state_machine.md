# Agent 状态机规格

## 1. 目标

本文档定义 tool-using Agent 在一次任务执行过程中的状态机。

`problem_formulation.md` 负责回答“Agent 训练问题如何建模”；本文档负责回答“Agent 在运行时如何一步步流转”。

这份文档要回答：

- Agent 执行任务时有哪些状态？
- 每个状态的输入、输出和责任是什么？
- 什么事件会触发状态转移？
- 工具调用成功、失败、空结果、schema 错误时如何处理？
- Agent 如何继续、重试、反问用户、终止或失败？
- 多工具任务和错误恢复如何进入状态机？
- 状态机如何产出 trajectory，供训练和 evaluator 使用？

## 2. 总览

一个最小 Agent 执行循环可以表示为：

```text
UserQuery
  -> BuildRuntimeState
  -> ModelDecision
  -> ActionValidation
  -> ToolExecution
  -> ObservationHandling
  -> ModelDecision
  -> ...
  -> Terminal
```

如果模型选择最终回答，则路径是：

```text
UserQuery
  -> BuildRuntimeState
  -> ModelDecision
  -> FinalAnswerValidation
  -> Terminal
```

如果模型遇到错误，则进入恢复路径：

```text
ToolExecution
  -> ObservationHandling
  -> RecoveryDecision
  -> ModelDecision
```

状态机的核心思想是：每一步都生成一个结构化 step record，并最终形成一条 trajectory。

## 3. 核心对象

状态机运行时依赖以下对象。

### 3.1 Runtime Context

`runtime_context` 是状态机在一次任务执行中维护的上下文。

```json
{
  "task_id": "weather_001",
  "messages": [],
  "tools": [],
  "history": [],
  "progress": {
    "step": 0,
    "max_steps": 5,
    "finished": false,
    "known_facts": [],
    "open_requirements": []
  },
  "recovery_state": null,
  "terminal_state": null,
  "labels": {
    "success": null,
    "failure_types": []
  }
}
```

### 3.2 Step Record

每轮决策都应该生成一个 step record：

```json
{
  "step_index": 0,
  "state_name": "ModelDecision",
  "model_input_state": {},
  "annotation_state": {},
  "action": {},
  "observation": null,
  "transition": {
    "from": "ModelDecision",
    "to": "ActionValidation",
    "event": "model_emitted_tool_call"
  }
}
```

### 3.3 Trajectory

状态机最终输出 trajectory：

```json
{
  "trajectory_id": "traj_weather_001_model_a",
  "task_id": "weather_001",
  "steps": [],
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

## 4. 状态集合

第一版状态机包含以下状态：

| State | 责任 | 是否调用模型 | 是否调用工具 |
|---|---|---:|---:|
| `Init` | 初始化任务和上下文 | no | no |
| `BuildRuntimeState` | 构造模型输入 | no | no |
| `ModelDecision` | 模型选择下一步 action | yes | no |
| `ParseAction` | 解析模型输出为结构化 action | no | no |
| `ValidateAction` | 校验 action 类型和字段 | no | no |
| `ValidateToolSchema` | 校验 tool call 参数 schema | no | no |
| `ExecuteTool` | 执行工具 | no | yes |
| `HandleObservation` | 处理工具 observation | no | no |
| `RecoveryDecision` | 判断是否进入恢复路径 | no | no |
| `ValidateFinalAnswer` | 检查 final answer 是否可终止 | no | no |
| `AskUser` | 反问用户或等待用户补充 | optional | no |
| `CheckTermination` | 判断是否达到终止条件 | no | no |
| `TerminalSuccess` | 成功终止 | no | no |
| `TerminalFailure` | 失败终止 | no | no |

## 5. 事件集合

状态转移由事件触发。

| Event | 说明 |
|---|---|
| `task_received` | 收到用户任务 |
| `runtime_state_ready` | 模型输入构造完成 |
| `model_emitted_tool_call` | 模型输出 tool call |
| `model_emitted_final_answer` | 模型输出 final answer |
| `model_emitted_ask_user` | 模型输出 ask user |
| `model_emitted_invalid_action` | 模型输出无法解析或不合法 |
| `action_valid` | action 结构合法 |
| `action_invalid` | action 结构非法 |
| `schema_valid` | tool arguments 通过 schema |
| `schema_invalid` | tool arguments 未通过 schema |
| `tool_success` | 工具成功返回 |
| `tool_empty_result` | 工具返回空结果 |
| `tool_error_retryable` | 工具返回可恢复错误 |
| `tool_error_unrecoverable` | 工具返回不可恢复错误 |
| `need_retry` | 需要重试或修正 |
| `need_continue` | 需要继续下一步 |
| `need_user_input` | 需要用户补充信息 |
| `final_answer_ready` | 可以输出最终回答 |
| `max_steps_exceeded` | 超过最大步数 |
| `terminal_success` | 任务成功终止 |
| `terminal_failure` | 任务失败终止 |

## 6. 主路径状态转移

### 6.1 单工具成功路径

```text
Init
  -> BuildRuntimeState
  -> ModelDecision
  -> ParseAction
  -> ValidateAction
  -> ValidateToolSchema
  -> ExecuteTool
  -> HandleObservation
  -> BuildRuntimeState
  -> ModelDecision
  -> ParseAction
  -> ValidateAction
  -> ValidateFinalAnswer
  -> TerminalSuccess
```

示例：

```text
用户：查询明天上海天气，并判断是否适合跑步
模型：调用 weather(location=上海, date=明天)
工具：返回天气
模型：基于天气给出跑步建议
终止：final_answer
```

### 6.2 无工具路径

```text
Init
  -> BuildRuntimeState
  -> ModelDecision
  -> ParseAction
  -> ValidateAction
  -> ValidateFinalAnswer
  -> TerminalSuccess
```

适用场景：

- 用户请求不需要外部信息。
- 模型已有足够上下文。
- 调用工具会成为 unnecessary tool call。

### 6.3 反问用户路径

单轮任务：

```text
ModelDecision
  -> ParseAction
  -> ValidateAction
  -> AskUser
  -> TerminalSuccess or TerminalFailure
```

多轮任务：

```text
AskUser
  -> WaitUserInput
  -> BuildRuntimeState
  -> ModelDecision
```

是否把 `ask_user` 视为成功，取决于 task spec。如果用户信息确实不足，合理反问可以是成功动作；如果信息充足还反问，则是 `unnecessary_ask_user`。

## 7. 状态转移表

| Current State | Event | Next State | 主要动作 |
|---|---|---|---|
| `Init` | `task_received` | `BuildRuntimeState` | 初始化 context |
| `BuildRuntimeState` | `runtime_state_ready` | `ModelDecision` | 构造 messages/tools |
| `ModelDecision` | `model_emitted_tool_call` | `ParseAction` | 解析 tool action |
| `ModelDecision` | `model_emitted_final_answer` | `ParseAction` | 解析 final action |
| `ModelDecision` | `model_emitted_ask_user` | `ParseAction` | 解析 ask user action |
| `ModelDecision` | `model_emitted_invalid_action` | `TerminalFailure` | 标记 `invalid_action` |
| `ParseAction` | `action_valid` | `ValidateAction` | 检查 action 字段 |
| `ParseAction` | `action_invalid` | `TerminalFailure` | 标记 `invalid_action` |
| `ValidateAction` | `model_emitted_tool_call` | `ValidateToolSchema` | 检查 tool 是否存在 |
| `ValidateAction` | `model_emitted_final_answer` | `ValidateFinalAnswer` | 检查是否可终止 |
| `ValidateAction` | `model_emitted_ask_user` | `AskUser` | 检查是否允许反问 |
| `ValidateToolSchema` | `schema_valid` | `ExecuteTool` | 执行工具 |
| `ValidateToolSchema` | `schema_invalid` | `HandleObservation` | 生成 schema error observation |
| `ExecuteTool` | `tool_success` | `HandleObservation` | 记录工具结果 |
| `ExecuteTool` | `tool_empty_result` | `HandleObservation` | 记录空结果 |
| `ExecuteTool` | `tool_error_retryable` | `HandleObservation` | 记录可恢复错误 |
| `ExecuteTool` | `tool_error_unrecoverable` | `HandleObservation` | 记录不可恢复错误 |
| `HandleObservation` | `need_continue` | `CheckTermination` | 更新 progress |
| `HandleObservation` | `need_retry` | `RecoveryDecision` | 更新 recovery_state |
| `RecoveryDecision` | `need_continue` | `BuildRuntimeState` | 让模型决定修正或重试 |
| `RecoveryDecision` | `terminal_failure` | `TerminalFailure` | 不可恢复 |
| `ValidateFinalAnswer` | `terminal_success` | `TerminalSuccess` | 成功终止 |
| `ValidateFinalAnswer` | `terminal_failure` | `TerminalFailure` | 失败终止 |
| `AskUser` | `need_user_input` | `TerminalSuccess` or `BuildRuntimeState` | 单轮终止或多轮等待 |
| `CheckTermination` | `max_steps_exceeded` | `TerminalFailure` | 标记 `max_steps_exceeded` |
| `CheckTermination` | `need_continue` | `BuildRuntimeState` | 继续下一轮 |

## 8. 每个状态的职责

### 8.1 Init

输入：

- task spec
- tool specs
- initial user message

输出：

- initialized runtime context

职责：

- 设置 `task_id`
- 设置 `messages`
- 设置 `tools`
- 初始化 `history/progress/recovery_state/labels`

### 8.2 BuildRuntimeState

输入：

- runtime context

输出：

- model input state

职责：

- 构造模型可见的 `messages`
- 附带可用 `tools`
- 不泄漏 `annotation_state/labels/eval_result`
- 保持和真实推理格式一致

### 8.3 ModelDecision

输入：

- model input state

输出：

- raw model output

职责：

- 由模型选择下一步 action
- action 可以是 `tool_call`、`final_answer`、`ask_user`

此状态是 policy `pi_theta(a_t | s_t)` 的实际运行点。

### 8.4 ParseAction

输入：

- raw model output

输出：

- structured action

职责：

- 将模型输出解析成 canonical action schema
- 如果无法解析，触发 `invalid_action`

### 8.5 ValidateAction

输入：

- structured action
- task config
- tool specs

输出：

- action validation result

职责：

- 检查 action type 是否允许
- 检查必要字段是否存在
- 检查 `tool_name` 是否存在
- 检查 `ask_user` 是否被当前任务允许

### 8.6 ValidateToolSchema

输入：

- tool action
- tool input schema

输出：

- schema validation result
- schema error observation, if invalid

职责：

- 校验 `arguments`
- 生成标准 `schema_error`
- 不执行 schema invalid 的工具调用

### 8.7 ExecuteTool

输入：

- valid tool action

输出：

- observation

职责：

- 调用 executor
- 捕获 tool result、empty result、tool error、system error
- 标准化 observation

### 8.8 HandleObservation

输入：

- action
- observation
- runtime context

输出：

- updated runtime context
- next event

职责：

- 追加 history
- 更新 messages
- 更新 known facts
- 更新 open requirements
- 更新 recovery state
- 标记 step-level failure types

### 8.9 RecoveryDecision

输入：

- recovery state
- error observation
- task config

输出：

- next event

职责：

- 判断错误是否可恢复
- 判断是否超过最大尝试次数
- 判断是否应重试、换工具、反问用户或终止

注意：此状态不直接替模型做业务 action，而是决定是否给模型下一次决策机会。

### 8.10 ValidateFinalAnswer

输入：

- final answer
- observations
- task success criteria

输出：

- terminal success/failure event

职责：

- 检查是否过早回答
- 检查是否缺少必要工具调用
- 检查是否基于 observation
- 检查是否覆盖任务要求

### 8.11 AskUser

输入：

- ask user action
- task config

输出：

- terminal state or wait state

职责：

- 单轮任务中将反问作为终止动作
- 多轮任务中等待用户补充信息
- 判断是否为 unnecessary ask user

### 8.12 TerminalSuccess / TerminalFailure

输入：

- runtime context
- labels

输出：

- final trajectory

职责：

- 设置 `terminal_state`
- 聚合 failure types
- 输出 trajectory
- 交给 evaluator 做最终评测

## 9. Observation 处理路径

| Observation | 状态机动作 | 下一步 |
|---|---|---|
| `tool_result/success` | 更新 known facts，清空相关 error state | `BuildRuntimeState` |
| `schema_error` | 记录缺失字段或类型错误，设置 recovery state | `RecoveryDecision` |
| `empty_result` | 记录空结果，判断是否可放宽参数或换工具 | `RecoveryDecision` |
| `tool_error/retryable` | 记录错误码和 attempt count | `RecoveryDecision` |
| `tool_error/non_retryable` | 判断是否有替代工具 | `RecoveryDecision` or `TerminalFailure` |
| `system_error/retryable` | 允许有限重试 | `RecoveryDecision` |
| `system_error/non_retryable` | 说明限制或失败终止 | `TerminalFailure` |

## 10. 错误恢复路径

### 10.1 Schema Error Recovery

```text
ValidateToolSchema
  -> HandleObservation(schema_error)
  -> RecoveryDecision
  -> BuildRuntimeState
  -> ModelDecision
  -> corrected tool_call
```

合理恢复：

- 补齐缺失参数
- 修正参数类型
- 删除不允许字段

不合理恢复：

- 原样重试
- 编造最终答案
- 换无关工具

### 10.2 Empty Result Recovery

```text
ExecuteTool
  -> HandleObservation(empty_result)
  -> RecoveryDecision
  -> ModelDecision
```

合理恢复：

- 放宽查询条件
- 换等价工具
- 说明没有查到结果
- 必要时反问用户

### 10.3 Retryable Tool Error Recovery

```text
ExecuteTool
  -> HandleObservation(tool_error_retryable)
  -> RecoveryDecision
  -> ModelDecision
```

合理恢复：

- 原参数重试一次
- 调整参数后重试
- 换 fallback tool

### 10.4 Unrecoverable Error

```text
ExecuteTool
  -> HandleObservation(tool_error_unrecoverable)
  -> RecoveryDecision
  -> TerminalFailure
```

如果存在替代工具，也可以：

```text
RecoveryDecision
  -> BuildRuntimeState
  -> ModelDecision
  -> switch_tool
```

## 11. 最大步数与循环控制

状态机必须设置 `max_steps`。

推荐规则：

```text
if progress.step >= max_steps and terminal_state is null:
    terminal_state.reason = "max_steps_exceeded"
    labels.failure_types += ["max_steps_exceeded"]
    goto TerminalFailure
```

循环检测：

| Loop Pattern | 判定 |
|---|---|
| 同一工具 + 同一参数重复调用 2 次以上 | `looping_tool_call` |
| schema error 后原样重试 | `retry_without_fix` |
| empty result 后无参数变化重复调用 | `poor_recovery` |
| 多工具任务中反复调用无依赖工具 | `extra_tool_call` or `looping_tool_call` |

## 12. 多工具状态机

多工具任务在普通状态机上增加 dependency tracking。

### 12.1 Multi-tool Context

```json
{
  "tool_plan_state": {
    "nodes": {
      "find_order": {
        "status": "completed"
      },
      "track_package": {
        "status": "pending"
      }
    },
    "edges": [
      {
        "from": "find_order",
        "to": "track_package",
        "type": "data_dependency"
      }
    ],
    "argument_bindings": []
  }
}
```

### 12.2 多工具转移规则

| 条件 | 状态机行为 |
|---|---|
| 工具节点完成 | 标记 node status 为 `completed` |
| 后续工具依赖已满足 | 允许模型调用后续工具 |
| 后续工具依赖未满足 | 如果模型提前调用，标记 `dependency_violation` |
| 参数来自前序 observation | 校验 argument binding |
| 多个工具无依赖 | 允许任意顺序 |
| 缺少必要工具 | final answer 时标记 `missing_required_tool` |

### 12.3 等价顺序

如果工具调用满足 partial order，就不应判错。

```text
valid_order = all(
  position(parent) < position(child)
  for edge in dependency_edges
)
```

对于无依赖工具：

```text
set(model_required_tools) == set(reference_required_tools)
```

## 13. 终止条件

状态机终止原因包括：

| Reason | Success | 说明 |
|---|---:|---|
| `final_answer` | depends | 模型输出最终回答，由 evaluator 判定是否成功 |
| `ask_user` | depends | 信息不足时合理反问可算成功动作 |
| `max_steps_exceeded` | false | 超过最大步数 |
| `invalid_action` | false | action 无法解析或不合法 |
| `tool_error_unrecoverable` | false or partial | 工具不可恢复且无替代方案 |
| `manual_stop` | false | 外部中止 |

Final answer 终止不自动等于成功。必须经过 evaluator 检查：

- 是否完成 success criteria
- 是否使用必要 observation
- 是否缺少必要工具调用
- 是否出现 hallucination 或 contradiction

## 14. 状态机如何产出训练数据

状态机每轮生成 step record，最终形成 trajectory。

对于 SFT：

```text
(model_input_state_t, expert_action_t)
```

对于 evaluator：

```text
(task_spec, model_trajectory, reference_trajectory)
```

对于 recovery training：

```text
(error_state_t, corrected_action_t)
```

对于 preference data：

```text
(successful_trajectory, failed_trajectory)
```

状态机必须保证：

- 每一步 action 和 observation 可追踪。
- 终止原因可追踪。
- failure types 可追踪。
- 不把 annotation/evaluator 信息泄漏进 model input。

## 15. 状态机伪代码

```text
context = init(task, tools)

while context.terminal_state is null:
    if context.progress.step >= context.progress.max_steps:
        terminate("max_steps_exceeded")
        break

    model_input_state = build_runtime_state(context)
    raw_output = model(model_input_state)
    action = parse_action(raw_output)

    if action is invalid:
        terminate("invalid_action")
        break

    if action.type == "tool_call":
        if not validate_action(action, context):
            terminate("invalid_action")
            break

        schema_result = validate_tool_schema(action)
        if schema_result is invalid:
            observation = make_schema_error(schema_result)
            handle_observation(context, action, observation)
            decide_recovery_or_terminate(context)
            continue

        observation = execute_tool(action)
        handle_observation(context, action, observation)
        decide_recovery_or_continue(context)
        continue

    if action.type == "final_answer":
        validate_final_answer(context, action)
        terminate_with_evaluator_result(context)
        break

    if action.type == "ask_user":
        handle_ask_user(context, action)
        break or wait_for_user
```

## 16. 实现检查清单

状态机实现达到优秀需要满足：

| 检查项 | 标准 |
|---|---|
| 状态完整 | 覆盖模型决策、解析、校验、执行、观察、恢复、终止 |
| 转移明确 | 每个状态有明确事件和下一状态 |
| 错误路径完整 | 覆盖 schema error、empty result、retryable error、unrecoverable error |
| 终止可解释 | 每条轨迹有 terminal reason |
| 训练可导出 | 每步可导出 SFT sample |
| 评测可挂接 | 每步可接入 step evaluator，整条轨迹可接入 trajectory evaluator |
| 多工具可表达 | 支持依赖图、参数绑定和等价顺序 |
| 恢复可评测 | 支持 recovery state、attempt count 和 recovery failure types |
| 防泄漏 | model input 不包含 annotation labels |

## 17. 覆盖矩阵

| 模块 | 当前状态 | 达到优秀的依据 |
|---|---|---|
| 主流程 | 达到优秀 | 覆盖从 user query 到 final answer 的完整路径 |
| 状态集合 | 达到优秀 | 明确定义每个状态责任、输入和输出 |
| 事件集合 | 达到优秀 | 定义所有主要转移事件 |
| 转移表 | 达到优秀 | 给出 current state、event、next state、主要动作 |
| 工具成功路径 | 达到优秀 | 覆盖 tool result 后继续决策 |
| schema error | 达到优秀 | 进入 recovery，并支持修参重试 |
| empty result | 达到优秀 | 支持放宽参数、换工具或解释无结果 |
| retryable error | 达到优秀 | 支持有限重试 |
| unrecoverable error | 达到优秀 | 支持终止或 fallback tool |
| ask user | 达到优秀 | 区分单轮终止和多轮等待 |
| 多工具 | 达到优秀 | 支持 dependency tracking、argument binding 和等价顺序 |
| 终止条件 | 达到优秀 | 终止原因清晰，且不把 final answer 自动等同成功 |
| 训练数据产出 | 达到优秀 | 明确 trajectory、SFT、recovery、preference 数据来源 |

## 18. 与 problem_formulation.md 的关系

本文档依赖 `problem_formulation.md` 中的概念和 schema：

- `State` 对应 runtime context 和 model input state。
- `Action` 对应模型在 `ModelDecision` 中输出的动作。
- `Observation` 对应 `ExecuteTool` 和 `HandleObservation` 的输出。
- `Transition` 对应本文档的状态转移表。
- `Trajectory` 由状态机 step records 聚合产生。
- `Evaluator` 挂接在 `ValidateFinalAnswer`、`TerminalSuccess`、`TerminalFailure` 之后。

可以把二者关系理解为：

```text
problem_formulation.md: 定义训练问题和数据契约
agent_state_machine.md: 定义运行时如何产生这些数据
```
