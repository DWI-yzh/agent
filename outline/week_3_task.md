# Week 3 工作计划：建立 baseline 与 evaluator

## 1. 本周目标

在不做训练的情况下，先跑通一个最小 Agent 闭环，并完成评测器搭建。

核心原则：

> 先评测，后训练。

没有 evaluator 的训练，本质上是盲调。

---

## 2. 本周要完成的核心工作

### 2.1 实现 baseline Agent

先实现一个 rule / prompt baseline，完成最小闭环：

1. 接收用户任务。
2. 按固定格式输出一个 tool call。
3. 调用 `env/executor.py` 执行工具。
4. 根据 executor 返回结果生成最终回答。
5. 记录完整执行日志。

建议新增文件：

```text
run_agent.py
```

baseline 不追求智能，只追求：

- 格式稳定
- 能调用工具
- 能记录结果
- 能被 evaluator 评估

---

## 3. 评估指标

本周需要在 `eval/metrics.py` 中实现以下指标。

### 3.1 Tool Selection Accuracy

判断在应调用工具的步骤中，模型选择的 tool 是否正确。

示例：

```text
gold_tool = "get_order"
pred_tool = "get_order"
=> correct
```

---

### 3.2 Argument Exact Match

判断参数是否完整且完全正确。

示例：

```text
gold_args = {"order_id": "48392"}
pred_args = {"order_id": "48392"}
=> correct
```

---

### 3.3 Slot-level F1

按字段粒度评估参数质量。

用于区分：

- 全部参数都错
- 部分字段正确
- 字段名正确但值错误
- 缺少字段

---

### 3.4 Schema Valid Rate

判断模型输出是否满足工具 schema，是否能被 executor 解析和执行。

重点关注：

- JSON 是否可解析
- tool 是否存在
- required field 是否齐全
- 参数类型是否正确

---

### 3.5 Execution Success Rate

判断 executor 是否返回 `success`。

该指标不只看模型输出格式，也会暴露：

- 参数值不存在
- 工具执行失败
- 空结果
- 非法取值

---

### 3.6 Task Success Rate

判断最终任务是否完成。

这是 task-level 指标，但本周不能只看它，必须结合 step-level 指标定位失败原因。

---

## 4. eval 数据集

本周先构造 50 到 100 条小规模评估任务。

建议新增文件：

```text
data/eval.jsonl
```

任务类型至少覆盖：

- 单步调用
- 短程多步
- 参数歧义
- 工具混淆
- 简单失败恢复

每条数据建议包含：

```json
{
  "task_id": "eval_001",
  "user_query": "查询订单 48392 的状态",
  "gold_tool": "get_order",
  "gold_args": {
    "order_id": "48392"
  },
  "task_type": "single_step"
}
```

---

## 5. 日志格式

baseline 执行时，需要记录 step-level 日志。

建议日志结构：

```json
{
  "task_id": "eval_001",
  "step_id": 1,
  "user_query": "查询订单 48392 的状态",
  "model_output": "{\"tool\": \"get_order\", \"args\": {\"order_id\": \"48392\"}}",
  "parsed_tool": "get_order",
  "parsed_args": {
    "order_id": "48392"
  },
  "executor_status": "success",
  "executor_result": {
    "status": "shipped"
  },
  "gold_tool": "get_order",
  "gold_args": {
    "order_id": "48392"
  }
}
```

建议输出到：

```text
logs/week3_eval_records.jsonl
```

---

## 6. 本周交付物

必须产出：

```text
run_agent.py
eval/metrics.py
data/eval.jsonl
logs/week3_eval_records.jsonl
```

其中：

- `run_agent.py`：跑通 baseline Agent。
- `eval/metrics.py`：计算评估指标。
- `data/eval.jsonl`：小规模评估集。
- `logs/week3_eval_records.jsonl`：baseline 执行日志。

---

## 7. 本周验收标准

达到以下标准，Week 3 视为完成：

1. baseline 可以端到端跑完。
2. 能调用 Week 2 的 executor。
3. 能输出 step-level 日志。
4. 能计算评估指标。
5. 能输出一张基础评估表。
6. 能从日志中定位基础错误类型。

---

## 8. 本周复盘问题

完成后回答：

1. baseline 主要错在 tool selection，还是 arg filling？
2. schema error 多，还是 execution error 多？
3. 当前任务集是否足以区分各类失败？
4. 哪类工具最容易被选错？
5. 哪类参数最容易填错？
6. 是否已经具备进入 Week 4 SFT 的评测基础？

---

## 9. 建议执行顺序

### Day 1

整理 `data/eval.jsonl`，先写 20 到 30 条单步任务。

### Day 2

实现 `run_agent.py`，跑通从用户输入到 executor 的最小闭环。

### Day 3

补齐日志记录，确保每一步都有结构化记录。

### Day 4

实现 `eval/metrics.py`，计算工具选择、参数匹配、schema 合法率和执行成功率。

### Day 5

扩充 eval 数据到 50 到 100 条，加入多步、歧义、工具混淆和失败恢复任务。

### Day 6

运行完整 baseline 评估，输出评估结果。

### Day 7

复盘 bad case，判断主要失败来源。

