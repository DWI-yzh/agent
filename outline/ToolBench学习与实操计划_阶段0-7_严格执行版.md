# ToolBench 学习与实操计划（阶段 0-7 严格执行版）

> 适用对象：已经具备 Python / LLM 基础，但对 Agent 设计、Agent 数据、Agent 训练与评估仍处在“半知半解”阶段的学习者。  
> 目标状态：走完本计划后，能够从“只会看 Agent 概念”成长为“能独立设计一个小型 Tool Agent sandbox、构造 SFT/DPO 数据、建立 step-level evaluator、完成一次小规模训练与错误诊断”的初级 Agent 选手。

---

## 0. 文档定位

这份文档不是 ToolBench 项目说明书，也不是论文复述，而是一份**严格可执行的学习手册**。

你要用它完成四件事：

1. **建立 Agent 训练的基本心智模型**：把 Agent 从“会聊天的模型”理解为“在环境中根据状态选择动作的策略函数”。
2. **学会 ToolBench 的核心结构**：数据、工具、轨迹、训练、推理、评测。
3. **跑通一个可控的工具调用训练闭环**：从样本分析、数据转换、SFT、评估，到 DPO/ORPO 数据构造。
4. **迁移成自己的 Mini-ToolBench**：为后续主动检索型 Agent、公安业务 Agent、Search-R1 / GRPO 学习打基础。

### 0.1 本文依据

- 你的原始 8 周 Agent 训练路线图：强调“工具编排型 Agent → SFT → DPO → step-level diagnosis → 主动检索型 Agent / RL”的学习顺序。
- ToolBench / ToolLLM：公开资料显示其定位为工具学习平台，提供 dataset、training scripts、evaluation scripts 和 ToolLLaMA，并围绕 API collection、instruction generation、solution path annotation、DFSDT、ToolEval、API Retriever 形成完整工具调用训练框架。
- ToolPrefer / StableToolBench 等后续工作：用于理解 ToolBench 生态中的偏好数据构造、稳定评测与 DPO 方向。

### 0.2 学习总原则

```text
不追求完整复现 ToolLLaMA；
追求把 ToolBench 拆成你能理解、能训练、能评估、能迁移的 Agent 训练模板。
```

你需要坚持以下原则：

| 原则 | 含义 | 为什么重要 |
|---|---|---|
| 先抽象，后工程 | 先理解 State / Action / Observation / Feedback，再看代码 | 避免被目录结构和脚本细节淹没 |
| 先小样本，后大数据 | 先手工分析 30 条样本，再考虑批量转换 | Agent 轨迹数据必须先看懂 |
| 先评估，后训练 | 没有 evaluator，不进入大规模训练 | 否则 loss 下降不等于 Agent 能力提升 |
| 先 close-domain，后 open-domain | 先固定工具集合，再做工具检索 | 降低变量数量，便于诊断 |
| 先 SFT，后 DPO/ORPO | 先学会正确轨迹，再压制错误动作 | 符合 Tool Agent 训练递进规律 |
| 先 ToolBench，再 Mini-ToolBench | 先学通用工具框架，再迁移业务 | 防止只会跑别人的项目 |

---

## 1. 总体阶段设计

| 阶段 | 名称 | 建议周期 | 核心问题 | 最终产出 |
|---|---:|---:|---|---|
| 阶段 0 | 学习边界与基础认知 | 0.5-1 天 | 我到底要从 ToolBench 学什么？ | 学习边界文档 |
| 阶段 1 | 项目结构与 Agent 抽象 | 1-2 天 | ToolBench 各模块对应 Agent 训练闭环的哪一环？ | 项目结构图 + 流程图 |
| 阶段 2 | 数据格式与轨迹结构 | 2-3 天 | 一个 tool-use trajectory 到底长什么样？ | 样本分析报告 |
| 阶段 3 | 最小工具调用闭环 | 3-5 天 | 模型如何输出 action，环境如何返回 observation？ | 可运行 inference demo |
| 阶段 4 | Evaluator 与错误分类 | 4-7 天 | 如何判断一次工具调用到底哪里错了？ | step-level evaluator |
| 阶段 5 | SFT 数据构造与小规模训练 | 5-7 天 | 如何把轨迹数据转成可训练样本？ | SFT 数据 + 训练报告 |
| 阶段 6 | DPO/ORPO 偏好数据构造 | 3-5 天 | 如何从失败轨迹构造 chosen/rejected？ | DPO 数据 + 错误迁移报告 |
| 阶段 7 | 迁移成业务 Mini-ToolBench | 3-5 天 | 如何把 ToolBench 方法迁移到自己的 Agent？ | 业务版 Mini-ToolBench 初版 |

建议总周期：**3-4 周专项学习**，对应你 8 周计划中的 Week 2-5。完成后再进入 OpenHands / Search-R1 / rStar 等后续项目。

---

# 阶段 0：学习边界与基础认知

## 0.1 阶段核心目标

明确 ToolBench 在你的 Agent 训练路线中的角色：

```text
ToolBench 不是单纯数据集；
不是单纯 benchmark；
不是必须完整复现的大项目；
而是你学习 Tool Agent 数据、训练、评估、诊断的样板工程。
```

本阶段要解决的核心问题：

1. ToolBench 是什么，不是什么？
2. 你要重点学习哪些模块？
3. 哪些内容暂时跳过？
4. ToolBench 和你的 8 周路线如何衔接？
5. 什么叫 Agent 训练，不只是普通 SFT？

## 0.2 阶段目的和意义

很多人学习 Agent 会直接陷入两个误区：

- 误区一：把 Agent 理解成“会调用工具的聊天模型”。
- 误区二：clone 一个仓库，跑不起来，然后以为自己不懂 Agent。

阶段 0 的意义是先把学习边界定清楚。你不需要立刻跑全量 ToolLLaMA，也不需要接 RapidAPI。你要先建立一个判断框架：

```text
Agent 训练 = 在特定环境中学习动作策略；
ToolBench = 用大规模 API 任务构造工具调用轨迹；
学习重点 = 轨迹数据 + 工具 schema + evaluator + SFT/DPO 数据。
```

这一步完成后，你后续看代码、看数据、做训练才不会失焦。

## 0.3 具体任务

### 任务 0.3.1：建立本地学习目录

建议目录：

```text
agent_training_study/
├── repos/
│   └── ToolBench/
├── notes/
├── data_samples/
├── scripts/
├── eval/
├── reports/
└── mini_toolbench/
```

### 任务 0.3.2：clone ToolBench

```bash
git clone https://github.com/OpenBMB/ToolBench.git repos/ToolBench
cd repos/ToolBench
```

### 任务 0.3.3：阅读三类材料

优先阅读：

1. `README.md`
2. ToolLLM 论文摘要、方法部分、数据构造部分、评测部分
3. README 中关于 data、training、inference、evaluation 的说明

阅读时只回答问题，不急于执行脚本。

### 任务 0.3.4：写学习边界文档

创建：

```text
notes/toolbench_learning_scope.md
```

至少回答：

1. 我为什么学习 ToolBench？
2. ToolBench 对应 Agent 训练闭环中的哪些环节？
3. 我暂时不学习哪些内容？
4. 我最终要迁移出什么？
5. 我如何判断自己完成了 ToolBench 学习？

### 任务 0.3.5：建立术语卡片

创建：

```text
notes/toolbench_glossary.md
```

至少解释以下术语：

| 术语 | 你要写清楚的内容 |
|---|---|
| Tool | 工具是什么，和普通函数/API有什么区别 |
| API schema | 工具调用时模型能看到的结构化说明 |
| Instruction | 用户自然语言任务 |
| Trajectory | 从用户任务到最终答案的多步 action/observation 链 |
| Action | 模型选择的工具与参数 |
| Observation | 工具执行后的返回结果 |
| DFSDT | ToolBench 中的深度优先搜索式决策树标注方法 |
| ToolEval | ToolBench 中的自动评测器 |
| API Retriever | 在大规模工具库中召回候选工具的检索器 |
| ToolLLaMA | 基于 ToolBench 数据训练得到的工具调用模型 |

## 0.4 产出内容

本阶段结束时至少要有：

```text
notes/toolbench_learning_scope.md
notes/toolbench_glossary.md
notes/stage0_review.md
```

`stage0_review.md` 建议包含：

```text
1. 本阶段我理解了什么
2. 我仍然不理解什么
3. 我决定暂时跳过什么
4. 下一阶段要重点看哪些目录
```

## 0.5 达成指标

| 指标 | 合格标准 | 优秀标准 |
|---|---|---|
| ToolBench 定位 | 能说出它是工具学习平台 | 能区分 dataset / training / inference / eval 四部分 |
| Agent 抽象 | 能说出 State / Action / Observation | 能把 ToolBench 样本映射到这三者 |
| 学习边界 | 能列出要学和不学的内容 | 能解释为什么不完整复现 ToolLLaMA |
| 术语掌握 | 能解释 10 个核心术语 | 能用自己的业务例子类比说明 |

阶段通过条件：

```text
你能在不看资料的情况下，用 5 分钟讲清楚：
“我为什么学 ToolBench，以及它如何帮助我理解 Agent 训练。”
```

---

# 阶段 1：项目结构与 Agent 抽象对齐

## 1.1 阶段核心目标

把 ToolBench 的工程目录映射到 Agent 训练闭环：

```text
数据构造 → 训练 → 推理 → 工具执行 → 评测 → 错误分析
```

你要知道每个目录做什么，不要求每行代码都看懂。

## 1.2 阶段目的和意义

初学者常见问题是：

```text
看了 README，知道 ToolBench 很厉害；
打开代码仓库后，不知道从哪个目录开始；
最后只会照着命令跑，不知道系统结构。
```

阶段 1 的意义是建立项目地图。只要项目地图清楚，后面做数据分析、训练、评测时就能快速定位模块。

你要形成一个核心认知：

```text
Agent 项目不是一个 train.py；
它至少包括 environment、data、policy、executor、evaluator、logger 六部分。
```

## 1.3 具体任务

### 任务 1.3.1：扫描目录结构

执行：

```bash
cd repos/ToolBench
tree -L 3 > ../../notes/toolbench_tree_l3.txt
```

如果没有 `tree`，可用：

```bash
find . -maxdepth 3 -type f | sort > ../../notes/toolbench_files_l3.txt
```

### 任务 1.3.2：建立目录功能表

重点查看这些模块：

| 目录/模块 | 重点问题 |
|---|---|
| `data/` | 原始数据、训练数据、评测数据如何组织 |
| `toolbench/` | 核心代码入口在哪 |
| `toolbench/train/` | SFT 训练入口和训练格式是什么 |
| `toolbench/inference/` | 推理时如何逐步调用工具 |
| `toolbench/retrieval/` | API Retriever 如何训练与调用 |
| `toolbench/tooleval/` | 自动评测如何实现 |
| `preprocess/` | 原始数据如何转成训练格式 |
| `scripts/` | 官方提供了哪些实验脚本 |

### 任务 1.3.3：画端到端流程图

创建：

```text
notes/toolbench_pipeline.md
```

流程图至少包含：

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

### 任务 1.3.4：建立 Agent 抽象映射表

创建：

```text
notes/toolbench_agent_mapping.md
```

表格如下：

| Agent 抽象 | ToolBench 对应内容 | 你自己的理解 | 后续可迁移到业务的形式 |
|---|---|---|---|
| State | user instruction + tool schema + history + observation |  | 案情状态 + 规则候选 + 历史问答 |
| Action | tool name + arguments |  | 调用抽取/检索/判断/推荐工具 |
| Observation | API result / error / empty result |  | 法条片段、抽取结果、规则判断结果 |
| Policy | ToolLLaMA / base LLM |  | Qwen3-4B + LoRA Agent |
| Evaluator | ToolEval |  | 规则评测 + LLM judge + 人工抽检 |
| Feedback | pass / fail / preference |  | 正确轨迹、失败轨迹、DPO pair、reward |

### 任务 1.3.5：写阶段复盘

创建：

```text
notes/stage1_project_structure_review.md
```

回答：

1. ToolBench 的训练入口在哪里？
2. ToolBench 的推理入口在哪里？
3. ToolBench 的评测入口在哪里？
4. ToolBench 中环境与模型如何交互？
5. 如果我要改成自己的工具，应该改哪些部分？

## 1.4 产出内容

```text
notes/toolbench_tree_l3.txt
notes/toolbench_pipeline.md
notes/toolbench_agent_mapping.md
notes/stage1_project_structure_review.md
```

## 1.5 达成指标

| 指标 | 合格标准 | 优秀标准 |
|---|---|---|
| 项目结构理解 | 能说出 5 个核心目录作用 | 能把 8 个目录映射到 Agent 闭环 |
| 流程理解 | 能画出 instruction→tool call→observation | 能解释多步调用循环 |
| 迁移意识 | 知道哪些模块能改 | 能说明如何替换成自己的工具环境 |
| 复盘质量 | 能写出阶段总结 | 能指出 3 个后续风险点 |

阶段通过条件：

```text
拿到任意一个 ToolBench 文件路径，
你能判断它属于 data / train / inference / retrieval / eval / preprocess 中哪一类。
```

---

# 阶段 2：数据格式与轨迹结构分析

## 2.1 阶段核心目标

真正看懂 ToolBench 的数据，尤其是工具调用轨迹。

你要掌握：

```text
一个 trajectory 如何被拆成多个 step；
一个 step 如何变成 SFT 样本；
一个错误 step 如何变成 DPO rejected 样本。
```

## 2.2 阶段目的和意义

Agent 训练和普通问答 SFT 最大的区别在数据。

普通问答数据通常是：

```text
instruction → answer
```

Tool Agent 数据是：

```text
state_t → action_t → observation_t → state_{t+1} → action_{t+1} → ... → final
```

如果你看不懂 trajectory，就无法真正理解：

1. 为什么 Agent 训练不是普通文本生成训练。
2. 为什么必须做 step-level evaluator。
3. 为什么 DPO pair 要从失败动作中构造。
4. 为什么后续 RL 要依赖环境反馈。

## 2.3 具体任务

### 任务 2.3.1：抽样 G1 / G2 / G3 数据

目标：每类至少手工分析 10 条。

创建脚本：

```text
scripts/sample_toolbench_data.py
```

输出：

```text
data_samples/g1_10.jsonl
data_samples/g2_10.jsonl
data_samples/g3_10.jsonl
```

如果数据目录结构变化，任务不变：你只需要保证抽到单工具、多工具、复杂多工具三类样本。

### 任务 2.3.2：逐条手工标注轨迹

对每条样本做以下分析：

| 分析项 | 要回答的问题 |
|---|---|
| user instruction | 用户到底想完成什么 |
| required tools | 需要哪些工具 |
| first action | 第一步为什么调用这个工具 |
| arguments | 参数从 instruction 还是 history 中来 |
| observation | 工具返回了什么 |
| next action | observation 如何影响下一步 |
| final answer | 最终答案是否基于工具结果 |
| possible failure | 这条轨迹可能错在哪里 |

### 任务 2.3.3：转成 State / Action / Observation 格式

创建：

```text
scripts/convert_to_sao_examples.py
```

输出格式建议：

```json
{
  "task_id": "example_001",
  "step_id": 1,
  "state": {
    "user_query": "...",
    "available_tools": ["..."],
    "history": [],
    "last_observation": null
  },
  "action": {
    "tool_name": "...",
    "arguments": {}
  },
  "observation": {
    "status": "success",
    "result": {}
  },
  "is_final": false
}
```

### 任务 2.3.4：总结 G1 / G2 / G3 的能力差异

创建：

```text
notes/g1_g2_g3_complexity_analysis.md
```

建议表格：

| 类型 | 工具数量 | 典型能力 | 主要错误 | 适合训练阶段 |
|---|---:|---|---|---|
| G1 | 单工具 | tool selection / arg filling | 选错工具、缺参数 | SFT 初期 |
| G2 | 同类多工具 | 多步顺序、局部 planning | 顺序错、结果误读 | SFT 中期 / DPO |
| G3 | 跨集合多工具 | 复杂规划、跨工具组合 | 长程失败、无效停止 | DPO / RL 前置 |

### 任务 2.3.5：建立数据质量问题清单

记录：

1. 哪些样本工具调用很清楚？
2. 哪些样本轨迹过长？
3. 哪些样本参数从上下文中难以恢复？
4. 哪些样本适合做 SFT？
5. 哪些样本适合构造 DPO pair？

## 2.4 产出内容

```text
data_samples/g1_10.jsonl
data_samples/g2_10.jsonl
data_samples/g3_10.jsonl
data_samples/state_action_observation_examples.jsonl
notes/toolbench_data_analysis.md
notes/g1_g2_g3_complexity_analysis.md
notes/toolbench_data_quality_notes.md
```

## 2.5 达成指标

| 指标 | 合格标准 | 优秀标准 |
|---|---|---|
| 样本分析数量 | ≥ 30 条 | ≥ 50 条 |
| SAO 转换 | 能转 10 条 | 能转 30 条以上 |
| G1/G2/G3 理解 | 能说出差异 | 能对应到训练阶段 |
| 错误预判 | 能列出 5 类错误 | 能为每类错误设计 evaluator |
| 数据判断 | 能区分 SFT 样本和 DPO 样本 | 能设计自动过滤规则 |

阶段通过条件：

```text
你能拿一条 ToolBench 样本，
手工拆成多个 step-level 训练样本，
并指出每一步可能出现的 tool selection / argument / observation 错误。
```

---

# 阶段 3：最小工具调用闭环

## 3.1 阶段核心目标

跑通或仿制一个最小 Agent 执行闭环：

```text
用户任务
→ 模型输出 action
→ 解析 tool call
→ executor 执行
→ 返回 observation
→ 模型继续或停止
```

这一步的目标不是追求模型效果，而是理解 Agent 运行机制。

## 3.2 阶段目的和意义

如果只看数据，你会知道“轨迹长什么样”；但如果不跑闭环，你不会真正理解：

1. action 解析为什么容易失败。
2. schema 校验为什么重要。
3. observation 为什么必须标准化。
4. 多步调用日志为什么是后续评估和 DPO 的基础。
5. 为什么 Agent 系统中模型只是 policy，环境同样重要。

阶段 3 是从“读数据”进入“跑系统”的分界点。

## 3.3 具体任务

### 任务 3.3.1：选择 close-domain 模式

不要一开始做 open-domain tool retrieval。

本阶段固定 5-10 个工具 schema，让模型只在给定工具中选择。

目标：降低变量，专注观察 action / observation 循环。

### 任务 3.3.2：实现统一 action parser

创建：

```text
scripts/action_parser.py
```

输入：模型输出文本。

输出：

```json
{
  "valid": true,
  "tool_name": "...",
  "arguments": {},
  "parse_error": null
}
```

至少处理：

1. JSON 可解析。
2. JSON 不可解析。
3. 缺少 tool_name。
4. arguments 不是 dict。
5. tool_name 不在候选工具中。

### 任务 3.3.3：实现 mock executor

创建：

```text
scripts/mock_executor.py
```

统一返回：

```json
{
  "status": "success",
  "tool": "...",
  "args": {},
  "result": {},
  "error_code": null,
  "error_message": null
}
```

错误返回：

```json
{
  "status": "error",
  "tool": "...",
  "args": {},
  "result": null,
  "error_code": "missing_required_field",
  "error_message": "required field `city` is missing"
}
```

### 任务 3.3.4：跑 50 条最小轨迹

创建：

```text
scripts/run_minimal_tool_agent.py
```

每条轨迹记录：

```json
{
  "task_id": "...",
  "step_id": 1,
  "user_query": "...",
  "available_tools": [],
  "model_output": "...",
  "parsed_action": {},
  "executor_result": {},
  "next_state": "...",
  "is_final": false
}
```

输出：

```text
logs/minimal_rollout_50.jsonl
```

### 任务 3.3.5：主动构造失败场景

至少构造 6 类失败：

| 错误类型 | 构造方式 |
|---|---|
| wrong tool | 把目标工具替换成相似工具 |
| missing argument | 删除必填参数 |
| wrong argument value | 修改参数值 |
| invalid schema | 输出非 JSON 或字段名错误 |
| empty result | executor 返回空结果 |
| no recovery | 第一次失败后继续胡编最终答案 |

输出：

```text
logs/failure_injection_cases.jsonl
```

## 3.4 产出内容

```text
scripts/action_parser.py
scripts/mock_executor.py
scripts/run_minimal_tool_agent.py
logs/minimal_rollout_50.jsonl
logs/failure_injection_cases.jsonl
notes/stage3_minimal_agent_review.md
```

## 3.5 达成指标

| 指标 | 合格标准 | 优秀标准 |
|---|---|---|
| action parser | 能解析合法 JSON | 能识别 5 类解析错误 |
| executor | 能返回 success/error | 返回格式完全统一 |
| rollout | ≥ 50 条 | ≥ 100 条 |
| 日志完整性 | 有 task_id / step_id / action / observation | 可直接用于 evaluator |
| 失败注入 | ≥ 6 类 | 每类 ≥ 5 条 |

阶段通过条件：

```text
你能不依赖 ToolBench 全量环境，
自己跑通一个最小 tool-use Agent 闭环，
并输出完整 step-level 日志。
```

---

# 阶段 4：Evaluator 与错误分类体系

## 4.1 阶段核心目标

建立一个能诊断工具调用过程的 step-level evaluator。

你要从只看最终答案，转向同时评估：

```text
tool 是否选对
参数是否填对
schema 是否合法
工具是否执行成功
observation 是否被正确理解
失败后是否能恢复
最终答案是否基于工具结果
```

## 4.2 阶段目的和意义

没有 evaluator，Agent 训练就是盲调。

SFT loss 降了，模型可能只是更会输出格式；DPO loss 降了，模型可能只是更偏向短答案。只有 step-level evaluator 能告诉你：

1. SFT 提升的是格式、工具选择，还是参数填充？
2. DPO 压制的是 wrong tool，还是 hallucinated answer？
3. 多步任务失败到底发生在第几步？
4. 是否有必要进入更复杂的 RL？

这一阶段是你从“会跑模型”进入“会诊断 Agent”的关键。

## 4.3 具体任务

### 任务 4.3.1：定义 gold label 格式

每条 eval 样本至少包含：

```json
{
  "task_id": "...",
  "user_query": "...",
  "gold_steps": [
    {
      "step_id": 1,
      "gold_tool": "...",
      "gold_args": {},
      "required_slots": ["..."],
      "optional_slots": ["..."]
    }
  ],
  "gold_final_answer": "..."
}
```

### 任务 4.3.2：实现 step-level metrics

创建：

```text
eval/metrics.py
```

至少实现：

| 指标 | 含义 |
|---|---|
| Schema Valid Rate | 模型输出是否可解析、可执行 |
| Tool Selection Accuracy | 工具名是否选对 |
| Argument Exact Match | 参数整体是否完全正确 |
| Slot-level Precision / Recall / F1 | 参数字段粒度质量 |
| Execution Success Rate | executor 是否返回 success |
| Step Order Accuracy | 多步任务顺序是否正确 |
| Recovery Success Rate | 失败后是否能正确修复 |
| Hallucinated Final Answer Rate | 是否未基于 observation 胡编 |

### 任务 4.3.3：建立错误分类体系

创建：

```text
eval/error_taxonomy.py
```

初始错误类型：

| 编码 | 错误类型 | 说明 |
|---|---|---|
| E01 | wrong_tool | 选择了错误工具 |
| E02 | missing_argument | 缺少必填参数 |
| E03 | wrong_argument_value | 参数值错误 |
| E04 | invalid_schema | 输出格式不可解析或不符合 schema |
| E05 | wrong_step_order | 多步顺序错误 |
| E06 | observation_misread | 误读工具返回结果 |
| E07 | no_recovery | 工具失败后未恢复 |
| E08 | hallucinated_answer | 未依据工具结果直接编造答案 |
| E09 | redundant_tool_call | 多余调用工具 |
| E10 | premature_stop | 过早停止，信息不足 |

### 任务 4.3.4：实现自动 bad case 输出

创建：

```text
eval/run_eval.py
```

输出：

```text
reports/eval_summary.md
reports/bad_cases.jsonl
reports/error_distribution.csv
```

每条 bad case 至少包含：

```json
{
  "task_id": "...",
  "step_id": 1,
  "error_type": "wrong_tool",
  "model_action": {},
  "gold_action": {},
  "observation": {},
  "comment": "模型选择了相似但不适用的工具"
}
```

### 任务 4.3.5：建立人工复核模板

创建：

```text
reports/manual_review_template.xlsx 或 reports/manual_review_template.csv
```

字段建议：

| 字段 | 说明 |
|---|---|
| task_id | 任务 ID |
| step_id | 步骤 ID |
| model_action | 模型动作 |
| gold_action | 标准动作 |
| auto_error_type | 自动错误类型 |
| human_error_type | 人工修正错误类型 |
| severity | 严重程度 |
| note | 备注 |

## 4.4 产出内容

```text
eval/metrics.py
eval/error_taxonomy.py
eval/run_eval.py
reports/eval_summary.md
reports/bad_cases.jsonl
reports/error_distribution.csv
reports/manual_review_template.csv
notes/stage4_evaluator_design.md
```

## 4.5 达成指标

| 指标 | 合格标准 | 优秀标准 |
|---|---|---|
| step-level 指标 | ≥ 6 个 | ≥ 8 个 |
| 错误类型 | ≥ 8 类 | ≥ 10 类并有示例 |
| bad case 输出 | 可自动生成 | 可按错误类型聚合 |
| 人工复核 | 有模板 | 能回写修正标签 |
| 模型对比 | 可对比 baseline vs SFT | 可输出错误迁移表 |

阶段通过条件：

```text
你能对一批 Agent 轨迹输出错误分布，
并说明模型主要错在 tool selection、arg filling、schema、recovery 还是 final answer。
```

---

# 阶段 5：SFT 数据构造与小规模训练

## 5.1 阶段核心目标

把 ToolBench trajectory 转成适合 Qwen / LLaMA-Factory 的 step-level SFT 数据，并完成一次小规模 LoRA 训练。

目标不是训练出强模型，而是掌握：

```text
trajectory → step-level sample → SFT dataset → LoRA training → evaluator 对比
```

## 5.2 阶段目的和意义

SFT 在 Tool Agent 中的主要价值不是“让模型更会聊天”，而是：

1. 学会稳定输出工具调用格式。
2. 学会在候选工具中选择正确工具。
3. 学会从用户 query / history / observation 中填参数。
4. 学会短程多步调用模式。
5. 给后续 DPO / RL 提供基础策略。

如果没有 SFT baseline，直接做 DPO 或 GRPO 会变得难以诊断。

## 5.3 具体任务

### 任务 5.3.1：选择训练子集

建议最小规模：

| 数据类型 | 数量 |
|---|---:|
| G1 单工具 | 500 |
| G2 多工具 | 300 |
| Eval | 100 |

标准规模：

| 数据类型 | 数量 |
|---|---:|
| G1 单工具 | 1000 |
| G2 多工具 | 1000 |
| G3 复杂多工具 | 500 |
| Eval | 200-500 |

### 任务 5.3.2：将 trajectory 拆成 step-level SFT

一条多步轨迹不要只做一个样本，而要拆成多个样本：

```text
step 1 state → step 1 action
step 2 state + observation_1 → step 2 action
step 3 state + observation_1/2 → final answer
```

### 任务 5.3.3：设计 SFT 数据格式

建议格式：

```json
{
  "instruction": "你是一个工具调用 Agent。请根据用户问题、可用工具和历史观察，输出下一步工具调用。如果信息充分，则输出最终答案。",
  "input": "用户问题：...\n可用工具：...\n历史观察：...",
  "output": "{\"tool_name\": \"...\", \"arguments\": {...}}"
}
```

最终答案格式：

```json
{
  "final_answer": "..."
}
```

### 任务 5.3.4：编写转换脚本

创建：

```text
scripts/convert_toolbench_to_qwen_sft.py
```

输出：

```text
data/toolbench_sft_train.jsonl
data/toolbench_sft_eval.jsonl
```

脚本至少支持：

1. 只转 G1。
2. 转 G1 + G2。
3. 限制最大步数。
4. 过滤超长样本。
5. 检查 output 是否合法 JSON。

### 任务 5.3.5：训练 LoRA

建议先用你熟悉的训练栈：

```text
Qwen3-4B-Instruct / Qwen2.5-3B-Instruct
+ LLaMA-Factory
+ LoRA / QLoRA
```

建议训练设置：

| 参数 | 建议 |
|---|---|
| cutoff_len | 2048 或 4096 |
| epoch | 2-3 |
| LoRA rank | 16 或 32 |
| learning rate | 1e-4 到 2e-4 起试 |
| eval interval | 每 100-200 step |

### 任务 5.3.6：SFT 前后对比

对比对象：

1. Prompt baseline。
2. SFT model。

对比指标：

```text
Schema Valid Rate
Tool Selection Accuracy
Argument Exact Match
Slot-level F1
Execution Success Rate
Task Success Rate
Wrong Tool Rate
Invalid Schema Rate
```

## 5.4 产出内容

```text
scripts/convert_toolbench_to_qwen_sft.py
data/toolbench_sft_train.jsonl
data/toolbench_sft_eval.jsonl
train/sft_config.yaml
reports/sft_data_statistics.md
reports/sft_before_after_report.md
reports/sft_bad_cases.jsonl
notes/stage5_sft_review.md
```

## 5.5 达成指标

| 指标 | 合格标准 | 优秀标准 |
|---|---|---|
| SFT 样本 | ≥ 800 | ≥ 2500 |
| Eval 样本 | ≥ 100 | ≥ 300 |
| 格式合法率 | 数据转换后 output JSON 合法率 ≥ 98% | ≥ 99.5% |
| Schema Valid Rate | 比 baseline 提升 ≥ 15% | ≥ 20% |
| Tool Selection Accuracy | 比 baseline 提升 ≥ 8% | ≥ 10% |
| Argument Exact Match | 比 baseline 提升 ≥ 8% | ≥ 10% |
| Invalid Schema Rate | 明显下降 | 下降 ≥ 50% |
| 训练报告 | 有前后对比 | 有错误归因分析 |

阶段通过条件：

```text
你能证明 SFT 到底提升了哪一类 Agent 能力，
而不是只说“loss 降了”或“模型感觉更好了”。
```

---

# 阶段 6：DPO / ORPO 偏好数据构造

## 6.1 阶段核心目标

从 SFT 的“模仿正确轨迹”进入偏好优化的“压制错误动作”。

本阶段不追求大规模 DPO，而是要学会：

```text
失败轨迹 → 错误分类 → chosen/rejected pair → DPO/ORPO 数据 → 错误结构变化评估
```

## 6.2 阶段目的和意义

SFT 的限制很明显：它只告诉模型“正确动作长什么样”，但没有显式告诉模型“哪些相似动作是错误的”。

DPO/ORPO 对 Tool Agent 的价值在于：

1. 压制相似但错误的工具选择。
2. 压制缺字段、错字段、错值的参数输出。
3. 压制工具失败后的胡编答案。
4. 强化失败恢复动作。
5. 优化 stopping policy。

这一阶段会让你理解：Agent 训练中的 preference pair 不应该只是“好回答 vs 坏回答”，而应该是**正确动作 vs 错误动作**。

## 6.3 具体任务

### 任务 6.3.1：用 SFT 模型 rollout

对 eval 集运行 SFT 模型，保存完整轨迹：

```text
logs/sft_rollout_eval.jsonl
```

每步必须包含：

```json
{
  "task_id": "...",
  "step_id": 1,
  "prompt": "...",
  "model_output": "...",
  "parsed_action": {},
  "gold_action": {},
  "executor_result": {},
  "error_type": "..."
}
```

### 任务 6.3.2：筛选失败步骤

用阶段 4 的 evaluator 自动筛选：

```text
wrong_tool
missing_argument
wrong_argument_value
invalid_schema
no_recovery
premature_stop
hallucinated_answer
```

输出：

```text
logs/sft_failed_steps.jsonl
```

### 任务 6.3.3：构造 chosen/rejected pair

每条 pair 格式：

```json
{
  "prompt": "用户问题 + 工具 schema + 历史 observation",
  "chosen": "正确下一步工具调用或最终答案",
  "rejected": "模型产生的错误工具调用或错误最终答案",
  "pair_type": "tool_selection",
  "error_type": "wrong_tool"
}
```

### 任务 6.3.4：设计 pair 类型分布

建议比例：

| Pair 类型 | 占比 | 目的 |
|---|---:|---|
| tool_selection | 30%-35% | 压制错误工具选择 |
| argument | 30%-35% | 压制缺参、错参 |
| recovery | 15%-20% | 学会失败修复 |
| stopping | 10%-15% | 学会继续/停止判断 |
| final_answer | 5%-10% | 压制胡编答案 |

### 任务 6.3.5：训练 DPO / ORPO 小实验

建议先做小规模：

```text
DPO pair：100-500 条
base model：SFT checkpoint
训练轮数：1-2 epoch
```

不需要追求大模型性能，只看错误变化。

### 任务 6.3.6：输出错误迁移报告

对比：

```text
Prompt baseline
SFT
SFT + DPO/ORPO
```

重点看：

1. wrong_tool 是否下降。
2. missing_argument 是否下降。
3. invalid_schema 是否反弹。
4. recovery 是否变好。
5. premature_stop 是否变多。
6. final_answer 是否更保守。

## 6.4 产出内容

```text
logs/sft_rollout_eval.jsonl
logs/sft_failed_steps.jsonl
scripts/build_dpo_pairs_from_rollout.py
data/toolbench_dpo_train.jsonl
data/toolbench_dpo_eval.jsonl
train/dpo_config.yaml
reports/dpo_pair_statistics.md
reports/dpo_error_shift_report.md
notes/stage6_preference_learning_review.md
```

## 6.5 达成指标

| 指标 | 合格标准 | 优秀标准 |
|---|---|---|
| rollout 日志 | ≥ 100 条任务 | ≥ 300 条任务 |
| DPO pair | ≥ 100 条 | ≥ 500 条 |
| pair 类型 | ≥ 3 类 | ≥ 5 类 |
| rejected 来源 | 至少来自真实模型失败 | 真实失败 + 规则扰动结合 |
| 错误改善 | 至少 1 类错误下降 | 至少 2-3 类错误下降 |
| 报告质量 | 有指标对比 | 有错误迁移解释 |

阶段通过条件：

```text
你能解释：
为什么这个 rejected 是错的，
为什么 chosen 更好，
以及 DPO 后具体改善了哪类 Agent 错误。
```

---

# 阶段 7：迁移成自己的 Mini-ToolBench

## 7.1 阶段核心目标

把 ToolBench 的方法迁移到你的业务或自定义场景中，形成一个小型可控 Agent 训练项目。

这一步才是本计划的最终目的。

你要做的不是继续研究 ToolBench，而是形成：

```text
自己的 tools
自己的 schemas
自己的 executor
自己的 trajectory data
自己的 evaluator
自己的 SFT/DPO pipeline
```

## 7.2 阶段目的和意义

如果你只会运行 ToolBench，那还不是 Agent 选手，只是项目使用者。

只有能把 ToolBench 迁移到自己的任务上，你才真正掌握了：

1. 如何定义 Agent 动作空间。
2. 如何设计工具 schema。
3. 如何构造轨迹数据。
4. 如何做 step-level 评估。
5. 如何用失败轨迹训练模型。
6. 如何判断是否需要进一步进入 Search Agent / RL。

这一阶段完成后，你就具备了初级 Agent 工程与训练能力。

## 7.3 具体任务

### 任务 7.3.1：定义业务场景

建议选择一个你熟悉且可控的场景，例如：

```text
案情文本 → 要素抽取 → 法条检索 → 刑/非刑判断 → 下一问推荐
```

不要一开始做全流程复杂系统。先做 5 个工具以内的小型闭环。

### 任务 7.3.2：设计 5 个工具

建议工具：

| 工具名 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `extract_case_facts` | case_text | facts | 抽取案情事实要素 |
| `search_legal_rules` | query, case_type | rule_snippets | 检索法条/规则片段 |
| `classify_case` | facts, rules | case_type_result | 判断刑事/行政/不足判断 |
| `check_upgrade_conditions` | facts, case_type | upgrade_result | 判断是否存在升格情形 |
| `recommend_next_question` | case_state, missing_slots | question | 推荐下一问 |

如果暂时不做公安业务，也可以换成通用场景：订单客服、会议安排、文档检索、代码诊断等。

### 任务 7.3.3：编写 tool schema

创建：

```text
mini_toolbench/env/schemas.py
```

每个 schema 至少包含：

```json
{
  "name": "extract_case_facts",
  "description": "从案情文本中抽取关键事实要素",
  "required_parameters": {
    "case_text": "string"
  },
  "optional_parameters": {},
  "returns": {
    "facts": "object"
  }
}
```

### 任务 7.3.4：实现 mock executor

创建：

```text
mini_toolbench/env/executor.py
```

先不要接真实知识库，使用本地 mock 规则返回。

统一返回：

```json
{
  "status": "success",
  "tool": "search_legal_rules",
  "args": {},
  "result": {},
  "error_code": null,
  "error_message": null
}
```

### 任务 7.3.5：构造初始训练数据

建议规模：

| 数据 | 数量 |
|---|---:|
| 单步 SFT | 100 |
| 两步 SFT | 100 |
| 三步 SFT | 50 |
| Eval | 50 |
| DPO pair | 100 |

数据结构：

```json
{
  "task_id": "case_001",
  "user_query": "...",
  "trajectory": [
    {
      "state": {},
      "action": {
        "tool_name": "extract_case_facts",
        "arguments": {}
      },
      "observation": {}
    }
  ],
  "final_answer": "..."
}
```

### 任务 7.3.6：复用阶段 4 的 evaluator

将指标迁移到你的业务工具：

```text
Schema Valid Rate
Tool Selection Accuracy
Argument Exact Match
Execution Success Rate
Fact Coverage
Legal Rule Recall
Classification Accuracy
Question Relevance
```

新增业务指标：

| 指标 | 含义 |
|---|---|
| Fact Coverage | 关键事实是否抽全 |
| Legal Rule Recall | 是否检索到关键法条/规则 |
| Rule-groundedness | 判断是否依据检索规则 |
| Classification Accuracy | 刑/非刑或类型判断是否正确 |
| Next-question Relevance | 下一问是否围绕缺失事实 |

### 任务 7.3.7：跑完整小闭环

执行顺序：

```text
1. prompt baseline
2. evaluator 评估
3. 构造 SFT 数据
4. LoRA SFT 小训练
5. evaluator 再评估
6. 收集失败轨迹
7. 构造 DPO pair
8. 输出阶段报告
```

### 任务 7.3.8：写最终学习总结

创建：

```text
reports/toolbench_final_learning_report.md
```

至少回答：

1. 我现在如何理解 Agent？
2. 我能独立设计哪些 Agent 组件？
3. 我能构造什么类型的训练数据？
4. 我能评估什么类型的错误？
5. 我目前离 RL / GRPO 还差什么？
6. 下一步应该接 OpenHands、Search-R1，还是自建业务 Agent？

## 7.4 产出内容

```text
mini_toolbench/
├── env/
│   ├── schemas.py
│   ├── tools.py
│   └── executor.py
├── data/
│   ├── sft_train.jsonl
│   ├── sft_eval.jsonl
│   ├── dpo_train.jsonl
│   └── eval.jsonl
├── scripts/
│   ├── run_agent.py
│   ├── convert_to_sft.py
│   └── build_dpo_pairs.py
├── eval/
│   ├── metrics.py
│   ├── error_taxonomy.py
│   └── run_eval.py
├── train/
│   ├── sft_config.yaml
│   └── dpo_config.yaml
└── reports/
    ├── baseline_report.md
    ├── sft_report.md
    ├── dpo_report.md
    └── final_learning_report.md
```

## 7.5 达成指标

| 指标 | 合格标准 | 优秀标准 |
|---|---|---|
| 自定义工具 | ≥ 5 个 | ≥ 5 个且 schema 完整 |
| executor | mock 可运行 | 支持 success/error/empty result |
| SFT 样本 | ≥ 200 | ≥ 300 |
| Eval 样本 | ≥ 50 | ≥ 100 |
| DPO pair | ≥ 100 | ≥ 300 |
| evaluator | 可运行 | 能输出错误分布和 bad case |
| SFT 对比 | 有前后指标 | 至少 2 个指标明显提升 |
| 最终报告 | 有总结 | 能说明下一阶段进入 Search Agent/RL 的条件 |

阶段通过条件：

```text
你能从零定义一个 5-tool Agent，
构造训练数据，
跑出 baseline/SFT 对比，
并基于失败轨迹构造 DPO pair。
```

---

# 8. 阶段间门槛检查

每个阶段结束后，必须做门槛检查。未通过时不要进入下一阶段。

| 阶段 | 进入下一阶段前必须满足 |
|---|---|
| 阶段 0 | 能说清 ToolBench 学习边界，不再把目标设为完整复现 ToolLLaMA |
| 阶段 1 | 能画出 ToolBench data→train→inference→eval 流程 |
| 阶段 2 | 能手工拆解至少 30 条 trajectory |
| 阶段 3 | 能跑通最小 action→executor→observation 闭环 |
| 阶段 4 | 能自动输出 step-level metrics 和 bad case |
| 阶段 5 | 能证明 SFT 改善了哪类 Agent 能力 |
| 阶段 6 | 能构造至少 100 条高质量 DPO pair |
| 阶段 7 | 能完成一个业务版 Mini-ToolBench 初版 |

---

# 9. 最终能力验收表

完成阶段 0-7 后，你应该具备以下能力。

| 能力 | 入门标准 | 自测问题 |
|---|---|---|
| Agent 抽象 | 能用 State / Action / Observation 描述一个 Agent | 当前状态是什么？动作空间是什么？反馈是什么？ |
| 工具设计 | 能定义 tool schema 和 executor | 这个工具需要哪些参数？返回如何标准化？ |
| 轨迹理解 | 能拆解多步 tool-use trajectory | 哪一步是 action？哪一步是 observation？ |
| SFT 数据构造 | 能把 trajectory 转成 step-level SFT | 输入输出格式是否稳定？是否可解析？ |
| Evaluator | 能设计 step-level 指标 | 模型错在工具、参数、schema 还是恢复？ |
| 错误诊断 | 能输出 error taxonomy 和 bad case | 主要错误类型是什么？下一步改数据还是改方法？ |
| DPO/ORPO 数据 | 能构造 chosen/rejected pair | rejected 为什么错？chosen 为什么更好？ |
| 业务迁移 | 能做一个 Mini-ToolBench | 我能否脱离 ToolBench 仓库复现方法？ |
| RL 准备 | 能判断何时进入 Search-R1/GRPO | 当前瓶颈是局部动作错误还是长程策略问题？ |

最终验收问题：

```text
如果给你一个新业务场景，
你是否能设计工具集合、构造 200 条 SFT 样本、建立 8 类错误指标、跑一次 baseline/SFT 对比，
并说明下一步是否需要 DPO 或 RL？
```

如果答案是“可以”，说明你已经从 Agent 半知半解状态进入初级 Agent 选手阶段。

---

# 10. 推荐学习节奏

## 10.1 每日学习节奏

建议每天固定输出一个小产物，而不是只阅读。

| 每日动作 | 时间建议 | 产物 |
|---|---:|---|
| 阅读/理解 | 30%-40% | notes |
| 样本/代码分析 | 30%-40% | jsonl / 脚本 |
| 复盘总结 | 20%-30% | review.md |

## 10.2 每阶段复盘模板

每个阶段结束写一份：

```text
1. 本阶段核心目标是否完成？
2. 我新增理解了什么？
3. 当前最大的卡点是什么？
4. 这个卡点属于概念、工程、数据、训练还是评估？
5. 下一阶段开始前是否需要补课？
6. 本阶段产出文件是否齐全？
7. 我是否满足阶段达成指标？
```

---

# 11. 不建议做的事

| 不建议 | 原因 |
|---|---|
| 一开始完整复现 ToolLLaMA | 资源重、变量多，不利于学习本质 |
| 一开始接真实 RapidAPI | API 不稳定，会干扰训练认知 |
| 只看最终 task success | 无法定位失败步骤 |
| 没有 evaluator 就 SFT | 训练结果无法归因 |
| 直接拿通用 DPO 数据 | 和 tool-use action 错误不匹配 |
| 一开始上 GRPO/PPO | reward、rollout、环境稳定性都没准备好 |
| 只学 ToolBench 不迁移 | 无法形成自己的 Agent 训练能力 |

---

# 12. 最终学习路线压缩版

```text
阶段 0：明确 ToolBench 学习边界
阶段 1：拆解项目结构，建立 Agent 抽象映射
阶段 2：手工分析 G1/G2/G3 trajectory
阶段 3：跑通最小 action→executor→observation 闭环
阶段 4：建立 step-level evaluator 与 error taxonomy
阶段 5：构造 SFT 数据并做小规模 LoRA 训练
阶段 6：从失败轨迹构造 DPO/ORPO pair
阶段 7：迁移成自己的 Mini-ToolBench
```

这条路线完成后，你不一定是高级 Agent 研究者，但你应该已经具备初级 Agent 选手最关键的能力：

```text
能建环境，能做数据，能训 baseline，能评估，能诊断，能迁移。
```

---

# 13. 参考资料

1. OpenBMB ToolBench GitHub：`https://github.com/OpenBMB/ToolBench`
2. ToolLLM ICLR 2024：`https://proceedings.iclr.cc/paper_files/paper/2024/hash/28e50ee5b72e90b50e7196fde8ea260e-Abstract-Conference.html`
3. ToolLLM arXiv：`https://arxiv.org/abs/2307.16789`
4. ToolPrefer / ToolPreference 方向：`https://arxiv.org/abs/2406.07115`
5. StableToolBench：`https://arxiv.org/abs/2403.07714`
