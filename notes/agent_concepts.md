# Agent 概念辨析与工程建模文档

> 版本：v1.0  
> 日期：2026-07-09  
> 适用对象：面向具备 LLM、RAG、工具调用、训练与评估经验的算法/工程研发人员  
> 文档定位：本文件整理自本轮对话中关于 Agent、LLM-Driven Workflow、Skill、Self-RAG、Policy、Planning 的辨析，并补充为一份可复用的方法论文档。

---

## 0. 核心结论

Agent 不应被理解成某个固定算法、固定框架或固定组件集合。它更准确地说是一种系统设计范式：

> Agent 是一种让大模型围绕目标进行“感知 — 决策 — 行动 — 观察 — 状态更新 — 再决策”的闭环问题解决系统。

因此，Agent 的关键不是“模型会不会判断下一步”，而是：

```text
State
  ↓
Policy 根据 State 在 Action Space 中选择 Action
  ↓
Environment / Tool 执行 Action
  ↓
Observation 返回
  ↓
State Update
  ↓
Continue / Stop / Final Output
```

这一区分非常重要。很多系统都让 LLM 判断下一步，例如 LLM-Driven Workflow、Skill、Self-RAG，但它们不一定构成完整 Agent。是否更接近 Agent，要看该决策是否进入完整的状态—动作—观察闭环，并持续影响后续状态与动作选择。

---

## 1. 为什么 Agent 容易让人混乱

Agent 这个词在工业界和学术界常被同时用于描述不同层级的东西：

| 视角 | 常见说法 | 实际含义 |
|---|---|---|
| 系统范式 | Agent / Autonomous Agent | 目标驱动的闭环执行系统 |
| 自主性 | lightweight agent / full agent / heavy agent | 模型能自主决定多少事情 |
| 控制循环 | ReAct / Plan-and-Execute / Reflection | 思考、行动、观察、修正如何组织 |
| 架构组件 | tools / memory / planner / verifier | 系统由哪些模块构成 |
| 能力封装 | skill / xxx.md / prompt module | 任务经验如何沉淀复用 |
| 产品形态 | coding agent / research agent / deep agent | 面向具体场景的系统包装 |

所以，Agent 不是一个单点概念，而是一个连续谱：

```text
普通 LLM 调用
  ↓
LLM-Driven Workflow
  ↓
Agentic Workflow
  ↓
Tool-Using Agent
  ↓
Planning Agent
  ↓
Long-Horizon / Deep Agent
```

判断一个系统是否“像 Agent”，不应只看它有没有工具、有没有记忆、有没有 planning，而应看它在状态、动作、观察、反馈、评估上的闭环程度。

---

## 2. 用“分布式系统”类比 Agent

Agent 更像“分布式系统”这样的系统范式，而不是某个固定算法。

分布式系统可以包含 RPC、消息队列、缓存、分片、一致性协议、服务发现、限流熔断、监控告警等，但不是每个分布式系统都必须配齐这些组件。一个简单微服务可以是分布式系统，一个复杂云原生平台也是分布式系统，只是复杂度不同。

Agent 也是类似的。

一个最小 Agent 可能只有：

```text
Goal -> LLM 判断 -> Tool Call -> Observation -> Final Answer
```

一个复杂 Agent 可能包含：

```text
Goal Understanding
  -> Planning
  -> Tool Selection
  -> Multi-step Execution
  -> Working Memory
  -> Retrieval
  -> Reflection
  -> Verifier
  -> Artifact Generation
  -> Error Recovery
```

所以可以这样定义：

> Agent 之于 LLM 应用，就像分布式系统之于后端工程。它不是单个组件，而是一套围绕目标完成、外部交互、状态演化和反馈修正组织系统能力的设计范式。

---

## 3. Agent 的四个核心维度

理解 Agent 时，建议从四个维度切入：

1. Autonomy：自主性有多强；
2. Control Loop：控制循环如何组织；
3. Architecture Components：系统由哪些部件构成；
4. Application / System Packaging：最后被封装成什么应用形态。

### 3.1 Autonomy：模型能自主决定多少事情

自主性不是二元的，而是连续的。

| 层级 | 特征 | 示例 |
|---|---|---|
| 低自主性 | 流程写死，模型只生成结果 | 固定抽取、固定检索、固定判断 |
| 中等自主性 | 模型在局部节点做路由或判断 | 是否检索、检索哪个方向 |
| 高自主性 | 模型根据状态持续选择工具和路径 | 多步研究、代码修复、法律分析 |
| 长期自治 | 跨任务、跨时间持续运行 | 定期监控、自动更新知识库 |

Autonomy 的核心来源是 Policy。Policy 越开放，模型越能根据当前状态自主选择动作，系统越接近强 Agent。

### 3.2 Control Loop：系统怎么跑

Agent 的控制循环常见形态包括：

#### 3.2.1 单轮调用

```text
Input -> LLM -> Output
```

这不是典型 Agent，因为没有外部动作与状态回流。

#### 3.2.2 固定 Workflow

```text
Step 1 -> Step 2 -> Step 3 -> Output
```

流程由代码控制，LLM 只承担局部任务。

#### 3.2.3 ReAct

```text
Thought -> Action -> Observation -> Thought -> Action -> Observation -> Final
```

ReAct 适合外部信息不确定、需要边查边判断的任务。其优点是灵活，缺点是容易局部贪心、工具调用发散、长任务漂移。

#### 3.2.4 Plan-and-Execute

```text
Planner 先生成步骤
  ↓
Executor 按步骤执行
```

适合结构较清晰的长任务，例如报告生成、系统排障、多文件分析。

#### 3.2.5 Plan-and-Execute ReAct

更成熟的方式是：

```text
上层：Plan-and-Execute 负责全局结构
下层：每个子任务内部用 ReAct 动态执行
```

这能兼顾全局规划和局部适应。

#### 3.2.6 Reflection / Verification Loop

```text
Generate -> Verify -> Find Defect -> Revise -> Verify -> Final
```

在法律、代码、报告、复杂问答等场景中，验证闭环比生成本身更关键。

### 3.3 Architecture Components：系统由哪些部件构成

常见组件包括：

| 组件 | 作用 | 是否必须 |
|---|---|---|
| LLM / Policy Model | 理解状态并选择动作 | 是 |
| Tool | 外部动作接口 | 取决于任务 |
| Working Memory / State | 保存当前任务状态 | 基本需要 |
| Long-term Memory | 跨会话保存经验或偏好 | 不一定 |
| Planner | 子目标组织与路径生成 | 复杂任务需要 |
| Executor | 执行动作、调用工具、返回结果 | 工具型任务需要 |
| Verifier / Critic | 检查结果质量和依据 | 高风险任务建议需要 |
| Router | 选择 skill、工具或子 agent | 多能力系统需要 |
| Skill Registry | 管理可复用任务能力 | 规模化系统需要 |
| Guardrails | 权限、格式、安全、合规约束 | 生产系统需要 |

关键点：Agent 不需要所有组件都存在。没有长期记忆，也可以是 Agent；没有复杂 planner，也可以是轻量 Agent。真正必要的是状态—动作—观察的闭环决策。

### 3.4 Application / System Packaging：系统如何产品化

同一个 Agent 范式可以被封装成不同应用：

| 形态 | 典型能力 |
|---|---|
| Coding Agent | 读写文件、运行测试、修复代码、提交 patch |
| Research Agent | 搜索、阅读、交叉验证、报告生成 |
| Legal Agent | 法条检索、案情要素抽取、构成要件比对、类案检索 |
| Data Analysis Agent | 查询数据、运行脚本、生成图表、解释指标 |
| DeepAgent | 泛指更复杂、更长链路、多工具、多阶段的 Agent 系统 |

DeepAgent 通常不是严格统一的学术术语，更像“复杂 Agent 系统”的产品化或能力级命名。

---

## 4. Agent、LLM-Driven Workflow、Skill、Self-RAG 的区别

这些概念都可能包含“让模型决定下一步”，但区别在于决策层级、动作空间、控制权和闭环程度。

### 4.1 总览表

| 概念 | 本质 | 模型决策位置 | 动作空间 | 控制权 | 是否一定是 Agent |
|---|---|---|---|---|---|
| LLM-Driven Workflow | LLM 驱动的流程系统 | 固定流程中的局部节点 | 有限分支/局部动作 | 主要在工程流程 | 不一定 |
| Skill | 任务能力说明/SOP | skill 说明中定义策略 | 本身无动作空间 | 取决于 runtime | 不是 |
| Self-RAG | 自反式检索增强生成方法 | 检索与生成过程中 | 检索、评价、生成 | 主要在检索生成机制 | 通常不是完整 Agent |
| Agent | 目标驱动闭环系统 | 全局任务状态上持续决策 | 工具、检索、代码、停止、重试等 | 模型 + runtime 共同控制 | 是 |

### 4.2 LLM-Driven Workflow：流程主导，模型参与

LLM-Driven Workflow 的典型结构是：

```text
固定主流程
  ↓
LLM 在某个节点做分类/路由/判断
  ↓
流程进入预设分支
```

例如刑事/非刑事判断：

```text
输入案情
  ↓
LLM 抽取事实
  ↓
LLM 判断是否需要检索
  ↓
if need_retrieval:
    检索法条
else:
    跳过检索
  ↓
LLM 输出结论
```

这里模型确实决定了一个分支，但整体控制权仍在 workflow。它更准确地叫：

```text
固定流程骨架 + LLM 局部决策节点
```

### 4.3 Skill：能力说明，不是运行系统

Skill 是一套面向特定任务的可复用方法说明，类似 SOP、任务手册或 domain playbook。

例如 `case_judge.md` 可以定义：

```text
适用场景：判断案情是刑事还是非刑事。
工作步骤：
1. 抽取核心事实。
2. 判断可能涉及的法益和案由方向。
3. 必要时检索法条。
4. 比对构成要件。
5. 输出结论、依据、不确定性。
禁止事项：
- 不得编造法条。
- 不得事实不足时强行定性。
- 不得把普通民事纠纷直接认定为刑事。
```

Skill 本身不维护状态、不执行工具、不接收 observation，也不判断是否继续循环。它可以被 Workflow 调用，也可以被 Agent 加载。

结论：

```text
Tool = 动作接口
Skill = 任务方法
Workflow = 流程编排
Agent = 闭环执行系统
```

### 4.4 Self-RAG：检索生成子过程上的 agentic 机制

Self-RAG 看起来像 Agent，是因为它也让模型判断：

```text
是否检索？
检索内容是否相关？
答案是否被证据支持？
是否继续生成？
```

如果只看检索生成子过程，它确实像一个 micro-agent：

```text
Generation State
  ↓
判断是否 Retrieve
  ↓
获得 Passage
  ↓
评价 Passage
  ↓
更新生成状态
  ↓
生成或继续检索
```

但从整体系统看，Self-RAG 通常只是 RAG / QA 流程中的一个 agentic retrieval module，而不是完整通用 Agent。

更准确的表述是：

```text
Self-RAG = agentic retrieval / reflective RAG mechanism
Agent = general state-action-observation control system
```

### 4.5 两种组合关系

#### Workflow 中嵌入 Agent

```text
固定业务流程
  ↓
某个节点调用一个小 Agent
  ↓
Agent 完成局部闭环
  ↓
结果返回 Workflow
```

例如案件审查 workflow 中嵌入一个 legal_judge_agent。

#### Agent 中嵌入 Workflow

```text
Agent 作为总控
  ↓
根据任务选择某个 Workflow
  ↓
执行 Workflow
  ↓
读取结果并决定下一步
```

例如 legal_agent 可以调用 case_judge_workflow、law_retrieval_workflow、question_recommendation_workflow。

---

## 5. Policy：Agent 的决策策略

Policy 是理解 Agent 的关键。

### 5.1 Policy 的定义

Policy 不是 Action。Policy 是：

```text
Policy = 根据当前 State 在 Action Space 中选择 Action 的决策规则 / 策略函数
```

形式化表示：

```text
π(a | s)
```

工程化表示：

```text
policy(state) -> action
```

例如：

```text
State：案情疑似诈骗，但非法占有目的和虚构事实不明确。
Policy：当前不能直接定性，应检索诈骗罪构成要件。
Action：search_law("诈骗罪 构成要件 非法占有目的 虚构事实")
Observation：返回相关法律依据。
State Update：补充法律依据，重新评估案情。
```

### 5.2 Policy 与其他概念的区别

| 概念 | 含义 | 与 Policy 的关系 |
|---|---|---|
| Action | 被实际执行的具体动作 | Policy 选择 Action |
| Action Space | 可选动作集合 | Policy 在其中选择 |
| Plan | 完成目标的路线图 | Policy 决定执行时如何走 |
| Skill | 任务 SOP | Skill 可约束或指导 Policy |
| Prompt | 指令文本 | Prompt 可承载 Policy，但不等于实际 Policy |
| Control Loop | 完整闭环结构 | Policy 是 loop 中的决策函数 |
| Autonomy | 自主性强弱 | Policy 的开放程度决定 Autonomy |

### 5.3 Policy 是否必须存在

如果一个系统要称为 Agent，广义上必须有 Policy。因为 Agent 的核心就是在状态下选择下一步动作。

但 Policy 不一定是训练出来的模型。它可以是：

| 类型 | 示例 |
|---|---|
| 规则 Policy | if confidence < 0.7: search_law |
| Workflow Policy | 固定流程 + 条件分支 |
| Prompt Policy | Prompt 中写明何时检索、何时停止 |
| LLM Policy | LLM 根据状态生成 tool call |
| Learned Policy | 通过 SFT/DPO/PPO/GRPO 优化后的动作选择行为 |

### 5.4 在法律判定任务中的 Policy

一个“刑事/非刑事判定 Agentic Workflow”至少包含几类 policy：

```text
Retrieval Policy：是否检索法条。
Tool Selection Policy：检索刑法、司法解释、类案还是民刑边界规则。
Stopping Policy：何时输出最终结论，何时输出信息不足。
Risk Policy：何时避免强判断，何时转人工复核。
Recovery Policy：检索为空、证据不相关、结论被 verifier 否定时如何处理。
```

---

## 6. Planning：不是只有链式和树状

### 6.1 Planning 的准确理解

Planning 不应被狭义理解为“把任务拆成若干步骤”。更准确地说：

> Planning 是围绕目标完成的一套子目标组织、路径选择、工具分配、反馈处理和动态重规划机制。

它服务于状态—动作—观察闭环。

### 6.2 Plan 与 Policy 的区别

```text
Plan = 路线图
Policy = 驾驶策略
```

例如：

```text
Plan：
1. 抽取案情事实
2. 识别可能案由
3. 检索法条
4. 比对构成要件
5. 输出结论

Policy：
如果事实已足够，则跳过类案检索；
如果检索为空，则改写 query；
如果证据不支持结论，则输出信息不足；
如果 verifier 失败，则重新分析。
```

### 6.3 Planning 的多种形态

| 规划形态 | 结构 | 适用任务 |
|---|---|---|
| 静态链式规划 | A -> B -> C | 流程清晰、路径确定的任务 |
| 动态链式规划 | 每步后再决定下一步 | ReAct、开放信息检索 |
| 树状多路径规划 | 多候选路径并行展开 | 高不确定、多方案比较 |
| 图结构/DAG 规划 | 多任务依赖、共享、汇聚 | 数据分析、复杂审查、代码任务 |
| 层级规划/HTN | 大目标 -> 子目标 -> 原子动作 | 复杂业务任务拆解 |
| 状态机式规划 | 状态与转移条件 | 强约束、失败恢复、合规流程 |
| 搜索式规划 | beam/MCTS/best-first | 数学、代码、多跳推理 |
| 反应式规划 | 根据 observation 即时反应 | 工具环境不确定任务 |
| 约束驱动规划 | 成本、权限、风险约束优先 | 法律、医疗、金融、政务 |
| 多智能体协同规划 | 多角色分工与仲裁 | 大型复杂任务 |

### 6.4 Closed-loop Planning

真正的 Agent planning 更强调闭环：

```text
Plan
  ↓
Execute
  ↓
Observe
  ↓
Feedback / Reflection
  ↓
Update Plan
  ↓
Continue / Stop
```

如果计划生成后不再根据观察结果调整，那只是 open-loop planning，更接近普通 workflow。Agent planning 的关键是能够根据工具结果、错误信息、空检索、测试失败、人类反馈或 verifier 反馈进行重规划。

---

## 7. 以“刑事/非刑事案件判断”为例

### 7.1 任务描述

输入一段案情文本，系统需要判断其属于刑事案件、非刑事案件，还是信息不足/边界不清。判断过程中，模型需要自主综合分析案情，并决定是否检索法条、司法解释或边界规则。

### 7.2 不推荐的极端方案

#### 纯分类器

```text
输入案情 -> LLM 直接输出刑事/非刑事
```

问题：边界案件容易误判，缺少依据，难以复核。

#### 固定全量检索 RAG

```text
输入案情 -> 固定检索法条 -> LLM 判断
```

问题：简单案件增加成本，检索噪声可能污染判断。

#### 开放式 Full Agent

```text
Agent 自主搜索所有材料、多轮扩展、长程分析
```

问题：法律任务高约束，开放动作空间容易发散，输出稳定性不足。

### 7.3 推荐方案：受限 Agentic Workflow

```text
输入案情
  ↓
案情事实抽取
  ↓
初步风险判断
  ↓
Retrieval Gate：是否需要检索？
  ├─ 否：直接输出结论 + 理由
  └─ 是：检索相关法条/司法解释/边界规则
          ↓
       证据相关性判断
          ↓
       构成要件比对
          ↓
       Verifier 审查
          ↓
       输出结论 / 信息不足 / 转人工复核
```

该方案不是开放式 full agent，而是一个：

```text
基于 Skill 的法律判定 Agentic Workflow
```

如果检索、重试、停止和 verifier 结果能回流到状态，并影响下一步动作，也可以称为：

```text
轻量法律判定 Agent
```

### 7.4 推荐输出类型

不要只做二分类。更稳的是四分类：

```text
1. 明显刑事风险
2. 疑似刑事，需进一步核实
3. 非刑事，但可能涉及民事/行政/治安责任
4. 信息不足，无法判断
```

### 7.5 示例状态结构

```json
{
  "case_facts": {
    "行为主体": "...",
    "行为对象": "...",
    "行为方式": "...",
    "金额或后果": "...",
    "主观目的": "...",
    "关键缺失事实": ["..."]
  },
  "preliminary_label": "疑似刑事",
  "confidence": 0.62,
  "need_retrieval": true,
  "retrieval_targets": ["诈骗罪构成要件", "民刑边界"],
  "retrieved_evidence": [],
  "risk_flags": ["非法占有目的不明确"],
  "next_action": "search_law"
}
```

### 7.6 示例 Action Space

```text
direct_judgment
search_law
search_judicial_interpretation
search_case_rule
ask_clarification
rewrite_query
verify_answer
final_answer
manual_review
```

### 7.7 示例停止条件

```text
事实和依据足以支撑结论 -> final_answer
关键事实缺失且无法通过检索弥补 -> insufficient_information
检索结果不相关且已重试达到上限 -> manual_review
verifier 判断依据不足 -> reanalyze 或 insufficient_information
```

---

## 8. Agent 训练与评估视角

Agent 训练不是单纯优化自然语言输出，而是在优化环境中的动作选择策略。

### 8.1 为什么需要 step-level evaluator

只看最终 task success 不能定位失败原因。Agent 可能最终答错，原因可能是：

```text
选错工具
参数填错
检索 query 不好
工具结果没读懂
过早停止
没有失败恢复
最终答案幻觉
```

所以需要 step-level evaluator。

### 8.2 常见指标

| 指标 | 说明 |
|---|---|
| Task Success Rate | 最终任务是否完成 |
| Tool Selection Accuracy | 工具是否选对 |
| Argument Exact Match | 参数是否正确 |
| Schema Valid Rate | tool call 是否可解析、可执行 |
| Execution Success Rate | executor 是否成功返回 |
| Retrieval Recall@k | 检索是否召回所需证据 |
| Evidence Coverage | 证据是否覆盖关键判断点 |
| Citation Correctness | 引用是否正确支撑结论 |
| Recovery Success Rate | 失败后能否修复 |
| Step Count / Cost / Latency | 成本、时延、调用步数 |

### 8.3 Error Taxonomy

建议至少记录：

```text
1. wrong tool
2. missing argument
3. wrong argument value
4. invalid schema
5. wrong step order
6. poor query formulation
7. retrieval miss
8. irrelevant evidence accepted
9. tool result misinterpretation
10. premature stopping
11. no recovery
12. hallucinated final answer
```

### 8.4 训练方法与 Agent 类型的匹配

| 场景 | 推荐训练路径 | 主要优化目标 |
|---|---|---|
| 工具编排型 Agent | SFT -> DPO | 工具选择、参数格式、失败恢复 |
| 主动检索型 Agent | SFT -> PPO/GRPO -> DPO | query reformulation、stopping policy、证据聚合 |
| 代码执行型 Agent | SFT -> DPO + PPO/GRPO | patch 质量、测试通过、错误恢复 |

SFT 更适合学习成功轨迹和格式；DPO 适合压制错误工具选择和错误参数；PPO/GRPO 更适合长程策略优化，例如主动检索中的 query 改写、停止策略和证据聚合路径。

---

## 9. 工程落地判断框架

做方案时，不要先说“我要做 Agent”。应该先回答以下问题：

```text
1. Goal：目标是什么？
2. State：系统当前需要保存哪些状态？
3. Action Space：模型允许选择哪些动作？
4. Policy：动作选择由规则、workflow、prompt、LLM 还是训练模型决定？
5. Observation：动作执行后会返回什么结果？
6. State Update：观察结果如何更新状态？
7. Stop Condition：何时停止、重试、转人工或输出最终答案？
8. Evaluator：如何评估每一步和最终任务？
9. Guardrails：权限、成本、合规、格式和安全边界是什么？
```

### 9.1 选择普通 Workflow

适合：

```text
流程稳定
动作空间固定
不需要模型自主选择工具
错误风险高且必须强控制
```

### 9.2 选择 LLM-Driven Workflow

适合：

```text
主流程固定
局部需要 LLM 判断、路由、抽取或打分
需要可控但有一定灵活性
```

### 9.3 选择 Agentic Workflow

适合：

```text
部分节点需要模型决定是否检索、是否重试、是否停止
工具范围有限
业务要求稳定可控
```

### 9.4 选择 Full / Heavy Agent

适合：

```text
任务长
动作空间大
工具多
需要动态规划、重规划、失败恢复、长程状态管理
可以接受更高成本和更复杂评估
```

---

## 10. 常见误区

### 误区一：有工具调用就是 Agent

不一定。固定 workflow 中也可以调用工具。关键是模型是否基于状态自主选择动作，并根据 observation 继续决策。

### 误区二：没有长期记忆就不是 Agent

不对。长期记忆不是 Agent 的必要条件。很多单任务 Agent 只需要 working memory。

### 误区三：Skill 就是 Agent

不对。Skill 是任务 SOP，不是运行时闭环系统。

### 误区四：Self-RAG 就是完整 Agent

不准确。Self-RAG 可以看成 retrieval-generation 子过程上的 agentic mechanism，但通常不是完整任务级 Agent。

### 误区五：Planning 只有链式和树状

不对。链式和树状只是两种路径结构。完整 planning 还包括动态链式、DAG、状态机、层级规划、搜索式规划、反应式规划、约束驱动规划和多智能体协同规划。

### 误区六：Plan 和 Policy 是一回事

不对。Plan 是路线图，Policy 是基于当前状态选择下一步动作的策略函数。

### 误区七：Prompt 写了规则，就等于模型真的具备该 Policy

不对。Prompt 只是 policy 的一种表达方式。真实 policy 要通过执行 trace、step-level evaluator 和 bad case analysis 验证。

---

## 11. 推荐术语口径

在项目沟通中，建议避免笼统说“做 Agent”。可以改成更具体的表述。

### 11.1 法律判定场景

```text
本方案不是开放式通用 Agent，而是一个受限的 Agentic Workflow。
在固定主流程下，引入模型的动态决策能力，使其能够基于案情复杂度自主判断是否检索法条，并在检索结果返回后完成构成要件比对、结果验证和结论生成。
```

### 11.2 技术训练场景

```text
本阶段重点不是构建通用开放世界 Agent，而是先搭建动作空间清晰、反馈可控、可评估可诊断的工具编排型 Agent sandbox，并围绕 tool selection、argument filling、schema valid rate、recovery success rate 建立 SFT/DPO 训练闭环。
```

### 11.3 主动检索场景

```text
主动检索型 Agent 的关键不是会调用 search 工具，而是能学习何时检索、如何构造 query、何时 reformulate、何时停止，以及如何基于多证据综合生成答案。
```

---

## 12. 一句话总括

Agent 是目标驱动的 LLM 闭环执行系统范式。它通过 Policy 在 State 上选择 Action，通过 Tool / Environment 获得 Observation，再更新 State 并持续决策。Workflow、Skill、Self-RAG 都可以包含 agentic decision，但只有当这种决策进入完整状态—动作—观察闭环，并持续影响后续动作与任务完成时，才更接近严格意义上的 Agent。

---

## 13. 最小工程模板

最后给出一个最小实现骨架，便于后续落地：

```python
class AgentState:
    def __init__(self, goal, context):
        self.goal = goal
        self.context = context
        self.history = []
        self.observations = []
        self.done = False
        self.final_answer = None


def policy(state):
    """根据当前 state 选择下一步 action。"""
    # 可以是规则、LLM、prompt、router 或训练后的模型
    raise NotImplementedError


def execute(action):
    """调用工具或环境，返回 observation。"""
    raise NotImplementedError


def update_state(state, action, observation):
    """把 action 和 observation 写回状态。"""
    state.history.append(action)
    state.observations.append(observation)
    return state


def should_stop(state):
    """判断是否完成、失败、转人工或继续。"""
    return state.done


def run_agent(goal, context):
    state = AgentState(goal, context)
    while not should_stop(state):
        action = policy(state)
        observation = execute(action)
        state = update_state(state, action, observation)
    return state.final_answer
```

这个模板体现的不是某个特定框架，而是 Agent 的基本抽象：目标、状态、策略、动作、观察、状态更新和停止条件。

