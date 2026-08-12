# ToolBench 学习范围与目标说明

## 1. 我为什么学习 ToolBench？

我学习 ToolBench，不是因为它包含大量 API，也不是为了完整复现 ToolLLaMA，而是因为它较完整地呈现了一个 Tool Agent 从数据构造、模型训练、环境执行到效果评估的基本闭环。

当前我对 Agent 的理解仍主要停留在概念和局部工程层面，例如知道 Agent 包含工具调用、规划、反馈和记忆，但还没有完整掌握以下问题：

* Agent 的状态、动作和环境反馈应当如何定义；
* 工具调用任务如何转化为可训练数据；
* 一条多步工具调用轨迹如何形成；
* SFT 模型实际学习的是格式、工具选择、参数填充，还是完整策略；
* 模型调用工具失败后，应如何定位具体错误；
* SFT、DPO 和 RL 分别适合解决哪些 Agent 问题。

ToolBench 同时包含工具定义、任务数据、工具调用轨迹、数据预处理、模型训练、工具执行、API Retriever 和 ToolEval，因此可以作为我学习 Tool Agent 设计与训练的完整样板。

通过学习 ToolBench，我希望把 Agent 从一个模糊的“大模型调用工具”概念，转化为一个明确的问题：

```text
Agent 根据当前 State 选择 Action，
环境执行 Action 并返回 Observation，
Agent 根据新的 State 继续决策，
直到完成任务或停止。
```

因此，我学习 ToolBench 的核心目的不是掌握某个仓库的使用方式，而是建立以下基础能力：

1. 能够用 State、Action、Observation、Feedback 和 Evaluator 描述 Agent；
2. 能够理解工具调用轨迹如何构造；
3. 能够构造 Tool Agent 的 SFT 和 DPO 数据；
4. 能够建立 step-level evaluator；
5. 能够根据 bad case 判断问题来自模型、数据、工具 schema、Retriever 还是环境；
6. 能够将这些方法迁移到自己的业务 Agent 中。

一句话概括：

```text
我学习 ToolBench，是为了借助一个相对完整的开源 Tool Agent 项目，
系统掌握 Agent 的环境设计、轨迹数据、训练方法和评估诊断方法。
```

---

## 2. ToolBench 对应 Agent 训练闭环中的哪些环节？

ToolBench 基本覆盖了 Tool Agent 从动作空间定义到模型评估的主要环节。

### 2.1 工具与动作空间定义

对应内容：

* API Collection；
* `toolenv`；
* API 文档；
* 参数 schema；
* 工具返回示例；
* 工具执行代码。

这一部分对应 Agent 的动作空间设计：

```text
Agent 能调用哪些工具？
每个工具需要哪些参数？
工具返回什么结果？
工具失败时返回什么错误？
```

需要从这一部分学习：

* 如何定义工具名称和工具描述；
* 如何区分相似工具；
* 如何设计 required 和 optional 参数；
* 如何统一工具正常返回和错误返回；
* 如何让工具调用结果可以被模型继续使用。

---

### 2.2 用户任务与训练任务构造

对应内容：

* Instruction Generation；
* G1、G2、G3 任务划分；
* 单工具和多工具任务。

这一部分对应 Agent 训练任务的构造：

```text
什么样的问题需要调用工具？
问题需要调用一个工具还是多个工具？
工具之间是否存在先后依赖？
任务成功是否可以验证？
```

需要从这一部分学习：

* 如何构造必须依赖工具才能完成的任务；
* 如何从单工具任务逐渐增加到多工具任务；
* 如何让任务中的参数能够从用户输入或历史 observation 中获得；
* 如何构造具备明确成功标准的 Agent 任务。

---

### 2.3 工具调用轨迹构造

对应内容：

* Solution Path Annotation；
* DFSDT；
* reasoning trace；
* tool execution；
* tool observation。

这一部分对应 Agent 的 demonstration trajectory：

```text
State₁ → Action₁ → Observation₁
→ State₂ → Action₂ → Observation₂
→ Final Answer
```

需要从这一部分学习：

* 一条成功轨迹如何形成；
* 中间 observation 如何影响下一步 action；
* 多步工具调用如何记录；
* 成功路径和失败路径有什么区别；
* 长轨迹如何拆成 step-level 训练样本。

---

### 2.4 SFT 数据与基础策略训练

对应内容：

* `toolllama_G123_dfs_train.json`；
* 数据预处理脚本；
* ToolLLaMA full fine-tuning；
* ToolLLaMA LoRA training。

这一部分对应 Agent 的基础策略学习：

```text
给定当前 State，学习输出正确的下一步 Action。
```

SFT 主要用于学习：

* 合法的工具调用格式；
* 工具选择模式；
* 参数生成模式；
* observation 后的下一步调用；
* 短程成功轨迹；
* 最终结果整合。

我需要重点学习的是 ToolBench 如何把原始 trajectory 转成模型可以学习的数据，而不是完全照搬 ToolLLaMA 的训练参数。

---

### 2.5 Agent 推理与环境交互

对应内容：

* `toolbench/inference`；
* action parser；
* API executor；
* observation 回传；
* close-domain inference；
* open-domain inference；
* DFSDT inference。

这一部分对应真正的 Agent 执行闭环：

```text
模型生成 Action
→ 系统解析 Action
→ 工具执行
→ 返回 Observation
→ 模型继续决策或结束
```

需要从这一部分学习：

* 如何解析模型输出；
* 如何校验工具名称和参数；
* 如何统一执行工具；
* 如何将 observation 加入下一步输入；
* 如何限制最大调用步数；
* 如何处理工具错误、空结果和重试；
* 如何判断什么时候停止。

---

### 2.6 工具检索

对应内容：

* API Retriever；
* retrieval 数据；
* Retriever training；
* open-domain inference。

这一部分解决的是大规模工具环境中的候选召回问题：

```text
用户任务
→ Retriever 召回 Top-K 工具
→ Tool Agent 在候选工具中选择
```

需要从这一部分理解：

* Retrieval Error 和 Policy Error 的区别；
* 为什么正确工具未召回时，后续模型无法完成任务；
* 为什么工具数量增加后不能把全部 schema 放进上下文；
* Retriever 和 Tool Agent policy 是两个不同模块。

---

### 2.7 Agent 评估与错误诊断

对应内容：

* ToolEval；
* Pass Rate；
* Preference / Win Rate；
* 自动 evaluator。

ToolEval 主要解决任务级成功和模型间比较，但我的学习还需要进一步增加 step-level evaluator，包括：

* Schema Valid Rate；
* Tool Selection Accuracy；
* Argument Exact Match；
* Slot-level F1；
* Execution Success Rate；
* Step Order Accuracy；
* Recovery Success Rate；
* Observation Misread Rate；
* Premature Stop Rate；
* Hallucinated Answer Rate。

这一部分对应：

```text
Task-level 指标回答“任务是否完成”；
Step-level 指标回答“为什么没有完成”。
```

---

### 2.8 DPO 与后续 RL 的数据基础

ToolBench 原始主线主要是成功轨迹 SFT，但其 DFSDT 搜索过程也会产生失败分支。

这些数据可以进一步转化为：

```text
正确工具 vs 错误工具
正确参数 vs 错误参数
正确恢复 vs 错误恢复
合理停止 vs 过早停止
```

从而用于构造 DPO preference pair。

更复杂的多步搜索、路径规划、停止策略和长期收益问题，可以作为后续 RL / GRPO 的研究入口。

因此，ToolBench 在我的 Agent 训练闭环中对应：

```text
工具定义
→ 任务构造
→ 轨迹标注
→ SFT 数据
→ 模型训练
→ 环境执行
→ 工具检索
→ 自动评估
→ 错误诊断
→ DPO / RL 扩展
```

---

## 3. 我暂时不学习哪些内容？

当前阶段的目标是从 Agent 半知半解状态成长为能够独立完成小型 Tool Agent 训练闭环的初级选手，而不是完整复现 ToolBench 论文结果。

因此，我暂时不把以下内容作为主要学习目标。

### 3.1 暂不完整复现 ToolLLaMA

暂时不追求：

* 完整下载和处理全部 ToolBench 数据；
* 完整训练 ToolLLaMA；
* 复现论文全部实验；
* 达到官方 leaderboard 指标；
* 使用与论文完全一致的模型、显卡和训练参数。

原因是完整复现资源成本较高，而且不能保证我真正理解 Agent 数据、环境和评估方法。

当前更重要的是：

```text
抽取小规模数据
→ 看懂 trajectory
→ 转换为自己的训练格式
→ 完成一次小规模 LoRA SFT
→ 建立 evaluator
```

---

### 3.2 暂不接入全部真实 RapidAPI

真实 API 可能存在：

* 下线；
* 限流；
* 超时；
* 参数变化；
* 返回格式变化；
* 权限问题。

这些环境噪声会干扰我对模型能力的判断。

因此，初期优先使用：

```text
固定工具集合
+ local mock executor
+ 可复现 observation
+ 可注入错误
```

等掌握基础闭环后，再考虑真实 API。

---

### 3.3 暂不从 Open-domain 开始

当前先学习 close-domain Tool Agent，即模型直接获得固定候选工具集合。

暂时不优先学习：

* 数千工具的大规模检索；
* 复杂 API Retriever 训练；
* 大规模 open-domain tool routing；
* Retriever 与 Tool Policy 联合优化。

原因是 open-domain 会同时引入 Retriever、Tool Policy、Schema、Executor 等多个变量，不利于初学阶段定位问题。

正确顺序是：

```text
先验证模型会不会在固定候选工具中正确调用
→ 再研究如何从大规模工具库中召回候选工具
```

---

### 3.4 暂不深入复现完整 DFSDT 搜索

当前需要理解 DFSDT 的思想和数据价值，但不要求完整复现其所有搜索逻辑和推理配置。

本阶段重点是：

* 理解多分支搜索；
* 理解成功路径和失败路径；
* 理解失败分支如何转化为 preference pair；
* 理解回退和恢复的作用。

暂时不追求：

* 完整树搜索性能；
* 大规模并行搜索；
* 搜索预算优化；
* 复杂路径剪枝策略。

---

### 3.5 暂不直接进入 PPO / GRPO

在以下条件未满足前，不进入大规模 Agent RL：

* evaluator 不稳定；
* error taxonomy 尚未建立；
* sandbox 尚不可控；
* reward 无法可靠计算；
* rollout 无法复现；
* 当前主要问题仍是 JSON、工具选择或参数填充；
* 尚未跑通 SFT 和 DPO。

当前训练顺序保持为：

```text
Prompt Baseline
→ SFT
→ DPO / ORPO
→ Error Diagnosis
→ 再判断是否进入 RL
```

---

### 3.6 暂不学习复杂 Agent 架构

当前不把以下内容作为主要目标：

* 多 Agent 协同；
* 长期记忆系统；
* 通用开放世界 Agent；
* 生产级浏览器 Agent；
* 复杂规划器；
* 大规模异步工具调度；
* 生产级高并发服务架构。

这些方向可以在完成 Mini-ToolBench 后逐步扩展。

---

## 4. 我最终要迁移出什么？

学习 ToolBench 的最终目标不是保留一个 ToolBench 仓库，也不是只得到一个 ToolLLaMA checkpoint，而是迁移出一套属于自己的 Agent 设计与训练方法。

最终需要迁移出以下五类成果。

### 4.1 一套 Agent 问题建模方法

面对一个新业务任务时，我能够明确写出：

* State 是什么；
* Action Space 是什么；
* Observation 是什么；
* Feedback 是什么；
* 成功和失败标准是什么；
* Evaluator 应包含哪些指标。

例如，在业务 Agent 中：

```text
State：
案情文本、已抽取事实、已检索法条、当前判断状态、历史工具结果

Action：
抽取事实、检索规则、打开法条、判断案件类型、推荐下一问、停止

Observation：
抽取结果、法条片段、规则匹配结果、错误信息

Feedback：
工具调用是否正确、法条是否召回、结论是否正确、轨迹是否合理
```

---

### 4.2 一个自己的 Mini-ToolBench

至少包含：

```text
mini_toolbench/
├── env/
│   ├── schemas.py
│   ├── tools.py
│   └── executor.py
├── data/
│   ├── sft_train.jsonl
│   ├── dpo_train.jsonl
│   └── eval.jsonl
├── train/
│   ├── sft_config.yaml
│   └── dpo_config.yaml
├── eval/
│   ├── metrics.py
│   ├── error_taxonomy.py
│   └── run_eval.py
├── logs/
└── reports/
```

这个项目应该脱离 ToolBench 原仓库也能独立运行。

---

### 4.3 一套 Tool Agent 数据构造方法

我应能够自行构造：

* 单工具任务；
* 多工具任务；
* 参数补全任务；
* 失败恢复任务；
* 成功 trajectory；
* step-level SFT 数据；
* chosen/rejected DPO pair；
* eval gold label。

需要完成的迁移不是复制 ToolBench 数据，而是掌握：

```text
业务任务
→ 工具 schema
→ 用户 instruction
→ 成功路径
→ 失败路径
→ SFT / DPO / Eval 数据
```

---

### 4.4 一套 Agent 评估和错误诊断体系

最终应形成自己的 evaluator，至少包括：

* Tool Selection Accuracy；
* Argument Exact Match；
* Slot-level F1；
* Schema Valid Rate；
* Execution Success Rate；
* Task Success Rate；
* Recovery Success Rate；
* Observation Misread Rate；
* Hallucinated Answer Rate；
* Step Count / Cost。

同时形成 error taxonomy，例如：

```text
wrong tool
missing argument
wrong argument value
invalid schema
wrong step order
observation misread
no recovery
premature stop
redundant tool call
hallucinated final answer
```

---

### 4.5 一套从 SFT 到 DPO 的最小训练闭环

最终应跑通：

```text
Prompt Baseline
→ Tool-use SFT
→ Rollout
→ Error Analysis
→ DPO Pair Construction
→ DPO / ORPO
→ SFT 与 DPO 效果对比
```

并能够解释：

* SFT 提升了什么；
* DPO 改善了什么；
* 哪些错误仍然没有解决；
* 剩余问题属于数据、环境、Retriever、Policy，还是长程策略；
* 是否有必要进一步进入 Search Agent 或 GRPO。

---

### 4.6 最终业务迁移方向

结合现有业务，最终可以迁移成一个业务版 Tool Agent：

```text
案情输入
→ extract_case_facts
→ search_legal_rules
→ classify_case
→ check_upgrade_conditions
→ recommend_next_question
→ 输出有依据的判断
```

ToolBench 提供的是通用方法模板，最终目标是把该模板迁移到：

* 法条检索；
* 案件判断；
* 问话推荐；
* 笔录审查；
* 案件升格审查；
* 业务知识驱动的 Agent 决策。

一句话概括：

```text
我要迁移出的不是 ToolLLaMA，
而是一套可以服务自己业务的 Agent 环境、数据、训练和评估方法。
```

---

## 5. 我如何判断自己完成了 ToolBench 学习？

不能通过“看完论文”“跑通官方命令”或“训练完成一个 checkpoint”判断学习完成。

应从概念、数据、工程、训练、评估和迁移六个层面验收。

### 5.1 概念层验收

我能够独立解释：

* ToolBench 为什么不只是数据集；
* Tool Agent 和普通问答模型的区别；
* State、Action、Observation、Feedback、Evaluator；
* Tool Schema 为什么属于结构化动作空间；
* Retriever 与 Tool Policy 的职责边界；
* Task-level 与 Step-level 评估的区别；
* SFT、DPO 和 RL 各自适合解决什么问题。

合格标准：

```text
能够不看资料，用 10 分钟完整讲清 ToolBench 的 Agent 训练闭环。
```

---

### 5.2 数据层验收

我能够：

* 区分 G1、G2、G3；
* 手工拆解至少 30 条 trajectory；
* 指出每一步 State、Action 和 Observation；
* 把长 trajectory 拆成 step-level SFT 数据；
* 判断成功路径适合 SFT，失败分支如何构造 DPO pair；
* 识别自动合成数据的质量风险。

合格标准：

```text
看到一条 ToolBench 样本，
能够解释其完整调用链、潜在错误和训练用途。
```

---

### 5.3 工程层验收

我能够独立实现：

* 至少 5 个 tool schema；
* 一个统一 executor；
* 一个 action parser；
* 标准化 success/error observation；
* 一个多步 Agent 执行循环；
* 完整 step-level 日志；
* 至少 6 类可注入错误。

合格标准：

```text
不依赖 ToolBench 全量环境，
能独立跑通一个 Action → Executor → Observation → Next Action 闭环。
```

---

### 5.4 训练层验收

我能够：

* 构造至少 200—800 条 Tool-use SFT 数据；
* 使用 Qwen 小模型完成一次 LoRA SFT；
* 对比 Prompt Baseline 和 SFT；
* 收集 SFT 模型失败轨迹；
* 构造至少 100 条 DPO pair；
* 完成一次小规模 DPO 或 ORPO 实验。

合格标准：

```text
能够说明 SFT 和 DPO 分别改善了哪类 Agent 能力，
而不只是报告 loss。
```

---

### 5.5 评估层验收

我能够实现并运行：

* Schema Valid Rate；
* Tool Selection Accuracy；
* Argument Exact Match；
* Slot-level F1；
* Execution Success Rate；
* Task Success Rate；
* Recovery Success Rate；
* 至少 8 类 error taxonomy；
* 自动 bad case 输出。

合格标准：

```text
面对一次任务失败，
能够定位它属于模型、schema、Retriever、Executor、Observation 还是 Synthesis 问题。
```

---

### 5.6 迁移层验收

我能够脱离 ToolBench 原始仓库，完成一个自己的 Mini-ToolBench，包括：

* 自定义业务工具；
* 自定义 schema；
* 本地 executor；
* SFT 数据；
* DPO 数据；
* Eval 数据；
* baseline；
* step-level evaluator；
* bad case report；
* SFT / DPO 前后对比报告。

合格标准：

```text
给出一个新的业务场景后，
我能够自行设计工具、数据、训练流程和评估体系。
```

---

## 6. 最终完成标准

当我能够完成以下完整闭环时，可以认为已经完成 ToolBench 入门学习：

```text
定义一个 Tool Agent 任务
→ 设计 5 个左右的工具和 schema
→ 构造可控 executor
→ 构造 SFT 和 Eval 数据
→ 跑 Prompt Baseline
→ 建立 Step-level Evaluator
→ 进行 LoRA SFT
→ 收集失败轨迹
→ 构造 DPO Pair
→ 分析 Baseline / SFT / DPO 错误变化
→ 迁移到自己的业务场景
```

最终不以“是否复现 ToolLLaMA 官方成绩”为标准，而以是否具备以下能力为标准：

```text
能建模
能搭环境
能做轨迹数据
能训练
能评估
能诊断
能迁移
```

完成这些内容后，我应当从“知道 Agent 的一些概念”成长为：

```text
能够独立完成一个小型 Tool Agent
设计、训练、评估和错误分析闭环的初级 Agent 选手。
```
