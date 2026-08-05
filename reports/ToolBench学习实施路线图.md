# ToolBench学习实施路线图

## 一、整体时间规划

### 总周期：3-4周（专项学习）
**对应8周Agent训练路线图中的Week 2-5**

| 周次 | 阶段 | 重点内容 | 每日时间投入 |
|------|------|----------|------------|
| 第1周 | 阶段0-2 | 基础认知+项目结构+数据分析 | 4-6小时/天 |
| 第2周 | 阶段3-4 | 最小闭环+评估体系 | 5-7小时/天 |
| 第3周 | 阶段5-6 | SFT训练+DPO数据构造 | 6-8小时/天 |
| 第4周 | 阶段7 | 业务迁移+Mini-ToolBench | 6-8小时/天 |

## 二、详细阶段实施计划

### 阶段0：学习边界与基础认知（0.5-1天）

#### 任务清单
1. ✅ 建立本地学习目录结构
2. ✅ Clone ToolBench项目
3. ✅ 阅读关键材料：
   - README.md（英文+中文）
   - 论文摘要与方法部分
   - 数据构造与评估说明
4. ✅ 创建学习边界文档
5. ✅ 建立术语卡片

#### 产出文件
```
notes/
├── toolbench_learning_scope.md
├── toolbench_glossary.md
└── stage0_review.md
```

#### 关键问题回答
1. **我为什么学习ToolBench？**
   - 学习工具调用Agent的完整训练流程
   - 掌握轨迹数据构造与评估方法
   - 为业务Agent开发打下基础

2. **ToolBench对应Agent训练闭环的哪些环节？**
   - 数据构造：instruction + trajectory
   - 环境定义：tool schema + executor
   - 策略训练：SFT + DPO
   - 评估诊断：step-level evaluator
   - 服务部署：inference pipeline

3. **我暂时不学习哪些内容？**
   - 完整复现ToolLLaMA训练
   - RapidAPI真实环境对接
   - 大规模分布式训练
   - Web UI深度定制

### 阶段1：项目结构与Agent抽象对齐（1-2天）

#### 任务清单
1. 扫描项目目录结构
2. 建立目录功能映射表
3. 绘制端到端流程图
4. 创建Agent抽象映射表
5. 编写阶段复盘

#### 产出文件
```
notes/
├── toolbench_tree_l3.txt
├── toolbench_pipeline.md
├── toolbench_agent_mapping.md
└── stage1_project_structure_review.md
```

#### 关键理解点
1. **ToolBench训练入口**：`toolbench/train/train.py`
2. **ToolBench推理入口**：`toolbench/inference/qa_pipeline.py`
3. **ToolBench评估入口**：`toolbench/tooleval/eval_pass_rate.py`
4. **环境交互机制**：state → action → observation循环
5. **自定义修改点**：tool schema、executor、数据格式

### 阶段2：数据格式与轨迹结构分析（2-3天）

#### 任务清单
1. 抽样G1/G2/G3数据（各10+条）
2. 手工标注轨迹结构
3. 转换为State-Action-Observation格式
4. 总结G1/G2/G3能力差异
5. 建立数据质量问题清单

#### 产出文件
```
data_samples/
├── g1_10.jsonl
├── g2_10.jsonl
├── g3_10.jsonl
└── state_action_observation_examples.jsonl

notes/
├── toolbench_data_analysis.md
├── g1_g2_g3_complexity_analysis.md
└── toolbench_data_quality_notes.md
```

#### 样本分析要点
| 分析项 | 具体问题 |
|--------|----------|
| user instruction | 用户到底想完成什么 |
| required tools | 需要哪些工具 |
| first action | 第一步为什么调用这个工具 |
| arguments | 参数来源（instruction/history） |
| observation | 工具返回了什么 |
| next action | observation如何影响下一步 |
| final answer | 最终答案是否基于工具结果 |
| possible failure | 这条轨迹可能错在哪里 |

### 阶段3：最小工具调用闭环（3-5天）

#### 任务清单
1. 选择close-domain模式（5-10个工具）
2. 实现统一action parser
3. 实现mock executor
4. 运行50+条最小轨迹
5. 主动构造6类失败场景

#### 产出文件
```
scripts/
├── action_parser.py
├── mock_executor.py
└── run_minimal_tool_agent.py

logs/
├── minimal_rollout_50.jsonl
└── failure_injection_cases.jsonl

notes/
└── stage3_minimal_agent_review.md
```

#### 失败场景构造
| 错误类型 | 构造方式 |
|----------|----------|
| wrong tool | 替换成相似但不适用工具 |
| missing argument | 删除必填参数 |
| wrong argument value | 修改参数值 |
| invalid schema | 输出非JSON或字段错误 |
| empty result | executor返回空结果 |
| no recovery | 失败后胡编最终答案 |

### 阶段4：Evaluator与错误分类体系（4-7天）

#### 任务清单
1. 定义gold label格式
2. 实现step-level metrics（8+个指标）
3. 建立错误分类体系（10+类错误）
4. 实现自动bad case输出
5. 建立人工复核模板

#### 产出文件
```
eval/
├── metrics.py
├── error_taxonomy.py
└── run_eval.py

reports/
├── eval_summary.md
├── bad_cases.jsonl
├── error_distribution.csv
└── manual_review_template.csv

notes/
└── stage4_evaluator_design.md
```

#### 核心评估指标
1. Schema Valid Rate：格式可解析性
2. Tool Selection Accuracy：工具选择准确率
3. Argument Exact Match：参数完全匹配率
4. Slot-level F1：参数字段粒度F1
5. Execution Success Rate：执行成功率
6. Step Order Accuracy：多步顺序准确率
7. Recovery Success Rate：失败恢复成功率
8. Hallucinated Final Answer Rate：幻觉答案率

### 阶段5：SFT数据构造与小规模训练（5-7天）

#### 任务清单
1. 选择训练子集（G1:500, G2:300, Eval:100）
2. 将trajectory拆成step-level SFT
3. 设计SFT数据格式（instruction+input+output）
4. 编写转换脚本
5. 训练LoRA模型（Qwen3-4B + LLaMA-Factory）
6. SFT前后对比分析

#### 产出文件
```
scripts/
└── convert_toolbench_to_qwen_sft.py

data/
├── toolbench_sft_train.jsonl
└── toolbench_sft_eval.jsonl

train/
└── sft_config.yaml

reports/
├── sft_data_statistics.md
├── sft_before_after_report.md
└── sft_bad_cases.jsonl

notes/
└── stage5_sft_review.md
```

#### 训练设置建议
| 参数 | 建议值 |
|------|--------|
| model | Qwen3-4B-Instruct |
| cutoff_len | 2048-4096 |
| epoch | 2-3 |
| LoRA rank | 16-32 |
| learning rate | 1e-4 ~ 2e-4 |
| batch size | 2-4 |
| eval interval | 每100-200 step |

### 阶段6：DPO/ORPO偏好数据构造（3-5天）

#### 任务清单
1. 用SFT模型rollout评估集
2. 筛选失败步骤（7类错误）
3. 构造chosen/rejected pair
4. 设计pair类型分布
5. 训练DPO/ORPO小实验
6. 输出错误迁移报告

#### 产出文件
```
logs/
├── sft_rollout_eval.jsonl
└── sft_failed_steps.jsonl

scripts/
└── build_dpo_pairs_from_rollout.py

data/
├── toolbench_dpo_train.jsonl
└── toolbench_dpo_eval.jsonl

train/
└── dpo_config.yaml

reports/
├── dpo_pair_statistics.md
└── dpo_error_shift_report.md

notes/
└── stage6_preference_learning_review.md
```

#### DPO pair类型分布
| Pair类型 | 占比 | 目的 |
|----------|------|------|
| tool_selection | 30%-35% | 压制错误工具选择 |
| argument | 30%-35% | 压制缺参、错参 |
| recovery | 15%-20% | 学会失败修复 |
| stopping | 10%-15% | 学会继续/停止判断 |
| final_answer | 5%-10% | 压制胡编答案 |

### 阶段7：迁移成业务Mini-ToolBench（3-5天）

#### 任务清单
1. 定义业务场景（5个工具以内）
2. 设计5个业务工具
3. 编写tool schema
4. 实现mock executor
5. 构造初始训练数据
6. 复用阶段4的evaluator
7. 跑完整小闭环
8. 编写最终学习总结

#### 产出文件
```
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

#### 公安业务工具示例
| 工具名 | 输入 | 输出 | 作用 |
|--------|------|------|------|
| extract_case_facts | case_text | facts | 抽取案情事实要素 |
| search_legal_rules | query, case_type | rule_snippets | 检索法条/规则片段 |
| classify_case | facts, rules | case_type_result | 判断刑事/行政/不足判断 |
| check_upgrade_conditions | facts, case_type | upgrade_result | 判断是否存在升格情形 |
| recommend_next_question | case_state, missing_slots | question | 推荐下一问 |

## 三、每日学习节奏

### 3.1 时间分配建议
| 时间段 | 活动 | 时长 | 产出 |
|--------|------|------|------|
| 上午 | 阅读/理解 | 2-3小时 | 笔记文档 |
| 下午 | 代码/数据 | 2-3小时 | 脚本/数据文件 |
| 晚上 | 复盘/总结 | 1-2小时 | review报告 |

### 3.2 每日检查清单
1. 今天的学习目标是什么？
2. 完成了哪些具体任务？
3. 遇到了什么困难？
4. 如何解决这些困难？
5. 明天的计划是什么？

### 3.3 每周复盘模板
```markdown
# 第X周学习复盘

## 1. 本周完成内容
- 阶段X任务完成情况
- 产出文件清单
- 关键理解点

## 2. 本周收获
- 新增理解的概念
- 掌握的技能
- 解决的难题

## 3. 本周不足
- 未完成的任务
- 理解不透彻的地方
- 时间管理问题

## 4. 下周计划
- 具体任务安排
- 重点突破方向
- 风险预判与应对
```

## 四、阶段间门槛检查

### 4.1 检查标准
| 阶段 | 进入下一阶段条件 |
|------|------------------|
| 0→1 | 能说清ToolBench学习边界，不再追求完整复现 |
| 1→2 | 能画出data→train→inference→eval流程 |
| 2→3 | 能手工拆解30+条trajectory |
| 3→4 | 能跑通最小action→executor→observation闭环 |
| 4→5 | 能自动输出step-level metrics和bad case |
| 5→6 | 能证明SFT改善了哪类Agent能力 |
| 6→7 | 能构造100+条高质量DPO pair |
| 7→完成 | 能完成业务版Mini-ToolBench初版 |

### 4.2 未通过处理
1. **识别问题根源**：概念/工程/数据/训练/评估
2. **针对性补课**：重新学习相关部分
3. **调整学习节奏**：延长当前阶段时间
4. **寻求帮助**：查阅资料/请教他人

## 五、学习资源与工具

### 5.1 核心资源
1. **ToolBench GitHub**：https://github.com/OpenBMB/ToolBench
2. **ToolLLM论文**：https://arxiv.org/abs/2307.16789
3. **学习计划文档**：ToolBench学习与实操计划_阶段0-7_严格执行版.md
4. **相关论文**：
   - ToolPrefer：https://arxiv.org/abs/2406.07115
   - StableToolBench：https://arxiv.org/abs/2403.07714

### 5.2 工具环境
1. **Python环境**：Python 3.9+
2. **深度学习框架**：PyTorch 2.0+
3. **训练工具**：LLaMA-Factory / transformers
4. **模型资源**：Hugging Face模型库
5. **开发工具**：VS Code / Jupyter Notebook

### 5.3 参考项目
1. **OpenHands**：多模态工具学习
2. **Search-R1**：搜索增强Agent
3. **rStar**：强化学习框架
4. **BMTools**：工具学习工具包

## 六、风险预判与应对

### 6.1 常见风险
| 风险类型 | 表现 | 应对策略 |
|----------|------|----------|
| 环境搭建 | 依赖包冲突、环境配置复杂 | 使用conda环境、逐步安装 |
| 数据理解 | 轨迹结构复杂、难以理解 | 手工分析少量样本、逐步深入 |
| 训练困难 | 显存不足、训练不稳定 | 使用LoRA、梯度累积、混合精度 |
| 评估缺失 | 只看最终答案、忽略过程 | 先建立evaluator、再训练 |
| 迁移困难 | 不知如何应用到业务 | 从简单场景开始、逐步增加复杂度 |

### 6.2 心态调整
1. **接受不完美**：不追求完整复现，关注核心方法
2. **循序渐进**：从简单到复杂，逐步深入
3. **产出导向**：每天都有具体输出
4. **问题驱动**：带着问题学习，针对性解决
5. **复盘反思**：定期总结，调整学习策略

## 七、最终能力验收

### 7.1 能力清单
| 能力 | 验收标准 | 自测问题 |
|------|----------|----------|
| Agent抽象 | 能用State/Action/Observation描述Agent | 当前状态是什么？动作空间是什么？反馈是什么？ |
| 工具设计 | 能定义tool schema和executor | 这个工具需要哪些参数？返回如何标准化？ |
| 轨迹理解 | 能拆解多步tool-use trajectory | 哪一步是action？哪一步是observation？ |
| SFT数据构造 | 能把trajectory转成step-level SFT | 输入输出格式是否稳定？是否可解析？ |
| Evaluator | 能设计step-level指标 | 模型错在工具、参数、schema还是恢复？ |
| 错误诊断 | 能输出error taxonomy和bad case | 主要错误类型是什么？下一步改数据还是改方法？ |
| DPO数据构造 | 能构造chosen/rejected pair | rejected为什么错？chosen为什么更好？ |
| 业务迁移 | 能做一个Mini-ToolBench | 能否脱离ToolBench仓库复现方法？ |
| RL准备 | 能判断何时进入Search-R1/GRPO | 当前瓶颈是局部动作错误还是长程策略问题？ |

### 7.2 最终验收问题
```
如果给你一个新业务场景，
你是否能设计工具集合、构造200条SFT样本、建立8类错误指标、
跑一次baseline/SFT对比，并说明下一步是否需要DPO或RL？
```

## 八、后续学习路径

### 完成ToolBench学习后
1. **Search Agent方向**：OpenHands / Search-R1
2. **强化学习方向**：rStar / GRPO / PPO
3. **多模态方向**：视觉/语音工具调用
4. **业务深化方向**：特定领域Agent开发

### 长期发展
1. **技术深度**：深入研究特定方向（评估/训练/检索）
2. **工程能力**：大规模分布式训练、服务部署
3. **产品思维**：从技术到产品的转化
4. **团队协作**：带领团队开发复杂Agent系统

---
**行动建议：**
1. **立即开始**：完成阶段0的学习边界文档
2. **建立习惯**：每日固定学习时间，产出导向
3. **定期复盘**：每周总结，调整学习策略
4. **保持专注**：专注于当前阶段目标，不跳跃前进
5. **享受过程**：将学习视为探索和成长的过程