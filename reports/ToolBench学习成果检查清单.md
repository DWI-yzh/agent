# ToolBench学习成果检查清单

## 一、学习阶段完成情况追踪

### 阶段0：学习边界与基础认知
- [ ] 建立本地学习目录结构
- [ ] Clone ToolBench项目到repos/
- [ ] 阅读README.md（英文+中文）
- [ ] 阅读论文摘要与方法部分
- [ ] 创建`notes/toolbench_learning_scope.md`
- [ ] 创建`notes/toolbench_glossary.md`
- [ ] 创建`notes/stage0_review.md`

**阶段验收标准：** 能说清ToolBench学习边界，不再把目标设为完整复现ToolLLaMA

### 阶段1：项目结构与Agent抽象对齐
- [ ] 扫描项目目录结构（tree -L 3）
- [ ] 创建`notes/toolbench_tree_l3.txt`
- [ ] 建立目录功能映射表
- [ ] 绘制端到端流程图（`notes/toolbench_pipeline.md`）
- [ ] 创建Agent抽象映射表（`notes/toolbench_agent_mapping.md`）
- [ ] 编写阶段复盘（`notes/stage1_project_structure_review.md`）

**阶段验收标准：** 能画出ToolBench data→train→inference→eval流程

### 阶段2：数据格式与轨迹结构分析
- [ ] 抽样G1数据（10+条）→ `data_samples/g1_10.jsonl`
- [ ] 抽样G2数据（10+条）→ `data_samples/g2_10.jsonl`
- [ ] 抽样G3数据（10+条）→ `data_samples/g3_10.jsonl`
- [ ] 手工标注30+条轨迹
- [ ] 转换为SAO格式 → `data_samples/state_action_observation_examples.jsonl`
- [ ] 创建G1/G2/G3复杂度分析 → `notes/g1_g2_g3_complexity_analysis.md`
- [ ] 建立数据质量问题清单 → `notes/toolbench_data_quality_notes.md`
- [ ] 编写数据分析报告 → `notes/toolbench_data_analysis.md`

**阶段验收标准：** 能手工拆解至少30条trajectory

### 阶段3：最小工具调用闭环
- [ ] 选择close-domain模式（5-10个工具）
- [ ] 实现action parser → `scripts/action_parser.py`
- [ ] 实现mock executor → `scripts/mock_executor.py`
- [ ] 实现最小Agent循环 → `scripts/run_minimal_tool_agent.py`
- [ ] 运行50+条最小轨迹 → `logs/minimal_rollout_50.jsonl`
- [ ] 构造6类失败场景 → `logs/failure_injection_cases.jsonl`
- [ ] 编写阶段复盘 → `notes/stage3_minimal_agent_review.md`

**阶段验收标准：** 能跑通最小action→executor→observation闭环

### 阶段4：Evaluator与错误分类体系
- [ ] 定义gold label格式
- [ ] 实现step-level metrics（8+个）→ `eval/metrics.py`
- [ ] 建立错误分类体系（10+类）→ `eval/error_taxonomy.py`
- [ ] 实现自动评估脚本 → `eval/run_eval.py`
- [ ] 生成评估报告 → `reports/eval_summary.md`
- [ ] 输出bad cases → `reports/bad_cases.jsonl`
- [ ] 统计错误分布 → `reports/error_distribution.csv`
- [ ] 创建人工复核模板 → `reports/manual_review_template.csv`
- [ ] 编写评估设计文档 → `notes/stage4_evaluator_design.md`

**阶段验收标准：** 能自动输出step-level metrics和bad case

### 阶段5：SFT数据构造与小规模训练
- [ ] 选择训练子集（G1:500, G2:300, Eval:100）
- [ ] 设计SFT数据格式
- [ ] 编写数据转换脚本 → `scripts/convert_toolbench_to_qwen_sft.py`
- [ ] 生成SFT训练数据 → `data/toolbench_sft_train.jsonl`
- [ ] 生成SFT评估数据 → `data/toolbench_sft_eval.jsonl`
- [ ] 创建训练配置 → `train/sft_config.yaml`
- [ ] 训练LoRA模型（Qwen3-4B）
- [ ] 生成数据统计报告 → `reports/sft_data_statistics.md`
- [ ] 生成SFT前后对比报告 → `reports/sft_before_after_report.md`
- [ ] 输出SFT bad cases → `reports/sft_bad_cases.jsonl`
- [ ] 编写阶段复盘 → `notes/stage5_sft_review.md`

**阶段验收标准：** 能证明SFT改善了哪类Agent能力

### 阶段6：DPO/ORPO偏好数据构造
- [ ] 用SFT模型rollout评估集 → `logs/sft_rollout_eval.jsonl`
- [ ] 筛选失败步骤 → `logs/sft_failed_steps.jsonl`
- [ ] 编写DPO pair构造脚本 → `scripts/build_dpo_pairs_from_rollout.py`
- [ ] 生成DPO训练数据 → `data/toolbench_dpo_train.jsonl`
- [ ] 生成DPO评估数据 → `data/toolbench_dpo_eval.jsonl`
- [ ] 创建DPO训练配置 → `train/dpo_config.yaml`
- [ ] 训练DPO小实验
- [ ] 生成DPO pair统计报告 → `reports/dpo_pair_statistics.md`
- [ ] 生成错误迁移报告 → `reports/dpo_error_shift_report.md`
- [ ] 编写阶段复盘 → `notes/stage6_preference_learning_review.md`

**阶段验收标准：** 能构造至少100条高质量DPO pair

### 阶段7：迁移成业务Mini-ToolBench
- [ ] 定义业务场景（5个工具以内）
- [ ] 设计5个业务工具schema → `mini_toolbench/env/schemas.py`
- [ ] 实现工具函数 → `mini_toolbench/env/tools.py`
- [ ] 实现executor → `mini_toolbench/env/executor.py`
- [ ] 构造初始训练数据 → `mini_toolbench/data/`（4个文件）
- [ ] 实现评估脚本 → `mini_toolbench/eval/`（3个文件）
- [ ] 实现Agent运行脚本 → `mini_toolbench/scripts/run_agent.py`
- [ ] 实现数据转换脚本 → `mini_toolbench/scripts/convert_to_sft.py`
- [ ] 实现DPO构造脚本 → `mini_toolbench/scripts/build_dpo_pairs.py`
- [ ] 创建训练配置 → `mini_toolbench/train/`（2个文件）
- [ ] 跑完整小闭环（baseline→SFT→DPO）
- [ ] 生成各阶段报告 → `mini_toolbench/reports/`（4个文件）
- [ ] 编写最终学习总结 → `reports/toolbench_final_learning_report.md`

**阶段验收标准：** 能完成一个业务版Mini-ToolBench初版

## 二、学习产出文件清单

### 2.1 笔记文档（notes/）
```
notes/
├── toolbench_learning_scope.md
├── toolbench_glossary.md
├── stage0_review.md
├── toolbench_tree_l3.txt
├── toolbench_pipeline.md
├── toolbench_agent_mapping.md
├── stage1_project_structure_review.md
├── toolbench_data_analysis.md
├── g1_g2_g3_complexity_analysis.md
├── toolbench_data_quality_notes.md
├── stage3_minimal_agent_review.md
├── stage4_evaluator_design.md
├── stage5_sft_review.md
├── stage6_preference_learning_review.md
└── stage7_business_migration_review.md
```

### 2.2 数据样本（data_samples/）
```
data_samples/
├── g1_10.jsonl
├── g2_10.jsonl
├── g3_10.jsonl
└── state_action_observation_examples.jsonl
```

### 2.3 脚本工具（scripts/）
```
scripts/
├── action_parser.py
├── mock_executor.py
├── run_minimal_tool_agent.py
├── convert_toolbench_to_qwen_sft.py
└── build_dpo_pairs_from_rollout.py
```

### 2.4 评估工具（eval/）
```
eval/
├── metrics.py
├── error_taxonomy.py
└── run_eval.py
```

### 2.5 训练数据（data/）
```
data/
├── toolbench_sft_train.jsonl
├── toolbench_sft_eval.jsonl
├── toolbench_dpo_train.jsonl
└── toolbench_dpo_eval.jsonl
```

### 2.6 训练配置（train/）
```
train/
├── sft_config.yaml
└── dpo_config.yaml
```

### 2.7 日志记录（logs/）
```
logs/
├── minimal_rollout_50.jsonl
├── failure_injection_cases.jsonl
├── sft_rollout_eval.jsonl
└── sft_failed_steps.jsonl
```

### 2.8 分析报告（reports/）
```
reports/
├── eval_summary.md
├── bad_cases.jsonl
├── error_distribution.csv
├── manual_review_template.csv
├── sft_data_statistics.md
├── sft_before_after_report.md
├── sft_bad_cases.jsonl
├── dpo_pair_statistics.md
├── dpo_error_shift_report.md
└── toolbench_final_learning_report.md
```

### 2.9 Mini-ToolBench（mini_toolbench/）
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

## 三、学习成果量化指标

### 3.1 知识理解指标
| 知识领域 | 掌握程度 | 衡量方式 |
|----------|----------|----------|
| Agent抽象 | 能用State/Action/Observation描述 | 能解释10个轨迹样本 |
| 工具设计 | 能定义schema和executor | 能设计5个业务工具 |
| 轨迹理解 | 能拆解多步调用 | 能手工分析30+条轨迹 |
| 数据构造 | 能构造SFT/DPO数据 | 能转换200+条训练样本 |
| 评估体系 | 能设计step-level指标 | 能实现8+个评估指标 |
| 错误诊断 | 能分类和诊断错误 | 能识别10+类错误 |
| 训练策略 | 能设计SFT/DPO训练 | 能完成baseline/SFT/DPO对比 |

### 3.2 技能掌握指标
| 技能 | 掌握程度 | 产出证据 |
|------|----------|----------|
| 代码阅读 | 能理解核心模块 | 项目结构分析报告 |
| 脚本开发 | 能实现基本功能 | 5个以上功能脚本 |
| 数据处理 | 能处理轨迹数据 | 数据转换脚本 |
| 模型训练 | 能完成小规模训练 | 训练配置+报告 |
| 评估分析 | 能进行错误分析 | 评估报告+bad cases |
| 业务迁移 | 能应用到新场景 | Mini-ToolBench完整项目 |

### 3.3 产出质量指标
| 产出类型 | 质量要求 | 验收标准 |
|----------|----------|----------|
| 文档报告 | 结构清晰、内容完整 | 7个阶段复盘报告 |
| 代码脚本 | 可运行、注释清晰 | 所有脚本可执行 |
| 数据文件 | 格式规范、质量合格 | JSON格式正确、无解析错误 |
| 训练模型 | 有baseline对比 | 至少2个指标提升 |
| 评估结果 | 有量化分析 | 错误分布统计+bad cases |
| 迁移项目 | 完整可运行 | Mini-ToolBench完整闭环 |

## 四、学习过程检查点

### 每周检查点
**第1周结束检查：**
- [ ] 阶段0-2全部完成
- [ ] 理解项目架构和数据格式
- [ ] 建立完整的学习笔记体系

**第2周结束检查：**
- [ ] 阶段3-4全部完成
- [ ] 实现最小闭环和评估体系
- [ ] 掌握step-level评估方法

**第3周结束检查：**
- [ ] 阶段5-6全部完成
- [ ] 完成SFT训练和DPO数据构造
- [ ] 掌握训练优化方法

**第4周结束检查：**
- [ ] 阶段7全部完成
- [ ] 成功迁移到业务场景
- [ ] 产出完整学习总结

### 每日检查清单
1. **目标明确**：今天要完成什么？
2. **时间投入**：实际投入多少时间？
3. **产出确认**：产出了哪些文件？
4. **问题记录**：遇到了什么困难？
5. **解决进展**：如何解决的？
6. **明日计划**：明天做什么？

### 阶段验收问题
每个阶段结束时回答以下问题：

1. **核心目标是否完成？**（是/否，具体证据）
2. **新增理解了什么？**（列出3个关键点）
3. **当前最大的卡点是什么？**（概念/工程/数据/训练/评估）
4. **这个卡点属于什么类型？**（技术问题/理解问题/资源问题）
5. **下一阶段需要补课吗？**（如果需要，补什么）
6. **产出文件是否齐全？**（检查文件清单）
7. **是否满足阶段达成指标？**（对照验收标准）

## 五、学习成果展示

### 5.1 最终学习汇报内容
完成所有阶段后，应能进行30分钟的学习汇报：

1. **学习背景与目标**（5分钟）
   - 为什么学习ToolBench
   - 学习目标设定
   - 学习边界确定

2. **核心内容掌握**（10分钟）
   - Agent抽象理解
   - 项目架构分析
   - 数据处理方法
   - 训练评估体系

3. **实践成果展示**（10分钟）
   - 最小闭环演示
   - SFT训练结果
   - DPO优化效果
   - Mini-ToolBench项目

4. **学习总结与展望**（5分钟）
   - 关键收获
   - 能力提升
   - 后续学习计划
   - 业务应用展望

### 5.2 学习成果证明材料
1. **学习笔记**：7个阶段复盘文档
2. **代码仓库**：包含所有脚本和项目
3. **训练报告**：SFT/DPO对比分析
4. **评估报告**：错误分类和bad cases
5. **迁移项目**：Mini-ToolBench完整代码
6. **最终总结**：完整学习总结报告

### 5.3 能力认证标准
**初级Agent选手认证条件：**
1. ✅ 能独立设计小型Tool Agent sandbox
2. ✅ 能构造SFT/DPO训练数据
3. ✅ 能建立step-level evaluator
4. ✅ 能完成小规模训练与错误诊断
5. ✅ 能将ToolBench方法迁移到业务场景

## 六、后续学习建议

### 6.1 巩固学习成果
1. **项目复盘**：重新审视每个阶段产出
2. **代码优化**：改进脚本和工具
3. **文档完善**：补充注释和说明
4. **经验总结**：提炼学习方法论

### 6.2 进阶学习方向
完成ToolBench学习后，可选择以下方向：

| 方向 | 适合人群 | 推荐项目 | 学习周期 |
|------|----------|----------|----------|
| Search Agent | 对搜索增强感兴趣 | OpenHands / Search-R1 | 4-6周 |
| 强化学习 | 想深入RL优化 | rStar / GRPO | 6-8周 |
| 多模态 | 想结合视觉/语音 | 多模态工具调用 | 4-6周 |
| 业务深化 | 特定领域应用 | 公安/医疗/金融Agent | 8-12周 |

### 6.3 职业发展建议
1. **技术深度**：选择1-2个方向深入
2. **工程能力**：学习大规模训练和部署
3. **产品思维**：从技术到产品的转化
4. **团队协作**：参与开源项目或团队项目

---
**最终提醒：**
1. **保持耐心**：Agent学习需要时间和实践
2. **注重基础**：概念理解比代码运行更重要
3. **产出导向**：每个阶段都要有具体产出
4. **及时复盘**：定期总结，调整学习策略
5. **享受过程**：将学习视为探索和成长的机会

**祝你学习顺利，早日成为优秀的Agent选手！**