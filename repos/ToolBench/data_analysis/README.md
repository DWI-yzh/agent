# ToolBench 阶段 2 数据分析

本目录用于执行《ToolBench 学习与实操计划（阶段 0-7 严格执行版）》的阶段 2。

## 任务 2.3.1：抽样 G1 / G2 / G3 数据

官方仓库的 `data_example/` 每组只有 5 条 query 和 5 条 answer，不能满足每组至少
10 条的要求。因此，本任务使用公开的 ToolBench SFT validation 镜像取得完整对话轨迹，
并使用原始 G2/G3 query 索引恢复分组。

分类规则：

- G2：query 与 G2 索引精确匹配；
- G3：query 与 G3 索引精确匹配；
- G1：query 不属于 G2/G3，且 system prompt 中只暴露一个原始 Tool；
- 无法可靠分类的记录直接跳过，不纳入抽样。

ToolBench 预处理数据还包含同一 trajectory 的中间前缀。脚本只保留最后一条 assistant
消息调用 `Finish` 的记录，确保输出是完整 trajectory，而不是中间 step。

### 执行

首次执行时自动下载缺失的输入文件：

```powershell
python data_analysis/scripts/sample_toolbench_data.py --download-missing
```

已有 `source_cache/` 时：

```powershell
python data_analysis/scripts/sample_toolbench_data.py
```

默认使用随机种子 `42`，每组抽取 10 条，其中：

- 5 条以 `Finish -> give_answer` 结束；
- 5 条以 `Finish -> give_up_and_restart` 结束。

这样既能分析成功路径，也能分析失败、重试和 DPO rejected 候选。缓存输入位于
`source_cache/`，已被 `.gitignore` 排除。

### 输出

```text
data_samples/
├── g1_10.jsonl
├── g2_10.jsonl
├── g3_10.jsonl
└── sample_manifest.json
```

每条记录保留完整 `conversations`，并附加：

- `sample_id` 和内容稳定的 `sample_uid`；
- `group` 与 `classification_basis`；
- `original_tools`、`tool_count`；
- `action_names`、`action_count`；
- `outcome`；
- 数据来源 `provenance`。

`sample_manifest.json` 记录候选数量、抽样参数、每个输出文件的记录数、唯一 query 数、
outcome 分布和 SHA-256，供复现与验收使用。

## 任务 2.3.2 / 2.3.3：本地标注工作台

本目录包含一个仅依赖 Python 标准库的本地应用：

```powershell
cd D:\work\agent-lab\repos\ToolBench\data_analysis
python app.py --open
```

默认地址：`http://127.0.0.1:8765`。如果不希望自动打开浏览器，可以运行：

```powershell
python app.py
```

页面包括：

- **任务 2.3.2 轨迹分析**：样本概览、逐 Action 参数来源、Observation 分类、实际/应有下一步和错误类型；
- **任务 2.3.3 SAO 转换**：独立页面显示原始证据、结构化 SAO 表单、JSON 预览和前后 State 差异；
- G1/G2/G3 共 30 条样本导航；
- 自动保存、步骤完成状态和 JSONL 导出。

标注数据保存在：

```text
annotations/records.json
```

导出入口：

- 2.3.2：`toolbench_trajectory_annotations.jsonl`；
- 2.3.3：`state_action_observation_examples.jsonl`。

应用只监听本机 `127.0.0.1`，不会把标注数据发送到外部服务。
