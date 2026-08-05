# ToolBench核心文件列表

## 一、项目结构核心文件

### 1.1 数据目录
```
data_example/
├── instruction/              # 指令数据
│   ├── G1_query.json        # 单工具指令
│   ├── G2_query.json        # 类内多工具指令
│   └── G3_query.json        # 集合内多工具指令
├── answer/                   # 答案轨迹
│   ├── G1_answer/           # 单工具答案
│   ├── G2_answer/           # 类内多工具答案
│   └── G3_answer/           # 集合内多工具答案
└── toolenv/                 # 工具环境
    ├── tools/               # API定义和实现
    └── response_examples/   # API响应示例
```

### 1.2 核心代码模块

#### 推理模块 (toolbench/inference/)
```
qa_pipeline.py               # 主推理管道(close-domain)
qa_pipeline_open_domain.py   # 开放域推理管道
toolbench_server.py          # 服务端接口
server.py                    # 服务器实现

Algorithms/                  # 推理算法
├── base_search.py          # 基础搜索算法
├── DFS.py                  # 深度优先搜索
├── single_chain.py         # 单链推理
└── __init__.py

Downstream_tasks/           # 下游任务
├── base_env.py            # 基础环境
├── rapidapi.py            # RapidAPI环境
└── __init__.py

LLM/                        # 模型接口
├── base_io.py             # 基础IO
├── chatgpt_function_model.py  # ChatGPT模型
├── davinci_model.py       # Davinci模型
├── llama_model.py         # LLaMA模型
├── tool_llama_model.py    # ToolLLaMA模型
├── tool_llama_lora_model.py # LoRA版ToolLLaMA
└── retriever.py           # 检索器接口

Tree/                       # 树结构
├── Tree.py                # 树实现
└── __init__.py

Prompts/                    # 提示词
├── ReAct_prompts.py       # ReACT提示
├── Tree_search_prompts.py # 树搜索提示
├── rank_prompts.py        # 排序提示
└── __init__.py
```

#### 训练模块 (toolbench/train/)
```
train.py                    # 全参数微调
train_lora.py               # LoRA训练
train_mem.py                # 内存优化训练
llama_condense_monkey_patch.py  # LLaMA条件补丁
llama_flash_attn_monkey_patch.py # FlashAttention补丁
```

#### 评估模块 (toolbench/tooleval/)
```
eval_pass_rate.py           # 通过率评估
eval_preference.py          # 偏好评估
convert_to_answer_format.py # 答案格式转换
automatic_eval_sample.py    # 自动评估采样
evaluators/                 # 评估器实现
dataset/                    # 评估数据集
evaluation/                 # 评估核心
```

#### 检索模块 (toolbench/retrieval/)
```
train.py                    # 检索器训练
api_evaluator.py            # API评估器
inference_example.py        # 推理示例
```

### 1.3 预处理模块
```
preprocess/
├── preprocess_retriever_data.py   # 检索器数据预处理
└── preprocess_toolllama_data.py   # ToolLLaMA数据预处理
```

### 1.4 辅助模块
```
toolbench/
├── tool_conversation.py    # 对话模板
├── utils.py                # 工具函数
└── model/                  # 模型适配
    ├── model_adapter.py    # 模型适配器
    ├── compression.py      # 压缩工具
    ├── make_delta.py       # delta制作
    └── apply_delta.py      # delta应用
```

## 二、关键配置文件

### 2.1 训练配置文件示例
```yaml
# train/sft_config.yaml
model_name_or_path: "huggyllama/llama-7b"
data_path: "data/toolllama_G123_dfs_train.json"
eval_data_path: "data/toolllama_G123_dfs_eval.json"
conv_template: "tool-llama-single-round"
bf16: True
output_dir: "toolllama"
num_train_epochs: 2
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5e-5
weight_decay: 0.0
warmup_ratio: 0.04
lr_scheduler_type: "cosine"
fsdp: "full_shard auto_wrap"
fsdp_transformer_layer_cls_to_wrap: "LlamaDecoderLayer"
model_max_length: 8192
gradient_checkpointing: True
lazy_preprocess: True
```

### 2.2 推理配置文件示例
```bash
# 推理脚本参数
--backbone_model toolllama                    # 模型类型
--model_path ToolBench/ToolLLaMA-7b          # 模型路径
--tool_root_dir data/toolenv/tools/          # 工具目录
--max_observation_length 1024                # 观察长度限制
--observ_compress_method truncate            # 观察压缩方法
--method DFS_woFilter_w2                     # 推理方法
--input_query_file data/test_instruction/G1_instruction.json  # 输入
--output_answer_file toolllama_dfs_inference_result  # 输出
--toolbench_key $TOOLBENCH_KEY               # ToolBench密钥
```

## 三、数据格式示例

### 3.1 指令数据格式 (G1_query.json)
```json
{
    "api_list": [
        {
            "category_name": "Logistics",
            "tool_name": "SQUAKE",
            "api_name": "Checkhealth",
            "api_description": " ",
            "required_parameters": [],
            "optional_parameters": [],
            "method": "GET"
        }
    ],
    "query": "检查SQUAKE API的健康状态",
    "relevant APIs": [["SQUAKE", "Checkhealth"]]
}
```

### 3.2 答案轨迹格式
```json
{
    "win": true,
    "tree": {
        "size": 7,
        "max_length": 7,
        "tree": {
            "is_terminal": false,
            "pruned": false,
            "finished": false,
            "depth": 0,
            "node_type": "Action Input",
            "description": "",
            "children": [
                {
                    "is_terminal": false,
                    "pruned": false,
                    "finished": false,
                    "depth": 1,
                    "node_type": "Action",
                    "description": "tool_name",
                    "children": [...]
                }
            ]
        }
    }
}
```

### 3.3 工具schema格式
```json
{
    "tool_description": "Return hello world.",
    "tool_name": "hello world",
    "title": "hello world",
    "api_list": [
        {
            "name": "get_hello_world",
            "url": "",
            "description": "To get 'hello world'.",
            "method": "GET",
            "required_parameters": [],
            "optional_parameters": []
        }
    ],
    "standardized_name": "hello_world"
}
```

## 四、API实现示例

### 4.1 简单API实现
```python
# data_example/toolenv/tools/Customized/hello_world/api.py
def get_hello_world():
    """
    To get hello world 
    """
    observation = "hello world"
    return observation
```

### 4.2 参数化API实现
```python
def get_weather(city: str, country: str = "US"):
    """
    Get weather information for a city
    """
    # API调用逻辑
    observation = {
        "city": city,
        "country": country,
        "temperature": 25,
        "condition": "sunny"
    }
    return observation
```

## 五、学习阶段关键文件

### 阶段0-1：理解文件
- README.md / README_ZH.md
- assets/paper.pdf
- toolbench/ 目录结构

### 阶段2：数据分析文件
- data_example/instruction/ 目录
- data_example/answer/ 目录
- data_example/toolenv/tools/ 目录

### 阶段3：最小闭环文件
- toolbench/inference/Downstream_tasks/base_env.py
- toolbench/inference/LLM/base_io.py
- toolbench/inference/Algorithms/single_chain.py

### 阶段4：评估文件
- toolbench/tooleval/eval_pass_rate.py
- toolbench/tooleval/eval_preference.py
- toolbench/tooleval/evaluators/

### 阶段5：训练文件
- toolbench/train/train.py
- toolbench/train/train_lora.py
- preprocess/preprocess_toolllama_data.py

### 阶段6：DPO文件
- toolbench/tooleval/convert_to_answer_format.py
- 自定义脚本构造DPO pair

### 阶段7：迁移文件
- 自定义mini_toolbench/目录结构
- 自定义工具schema和executor