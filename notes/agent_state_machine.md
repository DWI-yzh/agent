# Agent 运行时状态机 - 逻辑清晰深入版

## 📋 文档定位与使用指南

### **本文档目标**
1. **从零开始**：即使没有Agent开发经验，也能逐步理解状态机原理
2. **由浅入深**：从基础概念到具体实现，层层递进
3. **连接实践**：每个知识点都对应实际开发中的具体应用
4. **完整覆盖**：涵盖设计、实现、训练、评估的全流程

### **阅读建议**
- **初学者**：按顺序阅读，确保理解每个概念后再继续
- **有经验者**：可直接跳到感兴趣的章节
- **实战参考**：将代码示例与实际项目结合理解

---

## 第一部分：基础概念建立

### 1.1 为什么Agent需要状态机？

#### **传统LLM vs Agent的思维差异**

```python
# 传统LLM：单次问答模型
def traditional_llm(question):
    """输入问题，直接输出答案"""
    return model.generate(question)

# Agent：多步决策系统  
def agent_system(task):
    """需要管理状态、决策、执行、更新的完整流程"""
    state = initialize_state(task)
    
    while not terminated(state):
        # 1. 准备当前状态
        prepared_state = prepare_for_model(state)
        
        # 2. 模型基于状态决策
        action = model_decide(prepared_state)
        
        # 3. 系统执行动作
        result = execute_action(action)
        
        # 4. 更新状态继续
        state = update_state(state, action, result)
    
    return get_final_answer(state)
```

**核心洞察**：Agent不是"问答机"，而是"决策机+执行器"的组合。

#### **状态机的必要性**
1. **记忆需求**：Agent需要记住之前的对话和操作
2. **多步决策**：复杂任务需要多个步骤完成
3. **错误恢复**：需要基于当前状态决定如何恢复
4. **工具协调**：多个工具调用需要状态跟踪

### 1.2 状态机基础：三个核心概念

#### **状态（State）**：Agent当前知道的一切
```python
class AgentState:
    def __init__(self):
        # 对话历史：模型能看到的所有信息
        self.messages = []
        
        # 可用工具：当前可以调用的工具列表
        self.available_tools = []
        
        # 内部状态：系统维护但不暴露给模型的信息
        self.internal = {
            "step_count": 0,
            "max_steps": 10,
            "terminated": False,
            "failure_types": []
        }
```

#### **动作（Action）**：Agent可以做的事情
```python
# 动作类型枚举
class ActionType:
    TOOL_CALL = "tool_call"      # 调用工具
    FINAL_ANSWER = "final_answer" # 给出最终答案
    ASK_USER = "ask_user"        # 反问用户
    
# 动作数据结构
tool_action = {
    "type": "tool_call",
    "tool_name": "weather",
    "arguments": {"location": "上海", "date": "明天"}
}

final_answer = {
    "type": "final_answer",
    "content": "明天上海气温适中，适合户外活动"
}

ask_user = {
    "type": "ask_user", 
    "content": "您想查询哪个城市的天气？"
}
```

#### **状态转移（State Transition）**：动作如何改变状态
```python
def transition(current_state, action, result):
    """状态转移函数：定义状态如何变化"""
    new_state = current_state.copy()
    
    # 1. 添加动作到历史
    new_state.messages.append({
        "role": "assistant",
        "tool_calls": [action] if action.type == "tool_call" else None
    })
    
    # 2. 添加结果到历史（如果有）
    if result:
        new_state.messages.append({
            "role": "tool",
            "name": action.tool_name,
            "content": result
        })
    
    # 3. 更新内部状态
    new_state.internal["step_count"] += 1
    
    # 4. 检查终止条件
    if action.type == "final_answer":
        new_state.internal["terminated"] = True
    
    return new_state
```

---

## 第二部分：状态机设计详解

### 2.1 状态设计的两个层次

#### **运行时状态（Runtime State）**：模型实际看到的信息
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "查询明天上海天气"},
    {
      "role": "assistant",
      "tool_calls": [{"name": "weather", "arguments": {"location": "上海", "date": "明天"}}]
    },
    {"role": "tool", "name": "weather", "content": "温度18-24C，降雨20%"}
  ],
  "tools": [
    {
      "name": "weather",
      "description": "查询天气",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {"type": "string"},
          "date": {"type": "string"}
        },
        "required": ["location", "date"]
      }
    }
  ]
}
```

**设计原则**：
1. **真实性**：必须与推理时完全一致
2. **完整性**：包含所有必要的历史信息
3. **无泄漏**：不包含任何标注或未来信息

#### **标注状态（Annotation State）**：用于分析和评估的信息
```json
{
  "progress": {
    "step": 2,
    "known_facts": ["上海明天气温18-24C", "降雨概率20%"],
    "open_requirements": ["判断是否适合跑步"]
  },
  "expected_next_action": {
    "type": "final_answer",
    "content_requirements": ["引用温度", "给出跑步建议"]
  },
  "failure_labels": []
}
```

**关键区分**：标注状态**绝不**进入模型输入，只用于训练数据构造和评估。

### 2.2 动作空间设计

#### **三类核心动作的详细设计**

**1. 工具调用动作**
```python
def create_tool_call(tool_name, arguments):
    """创建工具调用动作"""
    return {
        "type": "tool_call",
        "tool_name": tool_name,
        "arguments": arguments,
        # 可选：唯一标识符，用于工具结果匹配
        "id": f"call_{uuid.uuid4().hex[:8]}"
    }

# 合法性检查
def validate_tool_call(action, available_tools):
    """验证工具调用合法性"""
    # 1. 工具是否存在
    if action["tool_name"] not in available_tools:
        return False, "unknown_tool"
    
    # 2. 参数格式是否正确
    if "arguments" not in action:
        return False, "missing_arguments"
    
    # 3. 参数类型是否符合schema（后续检查）
    return True, None
```

**2. 最终回答动作**
```python
def create_final_answer(content):
    """创建最终回答动作"""
    return {
        "type": "final_answer",
        "content": content,
        # 可选：回答的置信度或依据
        "confidence": 0.9,
        "sources": ["weather_tool_result"]
    }

# 终止性检查
def should_terminate_after_final_answer(state, action):
    """检查最终回答后是否应该终止"""
    # 1. 是否还有未完成的需求
    if state.progress.open_requirements:
        return False, "premature_final_answer"
    
    # 2. 回答是否基于已有信息
    if not is_grounded_in_observations(action, state):
        return False, "ungrounded_answer"
    
    return True, None
```

**3. 反问用户动作**
```python
def create_ask_user(question, clarification_needed):
    """创建反问用户动作"""
    return {
        "type": "ask_user",
        "content": question,
        "clarification_type": clarification_needed,  # "missing_param", "ambiguous", "confirmation"
        "expected_response_type": "text"  # 或 "choice", "confirmation"
    }

# 必要性检查
def is_ask_user_necessary(state, action):
    """检查反问是否必要"""
    # 1. 是否确实缺少必要信息
    missing_info = identify_missing_information(state)
    if not missing_info:
        return False, "unnecessary_ask_user"
    
    # 2. 反问内容是否针对缺失信息
    if not addresses_missing_info(action, missing_info):
        return False, "irrelevant_ask_user"
    
    return True, None
```

### 2.3 状态转移规则设计

#### **基本转移逻辑**
```python
class StateTransitionRules:
    """状态转移规则集"""
    
    @staticmethod
    def after_tool_success(current_state, action, result):
        """工具成功后的状态转移"""
        new_state = current_state.copy()
        
        # 1. 更新消息历史
        new_state.messages.extend([
            create_assistant_message_with_tool_call(action),
            create_tool_message(action["tool_name"], result)
        ])
        
        # 2. 提取已知事实
        facts = extract_facts_from_result(result)
        new_state.progress.known_facts.extend(facts)
        
        # 3. 更新待完成需求
        new_state.progress.open_requirements = [
            req for req in current_state.progress.open_requirements
            if not is_requirement_satisfied(req, facts)
        ]
        
        # 4. 清空错误状态
        new_state.progress.error_state = None
        
        return new_state
    
    @staticmethod
    def after_schema_error(current_state, action, error):
        """Schema错误后的状态转移"""
        new_state = current_state.copy()
        
        # 1. 记录错误消息
        new_state.messages.extend([
            create_assistant_message_with_tool_call(action),
            create_error_message("schema_error", error.details)
        ])
        
        # 2. 设置恢复状态
        new_state.progress.error_state = {
            "type": "schema_error",
            "tool_name": action["tool_name"],
            "missing_fields": error.missing_fields,
            "incorrect_types": error.incorrect_types,
            "attempt_count": current_state.progress.error_state["attempt_count"] + 1 
                          if current_state.progress.error_state else 1,
            "max_attempts": 3
        }
        
        # 3. 添加失败标签
        new_state.failure_labels.append("invalid_schema")
        if error.missing_fields:
            new_state.failure_labels.append("missing_argument")
        
        return new_state
    
    @staticmethod
    def after_final_answer(current_state, action):
        """最终回答后的状态转移"""
        new_state = current_state.copy()
        
        # 1. 添加最终回答到历史
        new_state.messages.append(
            create_assistant_message_with_final_answer(action)
        )
        
        # 2. 标记终止
        new_state.internal.terminated = True
        new_state.internal.termination_reason = "final_answer"
        
        # 3. 设置成功状态（待评估）
        new_state.internal.success = None  # 由评估器决定
        
        return new_state
```

#### **完整转移决策表**
```python
class TransitionDecisionTable:
    """状态转移决策表"""
    
    DECISION_TABLE = {
        # (当前状态, 动作类型, 执行结果) -> 转移规则
        ("any", "tool_call", "success"): "after_tool_success",
        ("any", "tool_call", "schema_error"): "after_schema_error",
        ("any", "tool_call", "empty_result"): "after_empty_result",
        ("any", "tool_call", "tool_error"): "after_tool_error",
        ("any", "final_answer", None): "after_final_answer",
        ("any", "ask_user", None): "after_ask_user",
        ("error_state", "tool_call", "success"): "after_recovery_success",
    }
    
    @classmethod
    def get_transition_rule(cls, state, action, result):
        """获取适用的转移规则"""
        # 确定状态类型
        state_type = "error_state" if state.progress.error_state else "any"
        
        # 确定结果类型
        if result is None:
            result_type = None
        elif isinstance(result, dict):
            result_type = result.get("type", "unknown")
        else:
            result_type = "unknown"
        
        # 查找规则
        key = (state_type, action["type"], result_type)
        return cls.DECISION_TABLE.get(key, "default_transition")
```

---

## 第三部分：状态机实现与运行

### 3.1 状态机核心循环

#### **完整的状态机实现**
```python
class AgentStateMachine:
    """完整的Agent状态机实现"""
    
    def __init__(self, model, tools, config=None):
        self.model = model
        self.tools = tools
        self.config = config or {
            "max_steps": 10,
            "allow_ask_user": True,
            "allow_retry": True,
            "max_retries": 3
        }
        self.transition_rules = StateTransitionRules()
        
    def run_task(self, task):
        """执行一个完整任务"""
        # 初始化
        trajectory = []
        state = self._initialize_state(task)
        
        # 主循环
        while not self._should_terminate(state):
            # 记录步骤开始
            step_record = self._create_step_record(state)
            
            try:
                # 1. 准备模型输入
                model_input = self._prepare_model_input(state)
                step_record["model_input"] = model_input
                
                # 2. 模型决策
                raw_action = self._model_decide(model_input)
                action = self._parse_action(raw_action)
                step_record["raw_action"] = raw_action
                step_record["parsed_action"] = action
                
                # 3. 验证动作合法性
                if not self._validate_action(action, state):
                    raise InvalidActionError(action)
                
                # 4. 执行动作
                result = self._execute_action(action, state)
                step_record["execution_result"] = result
                
                # 5. 状态转移
                state = self._apply_transition(state, action, result)
                step_record["new_state"] = state
                
                # 6. 检查是否需要立即终止
                if self._requires_immediate_termination(state, action, result):
                    break
                    
            except Exception as e:
                # 错误处理
                error_record = self._handle_error(e, state, step_record)
                trajectory.append(error_record)
                
                # 判断是否可恢复
                if self._can_recover_from_error(e, state):
                    state = self._enter_recovery_state(state, e)
                    continue
                else:
                    # 不可恢复错误，终止
                    state = self._terminate_with_failure(state, e)
                    break
            
            # 记录成功步骤
            trajectory.append(step_record)
            state.internal.step_count += 1
        
        # 返回最终结果
        return {
            "trajectory": trajectory,
            "final_state": state,
            "success": state.internal.get("success", False),
            "termination_reason": state.internal.get("termination_reason"),
            "failure_types": state.failure_labels
        }
    
    def _prepare_model_input(self, state):
        """准备模型输入状态"""
        # 关键：只包含运行时状态，不包含标注信息
        return {
            "messages": state.messages,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema
                }
                for tool in self.tools if tool.name in state.available_tools
            ]
        }
    
    def _model_decide(self, model_input):
        """模型决策"""
        # 实际调用模型API
        response = self.model.generate(
            messages=model_input["messages"],
            tools=model_input["tools"],
            tool_choice="auto"  # 让模型决定是否调用工具
        )
        return response
    
    def _parse_action(self, raw_response):
        """解析模型原始输出为结构化动作"""
        try:
            # 尝试解析为JSON
            if hasattr(raw_response, "tool_calls") and raw_response.tool_calls:
                # 工具调用
                tool_call = raw_response.tool_calls[0]
                return {
                    "type": "tool_call",
                    "tool_name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "id": tool_call.id
                }
            elif hasattr(raw_response, "content"):
                # 文本回复，需要进一步判断
                content = raw_response.content
                if self._looks_like_final_answer(content):
                    return {
                        "type": "final_answer",
                        "content": content
                    }
                elif self._looks_like_ask_user(content):
                    return {
                        "type": "ask_user",
                        "content": content
                    }
                else:
                    # 无法解析，作为无效动作
                    return None
        except Exception:
            return None
    
    def _execute_action(self, action, state):
        """执行动作"""
        if action["type"] == "tool_call":
            return self._execute_tool_call(action, state)
        elif action["type"] == "final_answer":
            return self._validate_final_answer(action, state)
        elif action["type"] == "ask_user":
            return self._handle_ask_user(action, state)
        else:
            raise UnknownActionTypeError(action["type"])
    
    def _apply_transition(self, state, action, result):
        """应用状态转移"""
        # 获取适用的转移规则
        rule_name = TransitionDecisionTable.get_transition_rule(state, action, result)
        
        # 应用转移规则
        rule_method = getattr(self.transition_rules, rule_name)
        new_state = rule_method(state, action, result)
        
        return new_state
```

### 3.2 错误处理与恢复机制

#### **错误分类与处理策略**
```python
class ErrorHandler:
    """错误处理器"""
    
    ERROR_HANDLING_STRATEGIES = {
        # 错误类型: (是否可重试, 最大重试次数, 处理策略)
        "schema_error": (True, 2, "prompt_for_correction"),
        "missing_argument": (True, 2, "extract_from_context_or_ask"),
        "wrong_argument_type": (True, 1, "type_conversion"),
        "tool_not_found": (False, 0, "terminate_with_explanation"),
        "permission_denied": (False, 0, "terminate_or_ask_for_auth"),
        "rate_limited": (True, 1, "wait_and_retry"),
        "timeout": (True, 2, "retry_or_fallback"),
        "empty_result": (True, 1, "broaden_query_or_explain"),
        "internal_error": (True, 1, "retry_or_use_alternative"),
    }
    
    def handle_error(self, error, state, action):
        """处理错误并决定恢复策略"""
        # 1. 错误分类
        error_type = self._classify_error(error)
        
        # 2. 获取处理策略
        retryable, max_retries, strategy = self.ERROR_HANDLING_STRATEGIES.get(
            error_type, (False, 0, "terminate")
        )
        
        # 3. 检查重试次数
        current_attempts = state.progress.error_state.get("attempt_count", 0) if state.progress.error_state else 0
        can_retry = retryable and current_attempts < max_retries
        
        # 4. 生成恢复动作
        if can_retry:
            recovery_action = self._generate_recovery_action(error, state, action, strategy)
            return {
                "should_continue": True,
                "recovery_action": recovery_action,
                "error_type": error_type,
                "attempt": current_attempts + 1
            }
        else:
            # 超过最大重试次数或不可重试
            return {
                "should_continue": False,
                "termination_reason": f"{error_type}_unrecoverable",
                "error_type": error_type,
                "final_attempt": True
            }
    
    def _generate_recovery_action(self, error, state, failed_action, strategy):
        """生成恢复动作"""
        if strategy == "prompt_for_correction":
            # 提示模型修正参数
            error_message = self._format_schema_error_message(error)
            return {
                "type": "system_prompt",
                "content": f"上次工具调用失败：{error_message}。请修正参数后重试。",
                "suggested_corrections": error.suggestions
            }
        elif strategy == "extract_from_context_or_ask":
            # 尝试从上下文中提取缺失参数
            missing_arg = error.missing_argument
            extracted = self._extract_from_context(missing_arg, state)
            
            if extracted:
                # 自动补全参数重试
                corrected_action = failed_action.copy()
                corrected_action["arguments"][missing_arg] = extracted
                return {
                    "type": "auto_correction",
                    "corrected_action": corrected_action,
                    "source": "context_extraction"
                }
            else:
                # 需要反问用户
                return {
                    "type": "ask_user_template",
                    "question": f"请提供{missing_arg}信息",
                    "clarification_type": "missing_param"
                }
        # ... 其他策略
```

### 3.3 终止条件判断

#### **多层次终止判断**
```python
class TerminationChecker:
    """终止条件检查器"""
    
    def should_terminate(self, state, action=None, result=None):
        """检查是否应该终止"""
        termination_checks = [
            self._check_max_steps,
            self._check_termination_action,
            self._check_unrecoverable_error,
            self._check_task_completion,
            self._check_infinite_loop,
            self._check_user_timeout,
        ]
        
        for check in termination_checks:
            should_terminate, reason = check(state, action, result)
            if should_terminate:
                return True, reason
        
        return False, None
    
    def _check_max_steps(self, state, action, result):
        """检查是否超过最大步数"""
        if state.internal.step_count >= state.internal.max_steps:
            return True, "max_steps_exceeded"
        return False, None
    
    def _check_termination_action(self, state, action, result):
        """检查是否为终止性动作"""
        if action and action["type"] == "final_answer":
            # 最终回答需要进一步验证
            if self._is_final_answer_valid(state, action):
                return True, "final_answer"
            else:
                # 无效的最终回答，但不一定终止（可能让模型继续）
                return False, "invalid_final_answer_continue"
        
        elif action and action["type"] == "ask_user":
            # 反问用户：单轮任务中终止，多轮任务中继续
            if not state.task.allow_multi_turn:
                return True, "ask_user_in_single_turn"
            else:
                return False, "ask_user_in_multi_turn"
        
        return False, None
    
    def _check_unrecoverable_error(self, state, action, result):
        """检查不可恢复错误"""
        if state.progress.error_state and not state.progress.error_state["retryable"]:
            if state.progress.error_state["attempt_count"] >= state.progress.error_state["max_attempts"]:
                return True, "unrecoverable_error"
        return False, None
    
    def _check_task_completion(self, state, action, result):
        """检查任务是否已完成"""
        if not state.progress.open_requirements:
            # 没有待完成需求，但模型没有输出final_answer
            # 可以等待模型输出，或主动提示
            return False, "task_complete_waiting_for_final"
        return False, None
    
    def _check_infinite_loop(self, state, action, result):
        """检查是否陷入死循环"""
        # 检查最近N步是否重复
        recent_actions = state.messages[-10:]  # 最近10条消息
        if len(recent_actions) >= 5:
            # 简单检查：最近5个assistant消息是否相同
            assistant_messages = [msg for msg in recent_actions if msg["role"] == "assistant"]
            if len(assistant_messages) >= 3:
                last_three = assistant_messages[-3:]
                if all(self._are_actions_similar(last_three[0], msg) for msg in last_three[1:]):
                    return True, "infinite_loop_detected"
        return False, None
```

---

## 第四部分：状态机在训练中的应用

### 4.1 从状态机到训练数据

#### **轨迹数据的构造**
```python
def create_training_data_from_trajectory(trajectory):
    """从轨迹中提取训练数据"""
    training_samples = []
    
    for i, step in enumerate(trajectory["steps"]):
        # 1. 提取状态-动作对
        state = step["model_input"]  # 模型实际看到的输入
        expert_action = step["parsed_action"]  # 专家动作
        
        # 2. 创建训练样本
        sample = {
            "sample_id": f"{trajectory['task_id']}_step_{i}",
            "input": {
                # 关键：只包含运行时状态
                "messages": state["messages"],
                "tools": state["tools"]
            },
            "target": {
                "action": expert_action
            },
            "metadata": {
                "task_id": trajectory["task_id"],
                "step_index": i,
                "action_type": expert_action["type"],
                "trajectory_success": trajectory["success"],
                "failure_types": step.get("failure_types", [])
            }
        }
        
        training_samples.append(sample)
    
    return training_samples
```

#### **不同类型轨迹的处理**
```python
class TrainingDataProcessor:
    """训练数据处理"""
    
    def process_trajectories(self, trajectories):
        """处理轨迹数据，生成训练样本"""
        all_samples = []
        
        for traj in trajectories:
            if traj["success"]:
                # 成功轨迹：直接作为正样本
                samples = self._create_sft_samples(traj)
                all_samples.extend(samples)
                
            elif self._has_recoverable_errors(traj):
                # 可恢复错误轨迹：用于错误恢复训练
                recovery_samples = self._create_recovery_samples(traj)
                all_samples.extend(recovery_samples)
                
            elif self._has_interesting_failures(traj):
                # 有意义的失败轨迹：用于失败分析或对比学习
                analysis_samples = self._create_analysis_samples(traj)
                all_samples.extend(analysis_samples)
                
            else:
                # 无意义的失败轨迹：丢弃或用于其他用途
                pass
        
        return all_samples
    
    def _create_sft_samples(self, trajectory):
        """从成功轨迹创建SFT样本"""
        samples = create_training_data_from_trajectory(trajectory)
        
        # 验证样本质量
        validated_samples = []
        for sample in samples:
            if self._validate_sample(sample):
                validated_samples.append(sample)
        
        return validated_samples
    
    def _create_recovery_samples(self, trajectory):
        """从错误恢复轨迹创建训练样本"""
        recovery_samples = []
        
        # 找到错误步骤
        error_steps = [i for i, step in enumerate(trajectory["steps"]) 
                      if step.get("failure_types")]
        
        for error_step in error_steps:
            # 提取错误状态
            error_state = trajectory["steps"][error_step]["model_input"]
            
            # 提取修正后的动作（下一步的正确动作）
            if error_step + 1 < len(trajectory["steps"]):
                corrected_action = trajectory["steps"][error_step + 1]["parsed_action"]
                
                # 创建恢复训练样本
                sample = {
                    "sample_id": f"{trajectory['task_id']}_recovery_{error_step}",
                    "input": error_state,
                    "target": corrected_action,
                    "metadata": {
                        "task_id": trajectory["task_id"],
                        "error_step": error_step,
                        "error_type": trajectory["steps"][error_step]["failure_types"],
                        "recovery_type": "error_correction"
                    }
                }
                
                recovery_samples.append(sample)
        
        return recovery_samples
    
    def _validate_sample(self, sample):
        """验证样本质量"""
        # 1. 检查信息泄漏
        if self._has_leakage(sample["input"]):
            return False
        
        # 2. 检查动作合法性
        if not self._is_action_valid(sample["target"]["action"]):
            return False
        
        # 3. 检查状态-动作一致性
        if not self._are_state_action_consistent(sample["input"], sample["target"]["action"]):
            return False
        
        return True
```

### 4.2 训练目标设计

#### **基于状态-动作对的训练**
```python
class AgentTrainer:
    """Agent训练器"""
    
    def train_sft(self, training_samples, model, config):
        """监督微调训练"""
        
        # 数据准备
        dataloader = self._create_dataloader(training_samples)
        
        # 训练循环
        for epoch in range(config.num_epochs):
            for batch in dataloader:
                # 提取批处理数据
                states = batch["input"]  # 状态
                expert_actions = batch["target"]  # 专家动作
                
                # 模型前向传播
                logits = model(states)
                
                # 计算损失：预测动作 vs 专家动作
                loss = self._compute_action_loss(logits, expert_actions)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # 记录指标
                metrics = self._compute_metrics(logits, expert_actions)
                self._log_training_metrics(epoch, metrics)
    
    def _compute_action_loss(self, logits, expert_actions):
        """计算动作预测损失"""
        total_loss = 0
        
        # 1. 动作类型分类损失
        action_type_logits = logits["action_type"]
        expert_action_types = [act["type"] for act in expert_actions]
        type_loss = F.cross_entropy(action_type_logits, expert_action_types)
        total_loss += type_loss
        
        # 2. 工具选择损失（如果是工具调用）
        tool_mask = [act["type"] == "tool_call" for act in expert_actions]
        if any(tool_mask):
            tool_logits = logits["tool_name"][tool_mask]
            expert_tools = [act["tool_name"] for act, mask in zip(expert_actions, tool_mask) if mask]
            tool_loss = F.cross_entropy(tool_logits, expert_tools)
            total_loss += tool_loss
        
        # 3. 参数生成损失
        # 使用token级损失或结构化损失
        
        return total_loss
    
    def train_with_rejection_sampling(self, model, task_pool, evaluator, config):
        """拒绝采样训练"""
        accepted_trajectories = []
        
        for iteration in range(config.num_iterations):
            # 1. 用当前策略采样轨迹
            trajectories = self._sample_trajectories(model, task_pool, config.num_samples)
            
            # 2. 评估轨迹质量
            scored_trajectories = []
            for traj in trajectories:
                score = evaluator.evaluate(traj)
                scored_trajectories.append((traj, score))
            
            # 3. 根据分数筛选
            scored_trajectories.sort(key=lambda x: x[1], reverse=True)
            accepted = scored_trajectories[:int(len(scored_trajectories) * config.acceptance_rate)]
            
            # 4. 创建训练数据
            training_samples = []
            for traj, score in accepted:
                samples = create_training_data_from_trajectory(traj)
                # 可选：根据分数加权样本
                weighted_samples = self._weight_samples_by_score(samples, score)
                training_samples.extend(weighted_samples)
            
            # 5. 训练模型
            if training_samples:
                self.train_sft(training_samples, model, config.epochs_per_iteration)
            
            accepted_trajectories.extend([traj for traj, _ in accepted])
        
        return accepted_trajectories
```

---

## 第五部分：状态机在评估中的应用

### 5.1 分层评估体系

#### **基于状态机的评估框架**
```python
class StateBasedEvaluator:
    """基于状态的评估器"""
    
    def evaluate_trajectory(self, trajectory, reference):
        """评估完整轨迹"""
        
        # 1. 单步评估
        step_evaluations = []
        for i, step in enumerate(trajectory["steps"]):
            step_eval = self._evaluate_step(step, reference["steps"][i] if i < len(reference["steps"]) else None)
            step_evaluations.append(step_eval)
        
        # 2. 流程评估
        process_evaluation = self._evaluate_process(trajectory, reference, step_evaluations)
        
        # 3. 结果评估
        result_evaluation = self._evaluate_result(trajectory, reference)
        
        # 4. 恢复能力评估
        recovery_evaluation = self._evaluate_recovery(trajectory)
        
        # 5. 综合评分
        final_score = self._aggregate_scores(
            step_evaluations,
            process_evaluation,
            result_evaluation,
            recovery_evaluation
        )
        
        return {
            "step_evaluations": step_evaluations,
            "process_evaluation": process_evaluation,
            "result_evaluation": result_evaluation,
            "recovery_evaluation": recovery_evaluation,
            "final_score": final_score,
            "failure_types": self._extract_failure_types(step_evaluations, process_evaluation)
        }
    
    def _evaluate_step(self, step, reference_step):
        """评估单步"""
        evaluation = {
            "step_index": step["step"],
            "action_correctness": None,
            "state_appropriateness": None,
            "observation_usage": None,
            "failure_types": []
        }
        
        # 1. 动作正确性评估
        if reference_step:
            evaluation["action_correctness"] = self._compare_actions(
                step["parsed_action"],
                reference_step["action"]
            )
        
        # 2. 状态适当性评估
        evaluation["state_appropriateness"] = self._evaluate_state_appropriateness(
            step["model_input"],
            step["parsed_action"]
        )
        
        # 3. 观察使用评估
        if step.get("execution_result"):
            evaluation["observation_usage"] = self._evaluate_observation_usage(
                step["parsed_action"],
                step["execution_result"],
                step.get("next_action")
            )
        
        # 4. 失败类型检测
        evaluation["failure_types"] = self._detect_failure_types(step, reference_step)
        
        return evaluation
    
    def _evaluate_process(self, trajectory, reference, step_evaluations):
        """评估流程合理性"""
        process_eval = {
            "step_order_correct": True,
            "dependency_satisfied": True,
            "efficiency_score": 1.0,
            "redundancy_score": 0.0,
            "process_failure_types": []
        }
        
        # 1. 检查步骤顺序
        if reference:
            process_eval["step_order_correct"] = self._check_step_order(
                [s["parsed_action"] for s in trajectory["steps"]],
                [s["action"] for s in reference["steps"]]
            )
        
        # 2. 检查依赖关系
        process_eval["dependency_satisfied"] = self._check_dependencies(trajectory)
        
        # 3. 评估效率
        process_eval["efficiency_score"] = self._calculate_efficiency(trajectory, reference)
        
        # 4. 检查冗余
        process_eval["redundancy_score"] = self._calculate_redundancy(trajectory)
        
        # 5. 流程级失败检测
        process_eval["process_failure_types"] = self._detect_process_failures(trajectory)
        
        return process_eval
    
    def _evaluate_recovery(self, trajectory):
        """评估恢复能力"""
        recovery_eval = {
            "error_detection_rate": 0.0,
            "recovery_success_rate": 0.0,
            "recovery_efficiency": 0.0,
            "escalation_appropriateness": 0.0
        }
        
        # 找到错误步骤
        error_steps = [i for i, step in enumerate(trajectory["steps"]) 
                      if step.get("failure_types")]
        
        if not error_steps:
            # 没有错误，恢复能力满分
            recovery_eval = {k: 1.0 for k in recovery_eval}
            return recovery_eval
        
        # 评估每个错误的恢复
        recovery_results = []
        for error_step in error_steps:
            result = self._evaluate_single_recovery(trajectory, error_step)
            recovery_results.append(result)
        
        # 聚合恢复评估
        recovery_eval["error_detection_rate"] = sum(r["detected"] for r in recovery_results) / len(recovery_results)
        recovery_eval["recovery_success_rate"] = sum(r["recovered"] for r in recovery_results) / len(recovery_results)
        recovery_eval["recovery_efficiency"] = sum(r["efficiency"] for r in recovery_results) / len(recovery_results)
        recovery_eval["escalation_appropriateness"] = sum(r["escalation_ok"] for r in recovery_results) / len(recovery_results)
        
        return recovery_eval
```

### 5.2 失败分析与诊断

#### **基于状态机的失败诊断**
```python
class FailureDiagnoser:
    """失败诊断器"""
    
    def diagnose_failures(self, trajectory, evaluation_results):
        """诊断失败原因"""
        diagnoses = []
        
        # 1. 单步失败诊断
        for step_eval in evaluation_results["step_evaluations"]:
            if step_eval["failure_types"]:
                diagnosis = self._diagnose_step_failure(
                    trajectory["steps"][step_eval["step_index"]],
                    step_eval
                )
                diagnoses.append(diagnosis)
        
        # 2. 流程失败诊断
        if evaluation_results["process_evaluation"]["process_failure_types"]:
            process_diagnosis = self._diagnose_process_failure(
                trajectory,
                evaluation_results["process_evaluation"]
            )
            diagnoses.append(process_diagnosis)
        
        # 3. 恢复失败诊断
        recovery_score = evaluation_results["recovery_evaluation"]["recovery_success_rate"]
        if recovery_score < 0.5:  # 恢复成功率低
            recovery_diagnosis = self._diagnose_recovery_failure(trajectory)
            diagnoses.append(recovery_diagnosis)
        
        # 4. 根本原因分析
        root_causes = self._identify_root_causes(diagnoses)
        
        return {
            "diagnoses": diagnoses,
            "root_causes": root_causes,
            "recommendations": self._generate_recommendations(diagnoses, root_causes)
        }
    
    def _diagnose_step_failure(self, step, step_evaluation):
        """诊断单步失败"""
        diagnosis = {
            "step_index": step["step"],
            "state_at_failure": step["model_input"],
            "action_taken": step["parsed_action"],
            "failure_types": step_evaluation["failure_types"],
            "possible_causes": [],
            "suggested_fixes": []
        }
        
        # 根据失败类型分析可能原因
        for failure_type in step_evaluation["failure_types"]:
            causes = self._get_possible_causes(failure_type, step)
            fixes = self._get_suggested_fixes(failure_type, step)
            
            diagnosis["possible_causes"].extend(causes)
            diagnosis["suggested_fixes"].extend(fixes)
        
        return diagnosis
    
    def _get_possible_causes(self, failure_type, step):
        """获取失败的可能原因"""
        causes_map = {
            "wrong_tool": [
                "模型不理解工具功能",
                "工具描述不清晰",
                "训练数据中缺少类似场景",
                "状态信息不足，无法正确选择工具"
            ],
            "missing_argument": [
                "用户查询中未提供该信息",
                "模型未能从上下文中提取信息",
                "参数提取逻辑有误",
                "工具schema要求过于严格"
            ],
            "invalid_schema": [
                "参数类型错误",
                "参数格式不符合要求",
                "模型未遵循工具schema",
                "schema定义不清晰"
            ],
            "premature_final_answer": [
                "模型过早判断任务完成",
                "状态中缺少任务完成度的表示",
                "训练数据中过早终止的样本",
                "模型过于自信"
            ],
            # ... 其他失败类型
        }
        
        return causes_map.get(failure_type, ["未知原因"])
    
    def _generate_recommendations(self, diagnoses, root_causes):
        """生成改进建议"""
        recommendations = {
            "data_collection": [],
            "model_training": [],
            "system_design": [],
            "evaluation_refinement": []
        }
        
        # 分析失败模式
        failure_patterns = self._analyze_failure_patterns(diagnoses)
        
        # 生成针对性建议
        for pattern, frequency in failure_patterns.items():
            if pattern == "wrong_tool_selection":
                recommendations["data_collection"].append(
                    "收集更多工具选择场景的训练数据，特别是边界情况"
                )
                recommendations["model_training"].append(
                    "增加工具选择准确性的专项训练"
                )
            elif pattern == "parameter_extraction":
                recommendations["system_design"].append(
                    "改进参数提取逻辑，增加上下文信息利用"
                )
                recommendations["data_collection"].append(
                    "收集参数提取困难的案例进行标注"
                )
            # ... 其他模式
        
        return recommendations
```

---

## 第六部分：实践指南与最佳实践

### 6.1 状态机设计的最佳实践

#### **设计原则**
```python
class StateMachineDesignPrinciples:
    """状态机设计原则"""
    
    PRINCIPLES = {
        # 1. 状态最小化原则
        "state_minimization": """
        状态应该包含完成任务所需的最小信息集。
        避免存储冗余或推导性信息。
        示例：存储原始工具结果，而不是预处理后的多个版本。
        """,
        
        # 2. 接口清晰原则  
        "clear_interfaces": """
        状态转移函数的输入输出应该清晰明确。
        每个模块的职责边界要分明。
        示例：状态管理器只负责状态更新，不负责工具执行。
        """,
        
        # 3. 错误隔离原则
        "error_isolation": """
        错误应该被限制在发生的地方，不影响其他状态。
        每个错误类型应该有明确的恢复路径。
        示例：工具执行错误只影响该工具相关状态，不影响整体任务状态。
        """,
        
        # 4. 可观测性原则
        "observability": """
        状态应该是可观测和可调试的。
        关键状态变化应该有日志记录。
        示例：记录每个状态转移的原因和结果。
        """,
        
        # 5. 可扩展性原则
        "extensibility": """
        状态设计应该支持未来扩展。
        预留字段或使用灵活的数据结构。
        示例：使用字典存储扩展字段，而不是固定结构。
        """
    }
```

#### **实现模式**
```python
class StateMachineImplementationPatterns:
    """状态机实现模式"""
    
    @staticmethod
    def pattern_simple_loop():
        """简单循环模式：适用于基础Agent"""
        state = initialize()
        
        while not should_terminate(state):
            # 准备-决策-执行-更新循环
            input_state = prepare_input(state)
            action = model_decide(input_state)
            result = execute_action(action)
            state = update_state(state, action, result)
        
        return get_result(state)
    
    @staticmethod  
    def pattern_hierarchical_states():
        """分层状态模式：适用于复杂Agent"""
        # 顶层状态机管理任务流程
        top_level_state = initialize_top_level()
        
        while not top_level_complete(top_level_state):
            # 根据当前阶段调用子状态机
            current_phase = top_level_state.current_phase
            
            if current_phase == "information_gathering":
                sub_state = information_gathering_phase(top_level_state)
                top_level_state = update_after_information_gathering(sub_state)
                
            elif current_phase == "analysis":
                sub_state = analysis_phase(top_level_state)
                top_level_state = update_after_analysis(sub_state)
                
            elif current_phase == "synthesis":
                sub_state = synthesis_phase(top_level_state)
                top_level_state = update_after_synthesis(sub_state)
    
    @staticmethod
    def pattern_event_driven():
        """事件驱动模式：适用于响应式Agent"""
        # 注册事件处理器
        event_handlers = {
            "user_input": handle_user_input,
            "tool_result": handle_tool_result,
            "error": handle_error,
            "timeout": handle_timeout
        }
        
        # 事件循环
        while True:
            event = wait_for_event()
            handler = event_handlers.get(event.type)
            if handler:
                handler(event, current_state)
```

### 6.2 调试与优化指南

#### **调试检查清单**
```python
class StateMachineDebugChecklist:
    """状态机调试检查清单"""
    
    @classmethod
    def check_state_leakage(cls, state):
        """检查状态信息泄漏"""
        issues = []
        
        # 1. 检查是否包含标注信息
        if hasattr(state, "expected_next_action"):
            issues.append("泄漏：包含expected_next_action")
        
        if hasattr(state, "reference_answer"):
            issues.append("泄漏：包含reference_answer")
        
        # 2. 检查是否包含未来信息
        if hasattr(state, "final_outcome"):
            issues.append("泄漏：包含final_outcome")
        
        # 3. 检查是否包含评估信息
        if hasattr(state, "evaluation_score"):
            issues.append("泄漏：包含evaluation_score")
        
        return issues
    
    @classmethod
    def check_state_consistency(cls, trajectory):
        """检查状态一致性"""
        issues = []
        
        for i in range(1, len(trajectory["steps"])):
            current_state = trajectory["steps"][i]["model_input"]
            previous_state = trajectory["steps"][i-1]["model_input"]
            action = trajectory["steps"][i-1]["parsed_action"]
            result = trajectory["steps"][i-1].get("execution_result")
            
            # 模拟状态转移
            expected_state = simulate_transition(previous_state, action, result)
            
            # 比较实际状态与预期状态
            if not states_equal(current_state, expected_state):
                issues.append(f"步骤{i}：状态不一致")
        
        return issues
    
    @classmethod
    def check_error_recovery(cls, trajectory):
        """检查错误恢复逻辑"""
        issues = []
        
        error_steps = [i for i, step in enumerate(trajectory["steps"]) 
                      if step.get("failure_types")]
        
        for error_step in error_steps:
            # 检查是否有恢复尝试
            if error_step + 1 >= len(trajectory["steps"]):
                issues.append(f"步骤{error_step}：错误后没有恢复尝试")
                continue
            
            next_action = trajectory["steps"][error_step + 1]["parsed_action"]
            error = trajectory["steps"][error_step].get("execution_result")
            
            # 检查恢复动作是否合理
            if not is_recovery_action_reasonable(next_action, error):
                issues.append(f"步骤{error_step}：恢复动作不合理")
        
        return issues
```

#### **性能优化策略**
```python
class StateMachineOptimization:
    """状态机优化策略"""
    
    @staticmethod
    def optimize_state_representation(state):
        """优化状态表示"""
        optimized = {}
        
        # 1. 压缩消息历史
        if len(state.messages) > 20:
            # 保留最近消息和关键早期消息
            optimized["messages"] = compress_messages(state.messages)
        else:
            optimized["messages"] = state.messages
        
        # 2. 简化工具描述
        optimized["tools"] = [
            {
                "name": tool["name"],
                # 只保留必要信息
                "description": tool.get("description", "")[:100],  # 截断长描述
                "input_schema": simplify_schema(tool.get("input_schema", {}))
            }
            for tool in state.get("tools", [])
        ]
        
        # 3. 移除不必要字段
        for key in ["metadata", "debug_info", "temporary_data"]:
            if key in state:
                del optimized[key]
        
        return optimized
    
    @staticmethod
    def cache_common_states(state_machine):
        """缓存常见状态"""
        cache = {}
        
        def cached_prepare_state(state):
            # 生成状态指纹
            fingerprint = hash_state(state)
            
            if fingerprint in cache:
                return cache[fingerprint]
            else:
                prepared = state_machine._prepare_state(state)
                cache[fingerprint] = prepared
                return prepared
        
        return cached_prepare_state
    
    @staticmethod
    def batch_tool_executions(actions, state):
        """批量执行工具调用"""
        # 分组工具调用
        tool_calls_by_type = {}
        for action in actions:
            if action["type"] == "tool_call":
                tool_name = action["tool_name"]
                if tool_name not in tool_calls_by_type:
                    tool_calls_by_type[tool_name] = []
                tool_calls_by_type[tool_name].append(action)
        
        # 批量执行
        results = {}
        for tool_name, calls in tool_calls_by_type.items():
            if can_batch_execute(tool_name):
                batch_result = execute_tool_batch(tool_name, calls)
                for i, call in enumerate(calls):
                    results[call["id"]] = batch_result[i]
            else:
                # 无法批量，单独执行
                for call in calls:
                    results[call["id"]] = execute_tool(call)
        
        return results
```

---

## 第七部分：总结与展望

### 7.1 核心要点回顾

#### **状态机在Agent开发中的核心作用**
1. **流程管理**：定义Agent如何逐步完成任务
2. **状态维护**：记录历史信息和当前进度
3. **错误处理**：提供系统化的恢复机制
4. **训练基础**：为监督学习提供状态-动作对
5. **评估框架**：支持分层评估和失败诊断

#### **关键设计决策**
```python
# 状态机设计的五个关键决策点
key_decisions = {
    "state_representation": "状态应该包含什么信息？",
    "action_space": "Agent可以执行哪些动作？",
    "transition_rules": "动作如何改变状态？",
    "termination_conditions": "什么时候应该终止？",
    "error_handling": "错误后如何恢复？"
}
```

### 7.2 实际应用建议

#### **新项目启动步骤**
1. **定义最小可行状态机**：从最简单的循环开始
2. **实现核心流程**：准备-决策-执行-更新循环
3. **添加错误处理**：基础错误恢复机制
4. **集成评估**：实现基本评估框架
5. **迭代优化**：基于实际使用反馈改进

#### **常见陷阱与避免方法**
```python
common_pitfalls = {
    "状态爆炸": "定期清理和压缩状态信息",
    "信息泄漏": "严格分离运行时状态和标注状态",
    "死循环": "实现循环检测和最大步数限制",
    "错误传播": "错误应该被隔离和处理，而不是传播",
    "过度设计": "从简单开始，按需增加复杂度"
}
```

### 7.3 进阶发展方向

#### **状态机的扩展与优化**
```python
future_directions = {
    "分层状态机": "支持更复杂的任务分解",
    "自适应状态机": "根据任务复杂度调整状态粒度",
    "分布式状态机": "支持跨多个Agent的状态协调",
    "增量学习状态机": "支持在线学习和状态机优化",
    "形式化验证": "使用形式化方法验证状态机正确性"
}
```

#### **与其他技术的结合**
```python
integration_opportunities = {
    "强化学习": "使用状态作为RL的状态表示",
    "课程学习": "基于状态复杂度设计学习课程",
    "元学习": "学习状态转移策略",
    "多模态": "扩展状态包含图像、音频等信息",
    "长期记忆": "集成外部记忆系统"
}
```

---

## 📚 附录：快速参考指南

### A. 状态机核心组件速查

```python
# 1. 状态定义模板
state_template = {
    "messages": [],      # 对话历史
    "tools": [],         # 可用工具
    "internal": {        # 内部状态（不暴露给模型）
        "step_count": 0,
        "terminated": False,
        "failure_types": []
    }
}

# 2. 动作定义模板
action_templates = {
    "tool_call": {
        "type": "tool_call",
        "tool_name": "string",
        "arguments": {},
        "id": "optional_string"
    },
    "final_answer": {
        "type": "final_answer", 
        "content": "string"
    },
    "ask_user": {
        "type": "ask_user",
        "content": "string"
    }
}

# 3. 状态转移函数模板
def transition_template(state, action, result):
    """状态转移函数模板"""
    new_state = state.copy()
    
    # 更新消息历史
    new_state["messages"].append(create_assistant_message(action))
    if result:
        new_state["messages"].append(create_tool_message(action, result))
    
    # 更新内部状态
    new_state["internal"]["step_count"] += 1
    
    # 根据动作类型特殊处理
    if action["type"] == "final_answer":
        new_state["internal"]["terminated"] = True
    
    return new_state
```

### B. 调试命令示例

```bash
# 1. 检查状态泄漏
python -m debug.state_leakage trajectory.json

# 2. 验证状态一致性
python -m debug.state_consistency trajectory.json

# 3. 模拟状态转移
python -m debug.simulate_transition --state state.json --action action.json

# 4. 性能分析
python -m cProfile -o profile.stats agent_state_machine.py
```

### C. 推荐的学习路径

1. **第一周**：理解状态、动作、转移的基本概念
2. **第二周**：实现简单的状态机循环
3. **第三周**：添加错误处理和恢复机制
4. **第四周**：集成评估和调试工具
5. **第五周**：优化性能和扩展功能

---

**最后提醒**：状态机是Agent系统的骨架，但不是全部。一个好的Agent系统还需要：
- 高质量的训练数据
- 强大的基础模型
- 有效的评估体系
- 用户友好的交互界面

状态机提供的是**结构和流程**，而上述元素提供的是**内容和质量**。两者结合，才能构建出真正有用的Agent系统。