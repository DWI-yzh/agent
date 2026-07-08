Week 2：搭建工具编排型 Agent sandbox

第二周的核心工作是：搭建工具编排型 Agent sandbox，也就是先做一个不依赖真实外部 API、完全可控、可复现、可评估的小型工具环境。这个阶段的重点不是训练模型，而是把后续训练、评测、错误诊断所需的“环境底座”搭起来。

1. 明确目标
   搭建一个可控工具环境，让 Agent 可以输出结构化 action，
   由 executor 调用工具，再返回标准化 observation。

2. 实现三类工具
   查询类：
   - search_doc(keyword, topk)
   - get_weather(city, date)
   - get_order(order_id)
   - lookup_customer(customer_id)

   操作类：
   - create_ticket(title, priority, assignee)
   - send_email(to, subject, body)
   - schedule_meeting(date, attendees)

   计算/转换类：
   - calculator(expr)
   - date_convert(text_date, format)
   - currency_convert(amount, from, to)

3. 必须完成三个核心文件
   env/tools.py
   - 定义每个工具的模拟逻辑

   env/schemas.py
   - 定义每个工具的参数 schema
   - 明确 required field、字段类型、取值范围

   env/executor.py
   - 统一执行入口
   - 负责解析 action
   - 校验 schema
   - 调用对应工具
   - 返回标准化结果

4. 标准返回格式
   {
       "status": "success" | "error",
       "tool": "get_weather",
       "args": {...},
       "result": {...},
       "error_code": None,
       "error_message": None
   }

5. 必须支持的错误类型
   - missing required field
   - invalid field format
   - unknown tool
   - empty result
   - permission denied
   - invalid value range

6. 本周交付物
   - env/tools.py
   - env/schemas.py
   - env/executor.py

7. 本周验收标准
   - 手动输入结构化 action 后可以稳定执行
   - 可以触发并捕获 schema 级错误
   - 每次执行都有统一日志输出