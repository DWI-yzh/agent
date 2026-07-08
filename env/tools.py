"""Deterministic mock tools for the Agent sandbox."""

from __future__ import annotations

import ast
from datetime import date, datetime, timedelta
import hashlib
from typing import Any


PERMISSION_DENIED = "PERMISSION_DENIED"
TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"


class ToolError(Exception):
    def __init__(self, error_code: str, error_message: str) -> None:
        super().__init__(error_message)
        self.error_code = error_code
        self.error_message = error_message


DOCUMENTS = [
    {
        "doc_id": "DOC-001",
        "title": "Agent sandbox design",
        "body": "A deterministic environment for action execution and observation feedback.",
    },
    {
        "doc_id": "DOC-002",
        "title": "Schema validation guide",
        "body": "Schemas define required fields, field formats, and allowed value ranges.",
    },
    {
        "doc_id": "DOC-003",
        "title": "Recovery policy",
        "body": "Agents should fix schema errors, retry transient failures, and stop on permission errors.",
    },
]

WEATHER = {
    ("shanghai", "today"): {"city": "Shanghai", "date": "today", "temperature_c": 28, "condition": "cloudy"},
    ("shanghai", "tomorrow"): {"city": "Shanghai", "date": "tomorrow", "temperature_c": 30, "condition": "sunny"},
    ("beijing", "today"): {"city": "Beijing", "date": "today", "temperature_c": 26, "condition": "clear"},
    ("beijing", "tomorrow"): {"city": "Beijing", "date": "tomorrow", "temperature_c": 25, "condition": "rain"},
}

ORDERS = {
    "ORD-1001": {"order_id": "ORD-1001", "status": "paid", "customer_id": "CUST-001", "total": 299.0},
    "ORD-1002": {"order_id": "ORD-1002", "status": "shipped", "customer_id": "CUST-002", "total": 129.5},
}

CUSTOMERS = {
    "CUST-001": {"customer_id": "CUST-001", "name": "Alice Chen", "tier": "gold"},
    "CUST-002": {"customer_id": "CUST-002", "name": "Bob Li", "tier": "standard"},
}

RATES_TO_USD = {
    "USD": 1.0,
    "CNY": 1 / 7.2,
    "EUR": 1.08,
    "JPY": 1 / 155.0,
}

SANDBOX_TODAY = date(2026, 7, 8)


def search_doc(keyword: str, topk: int) -> dict[str, Any]:
    needle = keyword.casefold()
    matches = [
        doc
        for doc in DOCUMENTS
        if needle in doc["title"].casefold() or needle in doc["body"].casefold()
    ]
    return {"items": matches[:topk], "count": len(matches[:topk])}


def get_weather(city: str, date: str) -> dict[str, Any]:
    normalized_city = _normalize_city(city)
    normalized_date = _normalize_relative_date(date)
    return WEATHER.get((normalized_city, normalized_date), {})


def get_order(order_id: str) -> dict[str, Any]:
    return ORDERS.get(order_id, {})


def lookup_customer(customer_id: str) -> dict[str, Any]:
    return CUSTOMERS.get(customer_id, {})


def create_ticket(title: str, priority: str, assignee: str) -> dict[str, Any]:
    ticket_id = _stable_id("TCK", title, priority, assignee)
    return {
        "ticket_id": ticket_id,
        "title": title,
        "priority": priority,
        "assignee": assignee,
        "status": "open",
    }


def send_email(to: str, subject: str, body: str) -> dict[str, Any]:
    domain = to.rsplit("@", 1)[-1].lower()
    if domain not in {"example.com", "internal.local"}:
        raise ToolError(PERMISSION_DENIED, f"Email domain is not allowed: {domain}")

    return {
        "message_id": _stable_id("MSG", to, subject, body),
        "to": to,
        "subject": subject,
        "status": "queued",
    }


def schedule_meeting(date: str, attendees: list[str]) -> dict[str, Any]:
    return {
        "meeting_id": _stable_id("MTG", date, ",".join(sorted(attendees))),
        "date": date,
        "attendees": attendees,
        "status": "scheduled",
    }


def calculator(expr: str) -> dict[str, Any]:
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_ast(tree.body)
    except Exception as exc:  # noqa: BLE001 - convert every calculator failure into a tool error.
        raise ToolError(TOOL_EXECUTION_ERROR, f"Invalid expression: {exc}") from exc

    return {"expr": expr, "value": value}


def date_convert(text_date: str, format: str) -> dict[str, Any]:
    parsed = _parse_date(text_date)
    if parsed is None:
        return {}

    if format == "iso":
        converted = parsed.isoformat()
    elif format == "us":
        converted = parsed.strftime("%m/%d/%Y")
    elif format == "cn":
        converted = parsed.strftime("%Y-%m-%d")
    else:
        raise ToolError(TOOL_EXECUTION_ERROR, f"Unsupported date format: {format}")

    return {"input": text_date, "format": format, "date": converted}


def currency_convert(amount: float, from_: str, to: str) -> dict[str, Any]:
    usd = amount * RATES_TO_USD[from_]
    converted = usd / RATES_TO_USD[to]
    return {
        "amount": amount,
        "from": from_,
        "to": to,
        "converted": round(converted, 2),
    }


TOOLS = {
    "search_doc": search_doc,
    "get_weather": get_weather,
    "get_order": get_order,
    "lookup_customer": lookup_customer,
    "create_ticket": create_ticket,
    "send_email": send_email,
    "schedule_meeting": schedule_meeting,
    "calculator": calculator,
    "date_convert": date_convert,
    "currency_convert": currency_convert,
}


def call_tool(tool_name: str, args: dict[str, Any]) -> Any:
    tool = TOOLS[tool_name]
    if tool_name == "currency_convert":
        return tool(args["amount"], args["from"], args["to"])
    return tool(**args)


def _normalize_city(city: str) -> str:
    aliases = {
        "上海": "shanghai",
        "shanghai": "shanghai",
        "北京": "beijing",
        "beijing": "beijing",
    }
    return aliases.get(city.strip().casefold(), city.strip().casefold())


def _normalize_relative_date(text: str) -> str:
    if text in {"today", "tomorrow"}:
        return text
    return text


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8].upper()
    return f"{prefix}-{digest}"


def _parse_date(text: str) -> date | None:
    lowered = text.strip().lower()
    if lowered == "today":
        return SANDBOX_TODAY
    if lowered == "tomorrow":
        return SANDBOX_TODAY + timedelta(days=1)

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _eval_ast(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right

    raise ValueError("unsupported expression")
