"""Tool input schemas and lightweight validation for the sandbox."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
INVALID_FIELD_FORMAT = "INVALID_FIELD_FORMAT"
INVALID_VALUE_RANGE = "INVALID_VALUE_RANGE"
UNKNOWN_TOOL = "UNKNOWN_TOOL"


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    error_code: str | None = None
    error_message: str | None = None


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_doc": {
        "required": ["keyword", "topk"],
        "properties": {
            "keyword": {"type": "string", "min_length": 1},
            "topk": {"type": "integer", "minimum": 1, "maximum": 5},
        },
    },
    "get_weather": {
        "required": ["city", "date"],
        "properties": {
            "city": {"type": "string", "min_length": 1},
            "date": {
                "type": "string",
                "pattern": r"^(today|tomorrow|\d{4}-\d{2}-\d{2})$",
            },
        },
    },
    "get_order": {
        "required": ["order_id"],
        "properties": {
            "order_id": {"type": "string", "pattern": r"^ORD-\d{4}$"},
        },
    },
    "lookup_customer": {
        "required": ["customer_id"],
        "properties": {
            "customer_id": {"type": "string", "pattern": r"^CUST-\d{3}$"},
        },
    },
    "create_ticket": {
        "required": ["title", "priority", "assignee"],
        "properties": {
            "title": {"type": "string", "min_length": 3, "max_length": 120},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
            "assignee": {"type": "string", "enum": ["alice", "bob", "carol", "ops"]},
        },
    },
    "send_email": {
        "required": ["to", "subject", "body"],
        "properties": {
            "to": {
                "type": "string",
                "pattern": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
            },
            "subject": {"type": "string", "min_length": 1, "max_length": 120},
            "body": {"type": "string", "min_length": 1, "max_length": 2000},
        },
    },
    "schedule_meeting": {
        "required": ["date", "attendees"],
        "properties": {
            "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
            "attendees": {
                "type": "array",
                "min_items": 1,
                "max_items": 10,
                "items": {"type": "string", "min_length": 1},
            },
        },
    },
    "calculator": {
        "required": ["expr"],
        "properties": {
            "expr": {
                "type": "string",
                "min_length": 1,
                "pattern": r"^[0-9+\-*/().\s%]+$",
            },
        },
    },
    "date_convert": {
        "required": ["text_date", "format"],
        "properties": {
            "text_date": {"type": "string", "min_length": 1},
            "format": {"type": "string", "enum": ["iso", "us", "cn"]},
        },
    },
    "currency_convert": {
        "required": ["amount", "from", "to"],
        "properties": {
            "amount": {"type": "number", "minimum": 0, "maximum": 100000},
            "from": {"type": "string", "enum": ["USD", "CNY", "EUR", "JPY"]},
            "to": {"type": "string", "enum": ["USD", "CNY", "EUR", "JPY"]},
        },
    },
}


def validate_tool_args(tool_name: str, args: Any) -> ValidationResult:
    """Validate action arguments against a tool schema."""

    schema = TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        return ValidationResult(False, UNKNOWN_TOOL, f"Unknown tool: {tool_name}")

    if not isinstance(args, dict):
        return ValidationResult(False, INVALID_FIELD_FORMAT, "args must be an object")

    properties = schema["properties"]
    for field in schema["required"]:
        if field not in args:
            return ValidationResult(
                False,
                MISSING_REQUIRED_FIELD,
                f"Missing required field: {field}",
            )

    for field in args:
        if field not in properties:
            return ValidationResult(
                False,
                INVALID_FIELD_FORMAT,
                f"Unknown argument for {tool_name}: {field}",
            )

    for field, rules in properties.items():
        if field in args:
            result = _validate_value(field, args[field], rules)
            if not result.ok:
                return result

    return ValidationResult(True)


def _validate_value(field: str, value: Any, rules: dict[str, Any]) -> ValidationResult:
    expected_type = rules["type"]

    if not _matches_type(value, expected_type):
        return ValidationResult(
            False,
            INVALID_FIELD_FORMAT,
            f"{field} must be {expected_type}",
        )

    if "enum" in rules and value not in rules["enum"]:
        return ValidationResult(
            False,
            INVALID_VALUE_RANGE,
            f"{field} must be one of {rules['enum']}",
        )

    if expected_type == "string":
        return _validate_string(field, value, rules)

    if expected_type in {"integer", "number"}:
        return _validate_number(field, value, rules)

    if expected_type == "array":
        return _validate_array(field, value, rules)

    return ValidationResult(True)


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    return False


def _validate_string(field: str, value: str, rules: dict[str, Any]) -> ValidationResult:
    min_length = rules.get("min_length")
    if min_length is not None and len(value.strip()) < min_length:
        return ValidationResult(
            False,
            INVALID_FIELD_FORMAT,
            f"{field} must not be empty",
        )

    max_length = rules.get("max_length")
    if max_length is not None and len(value) > max_length:
        return ValidationResult(
            False,
            INVALID_VALUE_RANGE,
            f"{field} must be at most {max_length} characters",
        )

    pattern = rules.get("pattern")
    if pattern and re.fullmatch(pattern, value) is None:
        return ValidationResult(
            False,
            INVALID_FIELD_FORMAT,
            f"{field} has invalid format",
        )

    return ValidationResult(True)


def _validate_number(field: str, value: int | float, rules: dict[str, Any]) -> ValidationResult:
    minimum = rules.get("minimum")
    if minimum is not None and value < minimum:
        return ValidationResult(
            False,
            INVALID_VALUE_RANGE,
            f"{field} must be >= {minimum}",
        )

    maximum = rules.get("maximum")
    if maximum is not None and value > maximum:
        return ValidationResult(
            False,
            INVALID_VALUE_RANGE,
            f"{field} must be <= {maximum}",
        )

    return ValidationResult(True)


def _validate_array(field: str, value: list[Any], rules: dict[str, Any]) -> ValidationResult:
    min_items = rules.get("min_items")
    if min_items is not None and len(value) < min_items:
        return ValidationResult(
            False,
            INVALID_VALUE_RANGE,
            f"{field} must contain at least {min_items} item(s)",
        )

    max_items = rules.get("max_items")
    if max_items is not None and len(value) > max_items:
        return ValidationResult(
            False,
            INVALID_VALUE_RANGE,
            f"{field} must contain at most {max_items} item(s)",
        )

    item_rules = rules.get("items")
    if item_rules:
        for index, item in enumerate(value):
            result = _validate_value(f"{field}[{index}]", item, item_rules)
            if not result.ok:
                return result

    return ValidationResult(True)
