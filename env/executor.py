"""Unified action executor for the Agent sandbox."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

try:
    from .schemas import UNKNOWN_TOOL, validate_tool_args
    from .tools import ToolError, call_tool
except ImportError:  # pragma: no cover - supports direct `python env/executor.py`.
    from schemas import UNKNOWN_TOOL, validate_tool_args
    from tools import ToolError, call_tool


INVALID_ACTION = "INVALID_ACTION"
EMPTY_RESULT = "EMPTY_RESULT"
SYSTEM_ERROR = "SYSTEM_ERROR"

LOGGER = logging.getLogger("agent_sandbox.executor")


def execute_action(action: dict[str, Any] | str) -> dict[str, Any]:
    """Parse, validate, execute, and observe one structured action."""

    parsed = _parse_action(action)
    if parsed["error_code"]:
        observation = _make_error(
            tool=parsed["tool"],
            args=parsed["args"],
            error_code=parsed["error_code"],
            error_message=parsed["error_message"],
        )
        _log("action_error", observation)
        return observation

    tool_name = parsed["tool"]
    args = parsed["args"]

    validation = validate_tool_args(tool_name, args)
    if not validation.ok:
        observation = _make_error(
            tool=tool_name,
            args=args,
            error_code=validation.error_code or SYSTEM_ERROR,
            error_message=validation.error_message or "Schema validation failed",
        )
        _log("schema_error", observation)
        return observation

    try:
        result = call_tool(tool_name, args)
    except KeyError:
        observation = _make_error(
            tool=tool_name,
            args=args,
            error_code=UNKNOWN_TOOL,
            error_message=f"Unknown tool: {tool_name}",
        )
        _log("unknown_tool", observation)
        return observation
    except ToolError as exc:
        observation = _make_error(
            tool=tool_name,
            args=args,
            error_code=exc.error_code,
            error_message=exc.error_message,
        )
        _log("tool_error", observation)
        return observation
    except Exception as exc:  # noqa: BLE001 - executor must standardize unexpected errors.
        observation = _make_error(
            tool=tool_name,
            args=args,
            error_code=SYSTEM_ERROR,
            error_message=str(exc),
        )
        _log("system_error", observation)
        return observation

    if _is_empty_result(result):
        observation = _make_error(
            tool=tool_name,
            args=args,
            error_code=EMPTY_RESULT,
            error_message="Tool returned an empty result",
        )
        _log("empty_result", observation)
        return observation

    observation = {
        "status": "success",
        "tool": tool_name,
        "args": args,
        "result": result,
        "error_code": None,
        "error_message": None,
    }
    _log("tool_success", observation)
    return observation


def _parse_action(action: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(action, str):
        try:
            action = json.loads(action)
        except json.JSONDecodeError as exc:
            return {
                "tool": None,
                "args": {},
                "error_code": INVALID_ACTION,
                "error_message": f"Action is not valid JSON: {exc.msg}",
            }

    if not isinstance(action, dict):
        return {
            "tool": None,
            "args": {},
            "error_code": INVALID_ACTION,
            "error_message": "Action must be an object or JSON object string",
        }

    tool_name = action.get("tool") or action.get("tool_name") or action.get("name")
    args = action.get("args")
    if args is None:
        args = action.get("arguments")

    if not tool_name:
        return {
            "tool": None,
            "args": args if isinstance(args, dict) else {},
            "error_code": INVALID_ACTION,
            "error_message": "Action must include tool/tool_name/name",
        }

    if args is None:
        return {
            "tool": tool_name,
            "args": {},
            "error_code": INVALID_ACTION,
            "error_message": "Action must include args or arguments",
        }

    if not isinstance(args, dict):
        return {
            "tool": tool_name,
            "args": {},
            "error_code": INVALID_ACTION,
            "error_message": "Action args must be an object",
        }

    return {"tool": tool_name, "args": args, "error_code": None, "error_message": None}


def _is_empty_result(result: Any) -> bool:
    if result is None:
        return True
    if result == {}:
        return True
    if result == []:
        return True
    if isinstance(result, dict) and result.get("count") == 0 and result.get("items") == []:
        return True
    return False


def _make_error(
    tool: str | None,
    args: dict[str, Any],
    error_code: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "tool": tool,
        "args": args,
        "result": None,
        "error_code": error_code,
        "error_message": error_message,
    }


def _log(event: str, observation: dict[str, Any]) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.info(
        json.dumps(
            {
                "event": event,
                "tool": observation["tool"],
                "status": observation["status"],
                "error_code": observation["error_code"],
            },
            ensure_ascii=False,
        )
    )


def main() -> int:
    raw = sys.stdin.read().strip()
    if not raw:
        print("Provide one JSON action on stdin.", file=sys.stderr)
        return 2

    observation = execute_action(raw)
    print(json.dumps(observation, ensure_ascii=False, indent=2))
    return 0 if observation["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
