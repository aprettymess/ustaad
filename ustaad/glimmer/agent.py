"""Minimal tool-calling agent loop for Muse Glimmer.

Rules that matter for Glimmer (cookbook): one tool call per assistant message; tools return
errors as JSON strings instead of raising (that's how the model self-corrects).
Tool results must be TEXT — llama.cpp rejects images inside tool results — so any images
go in the initial user message.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def spec(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


def run_agent(glimmer, system: str, user: Any, tools: list[Tool], max_steps: int = 10,
              stage: str = "agent", log=None, strength: str | None = None) -> str:
    log = log or (lambda *a, **k: None)
    by_name = {t.name: t for t in tools}
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for step in range(max_steps):
        r = glimmer.chat(messages, tools=[t.spec() for t in tools], strength=strength)
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            log(stage, "llm", f"final (step {step})", content=(msg.content or "")[:2000])
            return msg.content or ""
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name,
                                                      "arguments": c.function.arguments}} for c in calls]})
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool = by_name.get(c.function.name)
            if tool is None:
                result = {"error": f"unknown tool {c.function.name}", "hint": f"available: {list(by_name)}"}
            else:
                try:
                    result = tool.fn(**args)
                except Exception as e:  # noqa: BLE001 - errors go back to the model
                    result = {"error": str(e)}
            log(stage, "tool", f"{c.function.name}({json.dumps(args)[:200]})",
                result=result if isinstance(result, (dict, list)) else str(result)[:500])
            messages.append({"role": "tool", "tool_call_id": c.id,
                             "content": result if isinstance(result, str) else json.dumps(result)})
            if tool is not None and tool.name == "finish":
                return json.dumps(result)
    log(stage, "info", "agent hit max_steps")
    return ""
