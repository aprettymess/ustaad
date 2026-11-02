"""Muse Glimmer over any OpenAI-compatible local server (llama.cpp / Ollama / LM Studio).

Glimmer facts that shape this file (from meta-oss-cookbook):
- reasoning length via chat_template_kwargs.reasoning_strength (llama.cpp ignores reasoning_effort)
- reasoning tokens count against max_tokens -> empty content + finish_reason=length if too low
- server parses tool calls; one tool call per assistant message
- never add <|eom|> to stop tokens (we never send a custom `stop`)
"""
from __future__ import annotations

import base64
import json
import re
import time
from typing import Any, Optional, Type, TypeVar

import cv2
import numpy as np
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def encode_image(img: np.ndarray, max_side: int = 768) -> str:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def user_content(text: str, images: Optional[list[np.ndarray]] = None, labels: Optional[list[str]] = None,
                 max_side: int = 768) -> list[dict]:
    parts: list[dict] = []
    for i, img in enumerate(images or []):
        if labels:
            parts.append({"type": "text", "text": labels[i]})
        parts.append({"type": "image_url", "image_url": {"url": encode_image(img, max_side)}})
    parts.append({"type": "text", "text": text})
    return parts


def _extract_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        return m.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


class GlimmerClient:
    def __init__(self, cfg, log=None):
        from openai import OpenAI
        self.cfg = cfg
        self.log = log or (lambda *a, **k: None)
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout_s)
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    # -- low level
    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None,
             response_format: Optional[dict] = None, strength: Optional[str] = None,
             max_tokens: Optional[int] = None):
        kw: dict[str, Any] = dict(model=self.cfg.model, messages=messages,
                                  temperature=self.cfg.temperature,
                                  max_tokens=max_tokens or self.cfg.max_tokens,
                                  extra_body={"chat_template_kwargs": {
                                      "reasoning_strength": strength or self.cfg.reasoning_strength}})
        if tools:
            kw["tools"], kw["tool_choice"] = tools, "auto"
        if response_format:
            kw["response_format"] = response_format
        t0 = time.time()
        r = self.client.chat.completions.create(**kw)
        dt = time.time() - t0
        self.usage["calls"] += 1
        self.usage["seconds"] += dt
        if r.usage:
            self.usage["prompt_tokens"] += r.usage.prompt_tokens or 0
            self.usage["completion_tokens"] += r.usage.completion_tokens or 0
        return r

    # -- structured output with schema enforcement + one repair retry
    def structured(self, system: str, content: list[dict] | str, model: Type[T],
                   strength: Optional[str] = None, stage: str = "llm") -> T:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        rf = {"type": "json_schema", "json_schema": {"name": model.__name__, "schema": model.model_json_schema()}}
        last_err = None
        for attempt in range(2):
            r = self.chat(messages, response_format=rf, strength=strength)
            choice = r.choices[0]
            text = choice.message.content or ""
            if choice.finish_reason == "length" and not text.strip():
                last_err = "hit max_tokens while reasoning (raise glimmer.max_tokens or lower reasoning_strength)"
                strength = "low"
                continue
            try:
                out = model.model_validate_json(_extract_json(text))
                self.log(stage, "llm", f"{model.__name__} ok",
                         reasoning=(getattr(choice.message, "reasoning_content", None) or "")[:1500],
                         output=out.model_dump())
                return out
            except ValidationError as e:
                last_err = str(e)[:400]
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"That did not match the schema: {last_err}. Reply with valid JSON only."}]
        raise RuntimeError(f"Glimmer structured output failed for {model.__name__}: {last_err}")


def get_glimmer(cfg, log=None):
    if cfg.backend == "mock":
        from .mock import MockGlimmer
        return MockGlimmer(cfg, log)
    return GlimmerClient(cfg, log)
