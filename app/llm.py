# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM — Ollama or OpenAI-compatible backend (llama.cpp)."""

import asyncio
import ast
import httpx
import json
import re
from typing import Optional, Iterator, Dict, Any, Callable, List, Awaitable


def _parse_text_tool_call(content: str, tool_names) -> Optional[tuple]:
    """Recover a tool call a small model emitted as plain text.

    Models like gemma-e2b intermittently print the call into the content
    stream instead of returning a structured tool_calls field, e.g.:

        set_camera_power(on=false)
        analyze_image{question:<|"|>what is in front of you<|"|>}
        analyze_image({'question': 'What do you see?'})

    Returns (name, args_dict) when the content is essentially just such a
    call for a *known* tool, else None. Conservative on purpose: the call
    must start at the beginning of the (stripped) content so ordinary
    prose that merely mentions a tool name isn't hijacked.
    """
    if not content:
        return None
    text = content.strip()
    # Strip wrappers some models add around tool calls.
    for pre in ("```tool_code", "```python", "```json", "```", "tool_call:",
                "tool_code:", "functions.", "print("):
        if text.startswith(pre):
            text = text[len(pre):].strip()
    text = text.strip("`").strip()
    # Leaked special tokens: <|"|> → " ; drop any other <|...|> markers.
    text = re.sub(r'<\|"\|>', '"', text)
    text = re.sub(r"<\|[^|]*\|>", "", text)

    name = next(
        (n for n in tool_names if re.match(rf"\s*{re.escape(n)}\s*[\(\{{]", text)),
        None,
    )
    if name is None:
        return None
    open_idx = min(
        (i for i in (text.find("(", len(name)), text.find("{", len(name))) if i != -1),
        default=-1,
    )
    if open_idx == -1:
        return None
    open_ch = text[open_idx]
    close_ch = ")" if open_ch == "(" else "}"
    depth = 0
    end = -1
    for i in range(open_idx, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    body = text[open_idx + 1:end].strip()
    if not body:
        return (name, {})

    # 1) JSON object as written.
    for candidate in (body, "{" + body + "}"):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return (name, obj)
        except Exception:
            pass
    # 2) Python literal (single quotes, True/False/None).
    for candidate in (body, "{" + body + "}", "dict(" + body + ")"):
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return (name, obj)
        except Exception:
            pass
    # 3) key=value / key: value pairs, splitting on top-level commas only.
    args: Dict[str, Any] = {}
    depth = 0
    buf = ""
    parts = []
    for ch in body:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf)
    for part in parts:
        m = re.match(r"\s*['\"]?([\w\-]+)['\"]?\s*[:=]\s*(.+)$", part, re.S)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip().strip(",").strip()
        low = val.lower()
        if low in ("true", "false"):
            args[key] = (low == "true")
        elif low in ("none", "null"):
            args[key] = None
        elif re.fullmatch(r"-?\d+", val):
            args[key] = int(val)
        elif re.fullmatch(r"-?\d*\.\d+", val):
            args[key] = float(val)
        else:
            args[key] = val.strip("'\"")
    return (name, args) if args else (name, {})


def _summarize_tool_results(messages: list) -> str:
    """Build a one-line spoken fallback from the most recent tool messages.

    Used when the model declines to produce post-tool narration. Prefers the
    tool's own `summary` / `description` field, then the textual error, then
    a generic confirmation so the assistant always says something.
    """
    parts: list[str] = []
    for m in messages:
        if m.get("role") != "tool":
            continue
        try:
            data = json.loads(m.get("content") or "{}")
        except Exception:
            data = {}
        if not isinstance(data, dict):
            continue
        text = (
            data.get("summary")
            or data.get("description")
            or data.get("error")
            or data.get("status")
        )
        if text:
            parts.append(str(text))
    if not parts:
        return ""
    return parts[-1] if len(parts) == 1 else " ".join(parts)


class LLM:
    def __init__(
        self,
        model: str = "",
        base_url: str = "http://localhost:8080",
        backend: str = "openai",
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: str = "",
        timeout: float = 120.0,
        history_turns: int = 6,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.backend = (backend or "openai").lower()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.history_turns = max(0, history_turns)
        # Rolling [user, assistant, user, assistant, ...] of plain-text
        # turns, replayed into each request for conversational context.
        self.history: list[dict] = []
        self._loaded = False

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        """Record a completed exchange. Trims to the last history_turns."""
        if self.history_turns <= 0:
            return
        u = (user_text or "").strip()
        a = (assistant_text or "").strip()
        if not u or not a:
            return
        self.history.append({"role": "user", "content": u})
        self.history.append({"role": "assistant", "content": a})
        max_msgs = self.history_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def reset_history(self) -> None:
        self.history = []

    def load(self) -> bool:
        try:
            with httpx.Client(timeout=10.0) as client:
                if self.backend == "openai":
                    r = client.get(f"{self.base_url}/v1/models")
                    if r.status_code != 200:
                        return False
                    models = [m.get("id", "") for m in r.json().get("data", [])]
                    if not models:
                        return False
                    if not self.model or self.model not in models:
                        self.model = models[0]
                else:
                    r = client.get(f"{self.base_url}/api/tags")
                    if r.status_code != 200:
                        return False
                    names = [m.get("name", "") for m in r.json().get("models", [])]
                    base = self.model.split(":")[0]
                    if base not in [n.split(":")[0] for n in names] and self.model not in names:
                        print(f"Model '{self.model}' not found. Available: {', '.join(names)}")
                        return False
            self._loaded = True
            return True
        except Exception as e:
            print(f"LLM connection error: {e}")
            return False

    def _messages(
        self, prompt: str, system_prompt: Optional[str] = None,
        few_shot: Optional[list[dict]] = None,
    ) -> list:
        msgs = []
        sp = system_prompt or self.system_prompt
        if sp:
            msgs.append({"role": "system", "content": sp})
        if few_shot:
            msgs.extend(few_shot)
        if self.history:
            msgs.extend(self.history)
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _messages_multimodal(
        self, prompt: str, images_b64: list[str],
        system_prompt: Optional[str] = None,
        few_shot: Optional[list[dict]] = None,
    ) -> list:
        msgs = []
        sp = system_prompt or self.system_prompt
        if sp:
            msgs.append({"role": "system", "content": sp})
        if few_shot:
            msgs.extend(few_shot)
        if self.history:
            msgs.extend(self.history)
        content: list[dict] = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        msgs.append({"role": "user", "content": content})
        return msgs

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        images_b64: Optional[list[str]] = None,
        few_shot: Optional[list[dict]] = None,
    ) -> Iterator[tuple]:
        """Yields (content, metadata) tuples. Pass images_b64 for multimodal VLM requests."""
        if not self._loaded:
            yield ("", {})
            return
        mt = max_tokens or self.max_tokens
        t = temperature if temperature is not None else self.temperature
        if images_b64:
            msgs = self._messages_multimodal(prompt, images_b64, system_prompt, few_shot)
        else:
            msgs = self._messages(prompt, system_prompt, few_shot)

        if self.backend == "openai":
            yield from self._stream_openai(msgs, mt, t)
        else:
            yield from self._stream_ollama(msgs, mt, t)

    def _stream_openai(self, messages, max_tokens, temperature) -> Iterator[tuple]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", f"{self.base_url}/v1/chat/completions", json={
                    "model": self.model, "messages": messages, "stream": True,
                    "max_tokens": max_tokens, "temperature": temperature,
                }) as r:
                    if r.status_code != 200:
                        err = r.read().decode(errors="replace")[:300]
                        print(f"\n  [LLM error {r.status_code}] {err}")
                        yield ("", {})
                        return
                    for line in r.iter_lines():
                        if not line or not line.strip().startswith("data:"):
                            continue
                        line = line.strip()
                        if line == "data: [DONE]":
                            yield ("", {"done": True})
                            return
                        try:
                            data = json.loads(line[5:])
                            usage = data.get("usage")
                            if usage:
                                yield ("", {"done": True, "eval_count": usage.get("completion_tokens", 0)})
                                return
                            content = ((data.get("choices") or [{}])[0].get("delta") or {}).get("content", "")
                            if content:
                                yield (content, {})
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"LLM stream error: {e}")
            yield ("", {})

    def _stream_ollama(self, messages, max_tokens, temperature) -> Iterator[tuple]:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", f"{self.base_url}/api/chat", json={
                    "model": self.model, "messages": messages, "stream": True,
                    "keep_alive": "1h",
                    "options": {"num_predict": max_tokens, "temperature": temperature},
                }) as r:
                    if r.status_code != 200:
                        err = r.read().decode(errors="replace")[:300]
                        print(f"\n  [LLM error {r.status_code}] {err}")
                        yield ("", {})
                        return
                    for line in r.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            done = data.get("done", False)
                            meta = {}
                            if done:
                                meta = {"done": True, "eval_count": data.get("eval_count", 0)}
                            if content:
                                yield (content, meta)
                            elif done:
                                yield ("", meta)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"LLM stream error: {e}")
            yield ("", {})

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        dispatcher: Callable[[str, Any], Awaitable[Dict[str, Any]]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        few_shot: Optional[list[dict]] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any], Dict[str, Any]], None]] = None,
        max_rounds: int = 1,
    ) -> Iterator[tuple]:
        """Non-streaming tool round, then streaming final response.

        Flow:
          1. Make a non-streaming chat completion request with `tools` set so
             we can cleanly inspect the response for tool_calls.
          2. If no tool_calls, yield the assistant content as a single chunk.
          3. Otherwise dispatch each call via `dispatcher(name, args)`,
             append tool results, optionally repeat (max_rounds), then make a
             final streaming call so TTS still gets token-by-token output.

        Yields (content, metadata) tuples like generate_stream(). Tool calls
        themselves are surfaced via the optional `on_tool_call` callback so
        the caller can broadcast them to the UI.
        """
        if not self._loaded:
            yield ("", {})
            return

        mt = max_tokens or self.max_tokens
        t = temperature if temperature is not None else self.temperature
        messages = self._messages(prompt, system_prompt, few_shot)

        # Tool-calling currently lives on the OpenAI-compatible path only.
        # Ollama exposes /v1/chat/completions which accepts the `tools` field
        # for capable models; the native /api/chat tools shape differs and is
        # not wired up here.
        if self.backend != "openai":
            yield from self._stream_openai(messages, mt, t)
            return

        for _round in range(max(1, max_rounds)):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    payload: Dict[str, Any] = {
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": mt,
                        "temperature": t,
                        "stream": False,
                    }
                    if tools:
                        payload["tools"] = tools
                        payload["tool_choice"] = "auto"
                    r = client.post(
                        f"{self.base_url}/v1/chat/completions", json=payload,
                    )
                    if r.status_code != 200:
                        err = r.text[:300]
                        print(f"\n  [LLM tool-call error {r.status_code}] {err}")
                        yield ("", {})
                        return
                    data = r.json()
            except Exception as e:
                print(f"LLM tool-call request error: {e}")
                yield ("", {})
                return

            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                content = msg.get("content") or ""
                # Small models sometimes print the call as text instead of
                # returning structured tool_calls. Recover it so the tool
                # actually runs instead of the call being spoken aloud.
                recovered = _parse_text_tool_call(
                    content,
                    {(t.get("function") or {}).get("name") for t in (tools or [])},
                ) if tools else None
                if recovered is None:
                    if content:
                        yield (content, {})
                    yield ("", {"done": True, "eval_count": (data.get("usage") or {}).get("completion_tokens", 0)})
                    return
                r_name, r_args = recovered
                tool_calls = [{
                    "id": "text-0",
                    "type": "function",
                    "function": {"name": r_name, "arguments": json.dumps(r_args)},
                }]
                msg = {"content": None, "tool_calls": tool_calls}

            # Append assistant turn that requested tools, then run them.
            # Per OpenAI spec, content should be null when tool_calls is set.
            messages.append({
                "role": "assistant",
                "content": msg.get("content") or None,
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments")
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except Exception:
                    parsed_args = {}
                try:
                    result = asyncio.run(dispatcher(name, raw_args))
                except RuntimeError:
                    # Already in an event loop — fall back to a fresh loop.
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(dispatcher(name, raw_args))
                    finally:
                        loop.close()
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}

                if on_tool_call is not None:
                    try:
                        on_tool_call(name, parsed_args if isinstance(parsed_args, dict) else {}, result)
                    except Exception:
                        pass

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": name,
                    "content": json.dumps(result, default=str),
                })

        # Final streaming pass with tool results in context. Some models
        # (notably small ones like gemma) sometimes return zero tokens after
        # a tool round, treating the tool result as the answer. That leaves
        # the UI without an assistant bubble and TTS silent, so we fall back
        # to a narration synthesized from the tool results when the stream
        # is empty.
        produced_any = False
        last_meta: Dict[str, Any] = {}
        for chunk, meta in self._stream_openai(messages, mt, t):
            if chunk:
                produced_any = True
            if meta:
                last_meta = meta
            yield (chunk, meta)

        if not produced_any:
            fallback = _summarize_tool_results(messages)
            if fallback:
                yield (fallback, {})
                yield ("", {"done": True, "eval_count": last_meta.get("eval_count", 0)})

    def health_check(self) -> bool:
        if not self._loaded:
            return False
        try:
            with httpx.Client(timeout=5.0) as client:
                url = f"{self.base_url}/v1/models" if self.backend == "openai" else f"{self.base_url}/api/tags"
                return client.get(url).status_code == 200
        except Exception:
            return False

    def unload(self):
        self._loaded = False
