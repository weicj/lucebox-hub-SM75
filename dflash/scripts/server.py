"""
OpenAI-compatible HTTP server on top of test_dflash.

    pip install fastapi uvicorn transformers
    python3 scripts/server.py                 # serves on :8000

    curl http://localhost:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"luce-dflash","messages":[{"role":"user","content":"hi"}],"stream":true}'

Drop-in for Open WebUI / LM Studio / Cline by setting
  OPENAI_API_BASE=http://localhost:8000/v1  OPENAI_API_KEY=sk-any

Streams tokens as Server-Sent Events using the OpenAI delta format.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware          # FIX 1: add CORS
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool
from transformers import AutoTokenizer

from _prefill_hook import (
    PrefillConfig, add_cli_flags, config_from_args,
    compress_text_via_daemon,
)
from prefix_cache import DaemonStdoutBus, PrefixCache


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = Path(os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
))
DEFAULT_DRAFT_ROOT = ROOT / "models" / "draft"
DEFAULT_BIN = ROOT / "build" / ("test_dflash" + (".exe" if sys.platform == "win32" else ""))
DEFAULT_BUDGET = 22
MODEL_NAME = "luce-dflash"

# Architecture strings stored in `general.architecture` of every GGUF this
# server can drive. test_dflash dispatches by GGUF arch internally:
#   qwen35 / qwen36  -> existing DFlash + DDTree pipeline
#   laguna           -> dflash27b::run_laguna_daemon() (no spec-decode)
# server.py just needs to omit --draft + the DFlash/DDTree flags when the
# arch doesn't support speculative decoding yet.
_QWEN35_ARCHES = {"qwen35", "qwen36"}
_LAGUNA_ARCHES  = {"laguna"}

_ALLOWED_TEMPLATE_KWARGS = frozenset({"enable_thinking", "tools", "add_generation_prompt"})


def resolve_draft(root: Path) -> Path:
    for st in root.rglob("model.safetensors"):
        return st
    raise FileNotFoundError(f"no model.safetensors under {root}")


_QWEN35_FAMILY_TOKENIZERS = {
    "Qwen3.5-27B": "Qwen/Qwen3.5-27B",
    "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
}
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"

_LAGUNA_FAMILY_TOKENIZERS = {
    "Laguna-XS.2": "poolside/Laguna-XS.2",
    "Laguna-XS":   "poolside/Laguna-XS.2",
    "laguna-xs2":  "poolside/Laguna-XS.2",
}


def _read_gguf_str(reader, key: str) -> str | None:
    f = reader.fields.get(key)
    if f is None or not f.data:
        return None
    import numpy as np
    p = f.parts[f.data[0]]
    if not isinstance(p, np.ndarray):
        return None
    try:
        return bytes(p).decode("utf-8", errors="replace")
    except Exception:
        return None


def _arch_from_gguf(gguf_path: Path) -> str:
    """Return the value of ``general.architecture`` from the GGUF, or 'unknown'.

    server.py uses this to dispatch between the qwen35 stack (test_dflash +
    DFlash + DDTree) and the laguna stack (test_laguna_daemon, autoregressive
    only). 'unknown' falls back to the qwen35 path so existing setups keep
    working when the field is missing.
    """
    try:
        from gguf import GGUFReader  # type: ignore
        r = GGUFReader(str(gguf_path))
        v = _read_gguf_str(r, "general.architecture")
        return v.lower() if v else "unknown"
    except Exception:
        return "unknown"


def _tokenizer_id_from_gguf(gguf_path: Path) -> str:
    default = "Qwen/Qwen3.5-27B"
    try:
        from gguf import GGUFReader  # type: ignore
        r = GGUFReader(str(gguf_path))
        arch = (_read_gguf_str(r, "general.architecture") or "").lower()
        family = _LAGUNA_FAMILY_TOKENIZERS if arch in _LAGUNA_ARCHES else _QWEN35_FAMILY_TOKENIZERS
        if arch in _LAGUNA_ARCHES:
            default = next(iter(_LAGUNA_FAMILY_TOKENIZERS.values()))
        for key in ("general.basename", "general.name"):
            val = _read_gguf_str(r, key)
            if val is None:
                continue
            for known, repo in family.items():
                if known.lower() in val.lower():
                    return repo
    except Exception:
        pass
    return default


# ─── tool-call & reasoning parsers ─────────────────────────────────
# Ported from server_tools.py which ports from vLLM (Apache-2.0):
#   vllm/reasoning/qwen3_reasoning_parser.py
#   vllm/tool_parsers/qwen3coder_tool_parser.py

TOOL_CALL_COMPLETE_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_CALL_FUNCTION_RE = re.compile(
    r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL,
)
TOOL_CALL_PARAMETER_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)
TOOL_OPEN_TAG = "<tool_call>"
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


def normalize_stop(stop) -> list[str]:
    """Coerce OpenAI's stop field (str | list[str] | None) to list[str]."""
    if not stop:
        return []
    if isinstance(stop, str):
        return [stop]
    return [s for s in stop if isinstance(s, str) and s]


def first_stop_match(text: str, stops: list[str]) -> int:
    """Return the earliest index where any stop sequence appears, or -1."""
    best = -1
    for s in stops:
        i = text.find(s)
        if i != -1 and (best == -1 or i < best):
            best = i
    return best


def parse_reasoning(
    text: str,
    thinking_enabled: bool = True,
    started_in_thinking: bool = False,
) -> tuple[str, str | None]:
    """Extract reasoning content from Qwen3.x's <think>...</think> blocks.

    Handles paired, headless, and disabled thinking flavors.
    ``started_in_thinking`` accounts for prompts that end with ``<think>\n``
    so the generated text contains only the reasoning body + ``</think>``.
    Returns (cleaned_content, reasoning_content).
    """
    parts = text.partition(THINK_OPEN_TAG)
    saw_open_tag = bool(parts[1])
    rest = parts[2] if saw_open_tag else parts[0]
    if THINK_CLOSE_TAG not in rest:
        if thinking_enabled and (started_in_thinking or saw_open_tag):
            return "", (rest.strip() or None)
        return rest.strip(), None
    reasoning, _, content = rest.partition(THINK_CLOSE_TAG)
    return content.strip(), (reasoning.strip() or None)


def _find_tool_properties(tools, function_name):
    """Returns the parameters dict for a given function name, or {}."""
    for t in tools or []:
        fn = t.function if hasattr(t, "function") else t.get("function", {})
        if hasattr(fn, "model_dump"):
            fn = fn.model_dump()
        if fn.get("name") == function_name:
            params = fn.get("parameters", {})
            if isinstance(params, dict):
                return params.get("properties", {})
    return {}


def _convert_param_value(param_value: str, param_name: str, param_config: dict,
                         func_name: str):
    """Coerce stringified XML values to their JSON-schema type."""
    import ast
    if param_value.lower() == "null":
        return None
    if param_name not in param_config:
        return param_value
    cfg = param_config[param_name]
    if isinstance(cfg, dict) and "type" in cfg:
        ptype = str(cfg["type"]).strip().lower()
    elif isinstance(cfg, dict) and "anyOf" in cfg:
        ptype = "object"
    else:
        ptype = "string"
    if ptype in ("string", "str", "text", "varchar", "char", "enum"):
        return param_value
    if any(ptype.startswith(p) for p in ("int", "uint", "long", "short", "unsigned")):
        try: return int(param_value)
        except (ValueError, TypeError): return param_value
    if ptype.startswith("num") or ptype.startswith("float"):
        try:
            f = float(param_value)
            return f if f - int(f) != 0 else int(f)
        except (ValueError, TypeError):
            return param_value
    if ptype in ("boolean", "bool", "binary"):
        return param_value.lower() == "true"
    if (ptype in ("object", "array", "arr")
            or ptype.startswith("dict") or ptype.startswith("list")):
        try: return json.loads(param_value)
        except (json.JSONDecodeError, TypeError, ValueError): pass
    try: return ast.literal_eval(param_value)
    except (ValueError, SyntaxError, TypeError): return param_value


def parse_tool_calls(text: str, tools=None) -> tuple[str, list[dict]]:
    """Parse Qwen3.x <tool_call> XML blocks into OpenAI tool_calls format.

    Returns (cleaned_content, tool_calls_list).
    """
    tool_calls: list[dict] = []
    cleaned_parts: list[str] = []
    cursor = 0
    for m in TOOL_CALL_COMPLETE_RE.finditer(text):
        cleaned_parts.append(text[cursor:m.start()])
        cursor = m.end()
        body = m.group(1)
        fn_match = TOOL_CALL_FUNCTION_RE.search(body)
        if not fn_match:
            continue
        fn_text = fn_match.group(1) or fn_match.group(2) or ""
        end_idx = fn_text.find(">")
        if end_idx == -1:
            continue
        function_name = fn_text[:end_idx].strip()
        params_region = fn_text[end_idx + 1:]
        param_config = _find_tool_properties(tools, function_name)
        args: dict = {}
        for match_text in TOOL_CALL_PARAMETER_RE.findall(params_region):
            eq_idx = match_text.find(">")
            if eq_idx == -1:
                continue
            k = match_text[:eq_idx].strip()
            v = match_text[eq_idx + 1:]
            if v.startswith("\n"): v = v[1:]
            if v.endswith("\n"): v = v[:-1]
            args[k] = _convert_param_value(v, k, param_config, function_name)
        tool_calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    cleaned_parts.append(text[cursor:])
    return "".join(cleaned_parts).strip(), tool_calls


# FIX 2: _content_to_str helper used for BOTH OpenAI and Anthropic message
# content fields (str | list[dict]). Previously OpenAI list[dict] content
# was passed raw to the tokenizer and caused a crash.
def _content_to_str(content: "str | list[dict] | None") -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


# ─── pydantic schemas ──────────────────────────────────────────────

class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string per OpenAI spec


class ToolCall(BaseModel):
    id: str | None = None
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: str
    # FIX 2 cont: accept list[dict] in the model but always stringify it
    content: Any | None = None  # str, list, or null when tool_calls present
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolDef(BaseModel):
    type: str = "function"
    function: dict  # {name, description, parameters: {...JSON schema...}}


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 512
    temperature: float | None = None   # 0 = greedy, >0 = sample
    seed: int | None = None             # rng seed for sampling
    top_p: float | None = None         # nucleus, applied when temperature > 0
    top_k: int | None = None           # top-k, applied when temperature > 0
    frequency_penalty: float | None = None  # OAI -> rep_pen = 1 + freq_pen (sampling only)
    stop: list[str] | str | None = None  # FIX 3: accept stop field (Open WebUI sends it)
    tools: list[ToolDef] | None = None
    tool_choice: Any | None = None  # "auto" | "none" | {"function": {...}}
    chat_template_kwargs: dict | None = None
    stream_options: dict | None = None  # e.g. {"include_usage": true}


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[dict]


class AnthropicMessagesRequest(BaseModel):
    model: str = MODEL_NAME
    max_tokens: int
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    stop_sequences: list[str] | None = None
    chat_template_kwargs: dict | None = None


# ─── Responses API schemas (Codex wire protocol) ──────────────────

class ResponseInputMessage(BaseModel):
    type: str = "message"
    id: str | None = None
    role: str = "user"
    content: Any  # str or list[dict] content parts
    status: str | None = None


class ResponseFunctionCall(BaseModel):
    type: str = "function_call"
    id: str | None = None
    call_id: str
    name: str
    arguments: str
    status: str | None = None


class ResponseFunctionCallOutput(BaseModel):
    type: str = "function_call_output"
    id: str | None = None
    call_id: str
    output: Any  # str or structured
    status: str | None = None


class ResponseToolFunction(BaseModel):
    type: str = "function"
    name: str
    description: str | None = None
    parameters: dict | None = None
    strict: bool | None = None


class ResponseReasoningConfig(BaseModel):
    effort: str | None = None  # "low" | "medium" | "high"
    summary: str | None = None  # "auto" | "concise" | "detailed" | "none"


class ResponsesCreateRequest(BaseModel):
    model: str = MODEL_NAME
    input: Any  # str or list[InputItem dicts]
    instructions: str | None = None
    tools: list[dict] | None = None
    tool_choice: str | None = "auto"
    parallel_tool_calls: bool | None = None
    stream: bool | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    reasoning: ResponseReasoningConfig | None = None
    store: bool | None = None
    include: list[str] | None = None
    text: dict | None = None
    metadata: dict | None = None
    previous_response_id: str | None = None


def _samp_suffix(req) -> str:
    # Render ` samp=temp,top_p,top_k,rep_pen[,seed]` tail when the request asks for
    # non-greedy decoding. Empty string keeps the daemon protocol greedy-compatible.
    t  = float(getattr(req, "temperature", 0.0) or 0.0)
    if t <= 0.0:
        return ""
    tp = float(getattr(req, "top_p", 1.0) or 1.0)
    tk = int(getattr(req, "top_k", 0) or 0)
    rp = float(getattr(req, "frequency_penalty", 0.0) or 0.0) + 1.0
    seed = int(getattr(req, "seed", 0) or 0)
    return f" samp={t:.4f},{tp:.4f},{tk},{rp:.4f},{seed}"


def build_app(target: Path, draft: Path | None, bin_path: Path, budget: int, max_ctx: int,
              tokenizer: AutoTokenizer, stop_ids: set[int],
              prefill_cfg: PrefillConfig | None = None,
              drafter_tokenizer: AutoTokenizer | None = None,
              prefix_cache_slots: int = 4,
              prefill_cache_slots: int = 4,
              arch: str = "qwen35") -> FastAPI:
    import asyncio
    app = FastAPI(title="Luce DFlash OpenAI server")

    # FIX 1: CORS middleware so Open WebUI / browser frontends on other ports
    # can reach this server without being blocked by the browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    daemon_lock = asyncio.Lock()

    r_pipe, w_pipe = os.pipe()
    if sys.platform == "win32":
        import msvcrt
        os.set_inheritable(w_pipe, True)
        stream_fd_val = int(msvcrt.get_osfhandle(w_pipe))
    else:
        stream_fd_val = w_pipe

    bin_abs = str(Path(bin_path).resolve())
    dll_dir = str(Path(bin_abs).parent / "bin")
    env = {**os.environ}
    if sys.platform == "win32":
        env["PATH"] = dll_dir + os.pathsep + str(Path(bin_abs).parent) + os.pathsep + env.get("PATH", "")

    if arch in _LAGUNA_ARCHES:
        # test_dflash detects arch=laguna from the GGUF and dispatches
        # internally to dflash27b::run_laguna_daemon(). No --draft, no
        # --fast-rollback, no --ddtree (no Laguna spec-decode draft yet).
        # Tokens stream as int32 LE on stream_fd terminated by -1, byte-
        # identical to the qwen35 path so SSE/stream consumers stay shared.
        cmd = [bin_abs, str(target), "--daemon",
               f"--max-ctx={max_ctx}",
               f"--stream-fd={stream_fd_val}"]
    else:
        if draft is None:
            raise SystemExit("qwen35 arch requires --draft model.safetensors")
        cmd = [bin_abs, str(target), str(draft), "--daemon",
               "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",
               f"--max-ctx={max_ctx}",
               f"--stream-fd={stream_fd_val}"]
    if sys.platform == "win32":
        daemon_proc = subprocess.Popen(cmd, close_fds=False, env=env,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, bufsize=0)
    else:
        daemon_proc = subprocess.Popen(cmd, pass_fds=(w_pipe,), env=env,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, bufsize=0)
    os.close(w_pipe)

    bus = DaemonStdoutBus(daemon_proc.stdout)

    def _resolve_kv_k_type():
        kv = "q8_0"
        if os.environ.get("DFLASH27B_KV_F16", "0") != "0":
            kv = "f16"
        if os.environ.get("DFLASH27B_KV_Q4", "0") != "0":
            kv = "q4_0"
        if os.environ.get("DFLASH27B_KV_TQ3", "0") != "0":
            kv = "tq3_0"
        if os.environ.get("DFLASH27B_KV_K"):
            kv = os.environ["DFLASH27B_KV_K"].lower()
        return kv

    _fa_window = int(os.environ.get("DFLASH27B_FA_WINDOW", 2048))
    prefix_cache = PrefixCache(
        daemon_stdin=daemon_proc.stdin,
        await_reply=bus.await_reply,
        daemon_lock=daemon_lock,
        tokenizer=tokenizer,
        kv_k_type=_resolve_kv_k_type(),
        fa_window=_fa_window,
        cap=prefix_cache_slots,
    )
    if prefill_cfg is not None and prefill_cache_slots > 0:
        prefix_cache.init_full_cache(prefill_cache_slots)

    @app.on_event("startup")
    async def _startup():
        bus.start(asyncio.get_running_loop())
        await prefix_cache.startup_sync()

    # FIX 4: /health endpoint — Open WebUI and many clients ping this before
    # sending requests. Without it they show a permanent "disconnected" badge.
    @app.get("/health")
    def health():
        alive = daemon_proc.poll() is None
        if not alive:
            return JSONResponse({"status": "error", "detail": "daemon exited"}, status_code=503)
        return {"status": "ok"}

    # FIX 5: richer /v1/models response — Open WebUI uses `context_length` and
    # `created` to populate the model picker and context-bar correctly.
    @app.get("/v1/models")
    def list_models(request: Request):
        # Codex sends ?client_version= — serve the Codex-specific schema
        if "client_version" in request.query_params:
            return {"models": [{
                "slug": MODEL_NAME,
                "display_name": MODEL_NAME,
                "description": "Local DFlash speculative-decoding server",
                "default_reasoning_level": "low",
                "supported_reasoning_levels": [
                    {"effort": "low", "description": "No thinking"},
                    {"effort": "medium", "description": "Thinking enabled"},
                ],
                "shell_type": "shell_command",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 1,
                "context_window": max_ctx,
                "supports_reasoning_summaries": False,
                "supports_parallel_tool_calls": False,
            }]}
        return {
            "object": "list",
            "data": [{
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "luce",
                "created": 1700000000,
                "context_length": max_ctx,          # shown in Open WebUI header
                "max_context_length": max_ctx,
            }],
        }

    def _ids_to_bin(ids: list[int]) -> Path:
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return Path(path)

    def _render_messages(msgs_list: list[dict],
                         template_kwargs: dict | None = None,
                         tools_arg: list[dict] | None = None,
                         ) -> tuple[Path, list[int], str]:
        """Apply chat template to msgs_list and return (bin path, ids, raw prompt).

        The raw prompt is returned for spec-prefill: when compression fires we
        re-tokenise it with the drafter vocab.

        ``template_kwargs`` is passed through to ``apply_chat_template`` so callers
        can toggle template knobs like ``enable_thinking`` per-request.

        Thinking is disabled by default (enable_thinking=False) because Qwen3.6's
        think mode wrecks DFlash acceptance rates. Clients can opt in by sending
        ``"chat_template_kwargs": {"enable_thinking": true}`` in the request.
        """
        tpl_kwargs: dict = {"tokenize": False, "add_generation_prompt": True,
                            "enable_thinking": False}
        tpl_kwargs.update(
            {k: v for k, v in (template_kwargs or {}).items() if k in _ALLOWED_TEMPLATE_KWARGS}
        )
        if tools_arg:
            tpl_kwargs["tools"] = tools_arg
        prompt = tokenizer.apply_chat_template(msgs_list, **tpl_kwargs)
        started_in_thinking = bool(re.search(r"<think>\s*$", prompt))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        return _ids_to_bin(ids), ids, prompt

    def _tokenize_prompt(req: ChatRequest) -> tuple[Path, list[int], list[dict], bool]:
        """Returns (bin, ids, raw_msgs, started_in_thinking)."""
        msgs: list[dict] = []
        for m in req.messages:
            d: dict = {"role": m.role}
            if m.content is not None:
                d["content"] = _content_to_str(m.content)
            if m.name is not None:
                d["name"] = m.name
            if m.tool_call_id is not None:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls is not None:
                d["tool_calls"] = []
                for tc in m.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try: args_obj = json.loads(args)
                        except (json.JSONDecodeError, ValueError): args_obj = {"_raw": args}
                    else:
                        args_obj = args
                    d["tool_calls"].append({
                        "id": tc.id, "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": args_obj},
                    })
            msgs.append(d)

        tools_arg = None
        if req.tools:
            tools_arg = [t.model_dump() for t in req.tools]

        path, ids, _prompt = _render_messages(msgs, req.chat_template_kwargs, tools_arg)
        started_in_thinking = bool(re.search(r"<think>\s*$", _prompt))
        return path, ids, msgs, started_in_thinking

    def _maybe_compress(msgs: list[dict], prompt_bin: Path, prompt_ids: list[int],
                        template_kwargs: dict | None = None
                        ) -> tuple[Path, list[int]]:
        if not prefill_cfg or not prefill_cfg.enabled:
            return prompt_bin, prompt_ids
        if not prefill_cfg.should_compress(len(prompt_ids)):
            return prompt_bin, prompt_ids
        if drafter_tokenizer is None:
            return prompt_bin, prompt_ids

        last_user_idx = next((i for i in range(len(msgs) - 1, -1, -1)
                              if msgs[i]["role"] == "user"), None)
        if last_user_idx is None:
            return prompt_bin, prompt_ids
        long_text = msgs[last_user_idx]["content"]

        compressed_text = compress_text_via_daemon(
            daemon_stdin=daemon_proc.stdin,
            r_pipe=r_pipe,
            drafter_tokenizer=drafter_tokenizer,
            cfg=prefill_cfg,
            prompt_text=long_text,
            skip_park=prefill_cfg.skip_park,
        )

        new_msgs = list(msgs)
        new_msgs[last_user_idx] = {"role": "user", "content": compressed_text}
        new_bin, new_ids, _ = _render_messages(new_msgs, template_kwargs)
        try:
            prompt_bin.unlink()
        except Exception:
            pass
        return new_bin, new_ids

    def _token_stream(r, n_gen):
        generated = 0
        hit_stop = False
        while True:
            b = os.read(r, 4)
            if not b or len(b) < 4:
                break
            tok_id = struct.unpack("<i", b)[0]
            if tok_id == -1:
                break
            if hit_stop:
                continue
            if tok_id in stop_ids:
                hit_stop = True
                continue
            generated += 1
            yield tok_id
            if generated >= n_gen:
                hit_stop = True

    # FIX 6: _collect_tokens_sync — non-streaming paths previously called
    # list(_token_stream(...)) directly (blocking the event loop) or used
    # an async comprehension over _astream_tokens inside daemon_lock
    # (risking a deadlock if the threadpool stalled). Using run_in_executor
    # offloads the blocking os.read loop to a thread without holding any
    # asyncio primitive across the thread boundary.
    async def _collect_tokens_sync(r, n_gen) -> list[int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: list(_token_stream(r, n_gen)))

    async def _astream_tokens(r, n_gen):
        generated = 0
        hit_stop = False
        loop = asyncio.get_running_loop()
        while True:
            b = await loop.run_in_executor(None, os.read, r, 4)
            if not b or len(b) < 4:
                break
            tok_id = struct.unpack("<i", b)[0]
            if tok_id == -1:
                break
            if hit_stop:
                continue
            if tok_id in stop_ids:
                hit_stop = True
                continue
            generated += 1
            yield tok_id
            if generated >= n_gen:
                hit_stop = True

    # FIX 7: _write_cmd helper — centralises stdin write+flush and guards
    # against a dead daemon so callers get a clean 503 instead of a hang.
    def _write_cmd(cmd_line: str):
        if daemon_proc.poll() is not None:
            raise RuntimeError("dflash daemon has exited unexpectedly")
        daemon_proc.stdin.write(cmd_line.encode("utf-8"))
        daemon_proc.stdin.flush()

    def _build_cmd_line(req, cur_bin, cur_ids, gen_len, prefix_cache,
                        prompt_ids, full_snap_prep_ref: list,
                        compression_fired: bool):
        """
        FIX 8: extracted cmd_line construction so both streaming and
        non-streaming paths share identical logic and can't diverge.
        Returns (cmd_line, snap_prep).
        full_snap_prep_ref is a 1-element list used as an out-param.
        """
        if compression_fired:
            full_snap_prep = prefix_cache.prepare_full_snap(prompt_ids)
            full_snap_prep_ref[0] = full_snap_prep
            samp = _samp_suffix(req)
            if full_snap_prep is not None:
                fslot, _ = full_snap_prep
                return f"{cur_bin} {gen_len} snap={len(cur_ids)}:{fslot}" + samp + "\n", None
            else:
                return f"{cur_bin} {gen_len}" + samp + "\n", None
        else:
            full_snap_prep_ref[0] = None
            hit = prefix_cache.lookup(cur_ids)
            snap_prep = prefix_cache.prepare_inline_snap(cur_ids)
            if hit:
                slot, _prefix_len = hit
                cmd_line = f"RESTORE {slot} {cur_bin} {gen_len}"
            else:
                cmd_line = f"{cur_bin} {gen_len}"
            if snap_prep:
                cmd_line += f" snap={snap_prep[1]}:{snap_prep[0]}"
            return cmd_line + _samp_suffix(req) + "\n", snap_prep

    def _gen_len_for(prompt_len: int, max_tokens: int) -> int:
        return min(max_tokens, max_ctx - prompt_len - 20)

    # ── /v1/chat/completions ────────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        prompt_bin, prompt_ids, raw_msgs, started_in_thinking = _tokenize_prompt(req)
        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                nonlocal started_in_thinking
                async with daemon_lock:
                    full_snap_prep_ref = [None]
                    snap_prep = None

                    full_hit = prefix_cache.lookup_full(prompt_ids)
                    if full_hit is not None:
                        slot, cached_cur_bin, cached_cur_ids_len = full_hit
                        cur_bin = Path(cached_cur_bin)
                        prompt_len = cached_cur_ids_len
                        started_in_thinking = False  # cached: no think prefill
                        gen_len = _gen_len_for(prompt_len, req.max_tokens)
                        if gen_len <= 0:
                            try: prompt_bin.unlink()
                            except Exception: pass
                            err = {"id": completion_id, "object": "chat.completion.chunk",
                                   "created": created, "model": MODEL_NAME,
                                   "choices": [{"index": 0, "delta": {},
                                                "finish_reason": "length"}]}
                            yield f"data: {json.dumps(err)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}" + _samp_suffix(req) + "\n"
                    else:
                        cur_bin, cur_ids = await asyncio.to_thread(
                            _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                            req.chat_template_kwargs)
                        prompt_len = len(cur_ids)
                        gen_len = _gen_len_for(prompt_len, req.max_tokens)
                        if gen_len <= 0:
                            try: cur_bin.unlink()
                            except Exception: pass
                            err = {"id": completion_id, "object": "chat.completion.chunk",
                                   "created": created, "model": MODEL_NAME,
                                   "choices": [{"index": 0, "delta": {},
                                                "finish_reason": "length"}]}
                            yield f"data: {json.dumps(err)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        compression_fired = (cur_bin != prompt_bin)
                        cmd_line, snap_prep = _build_cmd_line(
                            req, cur_bin, cur_ids, gen_len, prefix_cache,
                            prompt_ids, full_snap_prep_ref, compression_fired)

                    # FIX 7: guard against dead daemon
                    try:
                        _write_cmd(cmd_line)
                    except RuntimeError as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    head = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_NAME,
                        "choices": [{"index": 0,
                                     "delta": {"role": "assistant"},
                                     "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(head)}\n\n"
                    window, mode = "", ("reasoning" if started_in_thinking else "content")

                    include_usage = bool(req.stream_options and req.stream_options.get("include_usage"))

                    def chunk(delta_obj, finish=None):
                        return {"id": completion_id, "object": "chat.completion.chunk",
                                "created": created, "model": MODEL_NAME,
                                "choices": [{"index": 0, "delta": delta_obj,
                                              "finish_reason": finish}]}

                    # State machine: mode ∈ {'reasoning', 'content', 'tool_buffer'}
                    mode = "reasoning" if started_in_thinking else "content"
                    window = ""
                    tool_buffer = ""
                    stops = normalize_stop(req.stop)
                    tag_holdback = max(len(THINK_OPEN_TAG), len(THINK_CLOSE_TAG), len(TOOL_OPEN_TAG))
                    stop_holdback = max((len(s) for s in stops), default=0)
                    HOLDBACK = max(tag_holdback, stop_holdback)
                    completion_tokens = 0
                    stop_hit = False

                    def emit_delta(text, kind):
                        if not text:
                            return None
                        return f"data: {json.dumps(chunk({kind: text}))}\n\n"

                    try:
                        async for tok_id in _astream_tokens(r_pipe, gen_len):
                            completion_tokens += 1
                            piece = tokenizer.decode([tok_id])
                            window += piece

                            if stops and mode != "tool_buffer":
                                si = first_stop_match(window, stops)
                                if si != -1:
                                    window = window[:si]
                                    stop_hit = True
                                    kind = "reasoning_content" if mode == "reasoning" else "content"
                                    out = emit_delta(window, kind)
                                    if out: yield out
                                    window = ""
                                    break

                            while True:
                                if mode == "tool_buffer":
                                    tool_buffer += window
                                    window = ""
                                    break

                                if mode == "reasoning":
                                    idx = window.find(THINK_CLOSE_TAG)
                                    if idx != -1:
                                        pre = window[:idx]
                                        out = emit_delta(pre, "reasoning_content")
                                        if out: yield out
                                        window = window[idx + len(THINK_CLOSE_TAG):]
                                        mode = "content"
                                        continue
                                    if len(window) > HOLDBACK:
                                        safe = window[:-HOLDBACK]
                                        out = emit_delta(safe, "reasoning_content")
                                        if out: yield out
                                        window = window[-HOLDBACK:]
                                    break

                                else:  # mode == "content"
                                    think_idx = window.find(THINK_OPEN_TAG)
                                    tool_idx  = window.find(TOOL_OPEN_TAG)
                                    hits = [(i, t) for i, t in
                                            ((think_idx, "think"), (tool_idx, "tool")) if i != -1]
                                    if hits:
                                        hits.sort()
                                        idx, which = hits[0]
                                        pre = window[:idx]
                                        out = emit_delta(pre, "content")
                                        if out: yield out
                                        if which == "think":
                                            window = window[idx + len(THINK_OPEN_TAG):]
                                            mode = "reasoning"
                                        else:
                                            tool_buffer = window[idx:]
                                            window = ""
                                            mode = "tool_buffer"
                                        continue
                                    if len(window) > HOLDBACK:
                                        safe = window[:-HOLDBACK]
                                        out = emit_delta(safe, "content")
                                        if out: yield out
                                        window = window[-HOLDBACK:]
                                    break

                        if stop_hit:
                            finish_reason = "stop"
                            yield f"data: {json.dumps(chunk({}, finish=finish_reason))}\n\n"
                            if include_usage:
                                usage_chunk = {"id": completion_id, "object": "chat.completion.chunk",
                                               "created": created, "model": MODEL_NAME, "choices": [],
                                               "usage": {"prompt_tokens": prompt_len,
                                                          "completion_tokens": completion_tokens,
                                                          "total_tokens": prompt_len + completion_tokens}}
                                yield f"data: {json.dumps(usage_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            if full_hit is None:
                                try: cur_bin.unlink()
                                except Exception: pass
                            return

                        # Flush remaining
                        if mode == "reasoning" and window:
                            out = emit_delta(window, "reasoning_content")
                            if out: yield out
                        elif mode == "content" and window:
                            out = emit_delta(window, "content")
                            if out: yield out
                        elif mode == "tool_buffer":
                            tool_buffer += window
                        window = ""

                        finish_reason = "stop"
                        if mode == "tool_buffer":
                            cleaned_after, tool_calls = parse_tool_calls(tool_buffer, tools=req.tools)
                            if tool_calls:
                                if cleaned_after:
                                    out = emit_delta(cleaned_after, "content")
                                    if out: yield out
                                tc_delta_list = [{
                                    "index": i, "id": tc["id"], "type": "function",
                                    "function": {"name": tc["function"]["name"],
                                                  "arguments": tc["function"]["arguments"]},
                                } for i, tc in enumerate(tool_calls)]
                                yield f"data: {json.dumps(chunk({'tool_calls': tc_delta_list}))}\n\n"
                                finish_reason = "tool_calls"
                            else:
                                out = emit_delta(tool_buffer, "content")
                                if out: yield out
                    finally:
                        if full_hit is None:
                            try: cur_bin.unlink()
                            except Exception: pass
                        else:
                            try: prompt_bin.unlink()
                            except Exception: pass

                    full_snap_prep = full_snap_prep_ref[0]
                    if full_snap_prep is not None:
                        fslot, _ = full_snap_prep
                        prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
                    elif snap_prep:
                        prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

                    yield f"data: {json.dumps(chunk({}, finish=finish_reason))}\n\n"
                    if include_usage:
                        usage_chunk = {
                            "id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": MODEL_NAME,
                            "choices": [],
                            "usage": {"prompt_tokens": prompt_len,
                                       "completion_tokens": completion_tokens,
                                       "total_tokens": prompt_len + completion_tokens},
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming
        async with daemon_lock:
            full_snap_prep_ref = [None]
            snap_prep = None

            full_hit = prefix_cache.lookup_full(prompt_ids)
            if full_hit is not None:
                slot, cached_cur_bin, cached_cur_ids_len = full_hit
                cur_bin = Path(cached_cur_bin)
                cur_ids = None
                prompt_len = cached_cur_ids_len
                gen_len = _gen_len_for(prompt_len, req.max_tokens)
                if gen_len <= 0:
                    try: prompt_bin.unlink()
                    except Exception: pass
                    return JSONResponse(
                        {"detail": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"},
                        status_code=400)
                cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}" + _samp_suffix(req) + "\n"
            else:
                cur_bin, cur_ids = await asyncio.to_thread(
                    _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                            req.chat_template_kwargs)
                prompt_len = len(cur_ids)
                gen_len = _gen_len_for(prompt_len, req.max_tokens)
                if gen_len <= 0:
                    try: cur_bin.unlink()
                    except Exception: pass
                    return JSONResponse(
                        {"detail": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"},
                        status_code=400)
                compression_fired = (cur_bin != prompt_bin)
                cmd_line, snap_prep = _build_cmd_line(
                    req, cur_bin, cur_ids, gen_len, prefix_cache,
                    prompt_ids, full_snap_prep_ref, compression_fired)

            try:
                _write_cmd(cmd_line)
            except RuntimeError as e:
                return JSONResponse({"detail": str(e)}, status_code=503)

            # FIX 6: use run_in_executor instead of list() blocking event loop
            tokens = await _collect_tokens_sync(r_pipe, gen_len)

            full_snap_prep = full_snap_prep_ref[0]
            if full_snap_prep is not None:
                fslot, _ = full_snap_prep
                prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
            elif snap_prep:
                prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

        if full_hit is None:
            try: cur_bin.unlink()
            except Exception: pass
        else:
            try: prompt_bin.unlink()
            except Exception: pass

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        # Stop sequences
        stops = normalize_stop(req.stop)
        if stops:
            i = first_stop_match(text, stops)
            if i != -1:
                text = text[:i]
        # Parse reasoning and tool calls
        thinking_enabled = True
        if req.chat_template_kwargs:
            thinking_enabled = req.chat_template_kwargs.get("enable_thinking", True)
        cleaned, tool_calls = parse_tool_calls(text, tools=req.tools)
        cleaned, reasoning = parse_reasoning(
            cleaned,
            thinking_enabled=thinking_enabled,
            started_in_thinking=started_in_thinking,
        )

        msg: dict = {"role": "assistant"}
        finish_reason = "stop"
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["content"] = cleaned if cleaned else None
            msg["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            msg["content"] = cleaned

        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": msg,
                "finish_reason": finish_reason,
            }],
            "usage": {"prompt_tokens": prompt_len,
                      "completion_tokens": len(tokens),
                      "total_tokens": prompt_len + len(tokens)},
        })

    # ── Anthropic Messages API ──────────────────────────────────────────────

    def _tokenize_anthropic(req: AnthropicMessagesRequest
                            ) -> tuple[Path, list[int], list[dict], bool]:
        msgs = []
        system_text = _content_to_str(req.system) if req.system else None
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        for m in req.messages:
            msgs.append({"role": m.role, "content": _content_to_str(m.content)})
        path, ids, prompt = _render_messages(msgs, req.chat_template_kwargs)
        think = _thinking_enabled(req.chat_template_kwargs) and prompt_starts_in_thinking(prompt)
        return path, ids, msgs, think

    @app.post("/v1/messages")
    async def anthropic_messages(req: AnthropicMessagesRequest):
        prompt_bin, prompt_ids, raw_msgs, started_in_thinking = _tokenize_anthropic(req)
        msg_id = "msg_" + uuid.uuid4().hex[:24]

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                nonlocal started_in_thinking
                async with daemon_lock:
                    full_snap_prep_ref = [None]
                    snap_prep = None

                    full_hit = prefix_cache.lookup_full(prompt_ids)
                    if full_hit is not None:
                        slot, cached_cur_bin, cached_cur_ids_len = full_hit
                        cur_bin = Path(cached_cur_bin)
                        cur_ids = None
                        prompt_len = cached_cur_ids_len
                        gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
                        if gen_len <= 0:
                            try: prompt_bin.unlink()
                            except Exception: pass
                            err = {"type": "error",
                                   "error": {"type": "invalid_request_error",
                                             "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}}
                            yield f"event: error\ndata: {json.dumps(err)}\n\n"
                            return
                        cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}" + _samp_suffix(req) + "\n"
                    else:
                        cur_bin, cur_ids = await asyncio.to_thread(
                            _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                            req.chat_template_kwargs)
                        prompt_len = len(cur_ids)
                        gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
                        if gen_len <= 0:
                            try: cur_bin.unlink()
                            except Exception: pass
                            err = {"type": "error",
                                   "error": {"type": "invalid_request_error",
                                             "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}}
                            yield f"event: error\ndata: {json.dumps(err)}\n\n"
                            return
                        compression_fired = (cur_bin != prompt_bin)
                        cmd_line, snap_prep = _build_cmd_line(
                            req, cur_bin, cur_ids, gen_len, prefix_cache,
                            prompt_ids, full_snap_prep_ref, compression_fired)

                    message_start = {
                        "type": "message_start",
                        "message": {
                            "id": msg_id, "type": "message", "role": "assistant",
                            "model": req.model or MODEL_NAME,
                            "content": [], "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": prompt_len, "output_tokens": 0},
                        },
                    }
                    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

                    try:
                        _write_cmd(cmd_line)
                    except RuntimeError as e:
                        yield f"event: error\ndata: {json.dumps({'type':'error','error':{'type':'server_error','message':str(e)}})}\n\n"
                        return

                    out_tokens = 0
                    window, mode = "", ("reasoning" if started_in_thinking else "content")
                    block_index = 0
                    active_kind = "thinking" if mode == "reasoning" else "text"
                    block = {"type": active_kind}
                    if active_kind == "thinking":
                        block["thinking"] = ""
                    else:
                        block["text"] = ""
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': block})}\n\n"
                    try:
                        async for tok_id in _astream_tokens(r_pipe, gen_len):
                            out_tokens += 1
                            outputs, window, mode = consume_stream_piece(
                                window, mode, tokenizer.decode([tok_id]))
                            for kind, text in outputs:
                                target_kind = "thinking" if kind == "reasoning_content" else "text"
                                if target_kind != active_kind:
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                    block_index += 1
                                    active_kind = target_kind
                                    new_block = {"type": active_kind, active_kind: ""}
                                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': new_block})}\n\n"
                                delta_type = "thinking_delta" if target_kind == "thinking" else "text_delta"
                                delta_key = "thinking" if target_kind == "thinking" else "text"
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': delta_type, delta_key: text}})}\n\n"
                        for kind, text in flush_stream_deltas(window, mode):
                            target_kind = "thinking" if kind == "reasoning_content" else "text"
                            if target_kind != active_kind:
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                                block_index += 1
                                active_kind = target_kind
                                new_block = {"type": active_kind, active_kind: ""}
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_index, 'content_block': new_block})}\n\n"
                            delta_type = "thinking_delta" if target_kind == "thinking" else "text_delta"
                            delta_key = "thinking" if target_kind == "thinking" else "text"
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_index, 'delta': {'type': delta_type, delta_key: text}})}\n\n"
                    finally:
                        if full_hit is None:
                            try: cur_bin.unlink()
                            except Exception: pass
                        else:
                            try: prompt_bin.unlink()
                            except Exception: pass

                    full_snap_prep = full_snap_prep_ref[0]
                    if full_snap_prep is not None:
                        fslot, _ = full_snap_prep
                        prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
                    elif snap_prep:
                        prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                    msg_delta = {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": out_tokens},
                    }
                    yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming Anthropic
        async with daemon_lock:
            full_snap_prep_ref = [None]
            snap_prep = None

            full_hit = prefix_cache.lookup_full(prompt_ids)
            if full_hit is not None:
                slot, cached_cur_bin, cached_cur_ids_len = full_hit
                cur_bin = Path(cached_cur_bin)
                cur_ids = None
                prompt_len = cached_cur_ids_len
                gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
                if gen_len <= 0:
                    try: prompt_bin.unlink()
                    except Exception: pass
                    return JSONResponse(
                        {"type": "error", "error": {"type": "invalid_request_error",
                         "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}},
                        status_code=400)
                cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}" + _samp_suffix(req) + "\n"
            else:
                cur_bin, cur_ids = await asyncio.to_thread(
                    _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                            req.chat_template_kwargs)
                prompt_len = len(cur_ids)
                gen_len = min(req.max_tokens, max_ctx - prompt_len - 20)
                if gen_len <= 0:
                    try: cur_bin.unlink()
                    except Exception: pass
                    return JSONResponse(
                        {"type": "error", "error": {"type": "invalid_request_error",
                         "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}},
                        status_code=400)
                compression_fired = (cur_bin != prompt_bin)
                cmd_line, snap_prep = _build_cmd_line(
                    req, cur_bin, cur_ids, gen_len, prefix_cache,
                    prompt_ids, full_snap_prep_ref, compression_fired)

            try:
                _write_cmd(cmd_line)
            except RuntimeError as e:
                return JSONResponse({"type": "error", "error": {"type": "server_error",
                                     "message": str(e)}}, status_code=503)

            # FIX 6: use run_in_executor — same fix as OpenAI non-streaming path
            tokens = await _collect_tokens_sync(r_pipe, gen_len)

            full_snap_prep = full_snap_prep_ref[0]
            if full_snap_prep is not None:
                fslot, _ = full_snap_prep
                prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
            elif snap_prep:
                prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

        if full_hit is None:
            try: cur_bin.unlink()
            except Exception: pass
        else:
            try: prompt_bin.unlink()
            except Exception: pass

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        cleaned, reasoning = parse_reasoning(
            text,
            thinking_enabled=_thinking_enabled(req.chat_template_kwargs),
            started_in_thinking=started_in_thinking,
        )
        content = [{"type": "text", "text": cleaned}]
        if reasoning:
            content.insert(0, {"type": "thinking", "thinking": reasoning})
        return JSONResponse({
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": req.model or MODEL_NAME,
            "content": content,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_len,
                      "output_tokens": len(tokens)},
        })

    # ── Responses API (Codex wire protocol) ───────────────────────────

    def _map_responses_input(req: ResponsesCreateRequest
                             ) -> tuple[list[ChatMessage], list[ToolDef] | None]:
        """Map Responses API input → ChatMessage list + ToolDef list."""
        messages: list[ChatMessage] = []

        # instructions → system message
        if req.instructions:
            messages.append(ChatMessage(role="system", content=req.instructions))

        # Parse input items
        input_items = req.input
        if isinstance(input_items, str):
            messages.append(ChatMessage(role="user", content=input_items))
        elif isinstance(input_items, list):
            for item in input_items:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "message")

                if item_type == "message":
                    role = item.get("role", "user")
                    if role == "developer":
                        role = "system"
                    content = item.get("content", "")
                    if isinstance(content, list):
                        # Extract text from content parts
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") in ("output_text", "text", "input_text"):
                                    text_parts.append(part.get("text", ""))
                        content = "".join(text_parts)
                    messages.append(ChatMessage(role=role, content=content))

                elif item_type == "function_call":
                    tc = ToolCall(
                        id=item.get("call_id", "call_" + uuid.uuid4().hex[:12]),
                        type="function",
                        function=ToolCallFunction(
                            name=item.get("name", ""),
                            arguments=item.get("arguments", "{}"),
                        ),
                    )
                    messages.append(ChatMessage(
                        role="assistant", content=None, tool_calls=[tc]))

                elif item_type == "function_call_output":
                    output = item.get("output", "")
                    if not isinstance(output, str):
                        output = json.dumps(output)
                    messages.append(ChatMessage(
                        role="tool",
                        tool_call_id=item.get("call_id", ""),
                        content=output))

                # Ignore reasoning, local_shell_call, etc. — we just
                # need the message/function_call/output items for the model.

        # Map tools
        tools: list[ToolDef] | None = None
        if req.tools:
            tool_defs = []
            for t in req.tools:
                if not isinstance(t, dict):
                    continue
                if t.get("type") == "function":
                    func_def = {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                    }
                    if "parameters" in t:
                        func_def["parameters"] = t["parameters"]
                    tool_defs.append(ToolDef(type="function", function=func_def))
            if tool_defs:
                tools = tool_defs

        return messages, tools

    @app.post("/v1/responses")
    async def responses_create(req: ResponsesCreateRequest):
        messages, tools = _map_responses_input(req)

        # Build an internal ChatRequest
        enable_thinking = False
        if req.reasoning and req.reasoning.effort and req.reasoning.effort != "low":
            enable_thinking = True

        chat_req = ChatRequest(
            model=req.model or MODEL_NAME,
            messages=messages,
            stream=bool(req.stream),
            max_tokens=req.max_output_tokens or 4096,
            temperature=req.temperature,
            top_p=req.top_p,
            tools=tools,
            tool_choice=req.tool_choice,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )

        response_id = "resp_" + uuid.uuid4().hex[:24]
        msg_item_id = "msg_" + uuid.uuid4().hex[:24]
        created_at = int(time.time())

        # Tokenize
        prompt_bin, prompt_ids, raw_msgs, started_in_thinking = _tokenize_prompt(chat_req)
        prompt_len = len(prompt_ids)

        if req.stream:
            return await _responses_stream(
                chat_req, prompt_bin, prompt_ids, raw_msgs,
                started_in_thinking, response_id, msg_item_id,
                created_at, prompt_len)
        else:
            return await _responses_non_stream(
                chat_req, prompt_bin, prompt_ids, raw_msgs,
                response_id, msg_item_id, created_at, prompt_len)

    async def _responses_non_stream(
            chat_req, prompt_bin, prompt_ids, raw_msgs,
            response_id, msg_item_id, created_at, prompt_len):
        """Non-streaming Responses API handler."""
        async with daemon_lock:
            full_snap_prep_ref = [None]
            snap_prep = None

            full_hit = prefix_cache.lookup_full(prompt_ids)
            if full_hit is not None:
                slot, cached_cur_bin, cached_cur_ids_len = full_hit
                cur_bin = Path(cached_cur_bin)
                cur_ids = None
                prompt_len = cached_cur_ids_len
                gen_len = _gen_len_for(prompt_len, chat_req.max_tokens)
                if gen_len <= 0:
                    try: prompt_bin.unlink()
                    except Exception: pass
                    return JSONResponse({
                        "type": "error",
                        "error": {"type": "invalid_request_error",
                                  "message": f"Prompt too long ({prompt_len})"}
                    }, status_code=400)
                cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}" + _samp_suffix(chat_req) + "\n"
            else:
                cur_bin, cur_ids = await asyncio.to_thread(
                    _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                    chat_req.chat_template_kwargs)
                prompt_len = len(cur_ids)
                gen_len = _gen_len_for(prompt_len, chat_req.max_tokens)
                if gen_len <= 0:
                    try: cur_bin.unlink()
                    except Exception: pass
                    return JSONResponse({
                        "type": "error",
                        "error": {"type": "invalid_request_error",
                                  "message": f"Prompt too long ({prompt_len})"}
                    }, status_code=400)
                compression_fired = (cur_bin != prompt_bin)
                cmd_line, snap_prep = _build_cmd_line(
                    chat_req, cur_bin, cur_ids, gen_len, prefix_cache,
                    prompt_ids, full_snap_prep_ref, compression_fired)

            try:
                _write_cmd(cmd_line)
            except RuntimeError as e:
                return JSONResponse({
                    "type": "error",
                    "error": {"type": "server_error", "message": str(e)}
                }, status_code=503)

            tokens = await _collect_tokens_sync(r_pipe, gen_len)

            full_snap_prep = full_snap_prep_ref[0]
            if full_snap_prep is not None:
                fslot, _ = full_snap_prep
                prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
            elif snap_prep:
                prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

        if full_hit is None:
            try: cur_bin.unlink()
            except Exception: pass
        else:
            try: prompt_bin.unlink()
            except Exception: pass

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        thinking_enabled = True
        if chat_req.chat_template_kwargs:
            thinking_enabled = chat_req.chat_template_kwargs.get("enable_thinking", True)
        cleaned, tool_calls = parse_tool_calls(text, tools=chat_req.tools)
        cleaned, reasoning = parse_reasoning(cleaned, thinking_enabled=thinking_enabled)

        # Build output items
        output: list[dict] = []
        if tool_calls:
            for tc in tool_calls:
                output.append({
                    "type": "function_call",
                    "id": tc["id"],
                    "status": "completed",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                })
        else:
            output.append({
                "type": "message",
                "id": msg_item_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": cleaned, "annotations": []}],
            })

        return JSONResponse({
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": "completed",
            "model": chat_req.model or MODEL_NAME,
            "output": output,
            "output_text": cleaned,
            "usage": {
                "input_tokens": prompt_len,
                "output_tokens": len(tokens),
                "total_tokens": prompt_len + len(tokens),
            },
        })

    async def _responses_stream(
            chat_req, prompt_bin, prompt_ids, raw_msgs,
            started_in_thinking, response_id, msg_item_id,
            created_at, prompt_len):
        """Streaming Responses API handler — emits Responses SSE events."""

        async def sse() -> AsyncIterator[str]:
            nonlocal prompt_len, started_in_thinking

            async with daemon_lock:
                full_snap_prep_ref = [None]
                snap_prep = None

                full_hit = prefix_cache.lookup_full(prompt_ids)
                if full_hit is not None:
                    slot, cached_cur_bin, cached_cur_ids_len = full_hit
                    cur_bin = Path(cached_cur_bin)
                    prompt_len = cached_cur_ids_len
                    started_in_thinking = False
                    gen_len = _gen_len_for(prompt_len, chat_req.max_tokens)
                    if gen_len <= 0:
                        try: prompt_bin.unlink()
                        except Exception: pass
                        yield _resp_sse("response.failed", {
                            "response": _resp_shell(response_id, chat_req.model, created_at,
                                                     "failed")})
                        return
                    cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}" + _samp_suffix(chat_req) + "\n"
                else:
                    cur_bin, cur_ids = await asyncio.to_thread(
                        _maybe_compress, raw_msgs, prompt_bin, prompt_ids,
                        chat_req.chat_template_kwargs)
                    prompt_len = len(cur_ids)
                    gen_len = _gen_len_for(prompt_len, chat_req.max_tokens)
                    if gen_len <= 0:
                        try: cur_bin.unlink()
                        except Exception: pass
                        yield _resp_sse("response.failed", {
                            "response": _resp_shell(response_id, chat_req.model, created_at,
                                                     "failed")})
                        return
                    compression_fired = (cur_bin != prompt_bin)
                    cmd_line, snap_prep = _build_cmd_line(
                        chat_req, cur_bin, cur_ids, gen_len, prefix_cache,
                        prompt_ids, full_snap_prep_ref, compression_fired)

                try:
                    _write_cmd(cmd_line)
                except RuntimeError as e:
                    yield _resp_sse("error", {
                        "error": {"type": "server_error", "message": str(e)}})
                    return

                # Lifecycle: response.created
                yield _resp_sse("response.created", {
                    "response": _resp_shell(response_id, chat_req.model, created_at,
                                             "in_progress")})

                # Announce output item
                yield _resp_sse("response.output_item.added", {
                    "output_index": 0,
                    "item": {"type": "message", "id": msg_item_id,
                             "status": "in_progress", "role": "assistant",
                             "content": []}})

                # Announce content part
                yield _resp_sse("response.content_part.added", {
                    "item_id": msg_item_id, "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []}})

                # Stream tokens with state machine
                mode = "reasoning" if started_in_thinking else "content"
                window = ""
                tool_buffer = ""
                accumulated_text = ""
                tag_holdback = max(len(THINK_OPEN_TAG), len(THINK_CLOSE_TAG), len(TOOL_OPEN_TAG))
                HOLDBACK = tag_holdback
                completion_tokens = 0
                tool_call_active = False

                try:
                    async for tok_id in _astream_tokens(r_pipe, gen_len):
                        completion_tokens += 1
                        piece = tokenizer.decode([tok_id])
                        window += piece

                        while True:
                            if mode == "tool_buffer":
                                tool_buffer += window
                                window = ""
                                break

                            if mode == "reasoning":
                                idx = window.find(THINK_CLOSE_TAG)
                                if idx != -1:
                                    window = window[idx + len(THINK_CLOSE_TAG):]
                                    mode = "content"
                                    continue
                                if len(window) > HOLDBACK:
                                    window = window[-HOLDBACK:]
                                break

                            else:  # content
                                think_idx = window.find(THINK_OPEN_TAG)
                                tool_idx = window.find(TOOL_OPEN_TAG)
                                hits = [(i, t) for i, t in
                                        ((think_idx, "think"), (tool_idx, "tool")) if i != -1]
                                if hits:
                                    hits.sort()
                                    idx, which = hits[0]
                                    pre = window[:idx]
                                    if pre:
                                        accumulated_text += pre
                                        yield _resp_sse("response.output_text.delta", {
                                            "item_id": msg_item_id, "output_index": 0,
                                            "content_index": 0, "delta": pre})
                                    if which == "think":
                                        window = window[idx + len(THINK_OPEN_TAG):]
                                        mode = "reasoning"
                                    else:
                                        tool_buffer = window[idx:]
                                        window = ""
                                        mode = "tool_buffer"
                                    continue
                                if len(window) > HOLDBACK:
                                    safe = window[:-HOLDBACK]
                                    accumulated_text += safe
                                    yield _resp_sse("response.output_text.delta", {
                                        "item_id": msg_item_id, "output_index": 0,
                                        "content_index": 0, "delta": safe})
                                    window = window[-HOLDBACK:]
                                break

                    # Flush remaining window
                    if mode == "content" and window:
                        accumulated_text += window
                        yield _resp_sse("response.output_text.delta", {
                            "item_id": msg_item_id, "output_index": 0,
                            "content_index": 0, "delta": window})
                    elif mode == "tool_buffer":
                        tool_buffer += window
                    window = ""

                finally:
                    if full_hit is None:
                        try: cur_bin.unlink()
                        except Exception: pass
                    else:
                        try: prompt_bin.unlink()
                        except Exception: pass

                full_snap_prep = full_snap_prep_ref[0]
                if full_snap_prep is not None:
                    fslot, _ = full_snap_prep
                    prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
                elif snap_prep:
                    prefix_cache.confirm_inline_snap(*snap_prep, cur_ids)

                # Build final output items
                final_output: list[dict] = []
                if mode == "tool_buffer" and tool_buffer:
                    cleaned_after, tool_calls = parse_tool_calls(tool_buffer, tools=chat_req.tools)
                    if tool_calls:
                        if cleaned_after:
                            accumulated_text += cleaned_after
                        for tc in tool_calls:
                            tool_call_active = True
                            tc_item_id = tc["id"]
                            # Emit function_call_arguments.delta for each tool call
                            yield _resp_sse("response.function_call_arguments.delta", {
                                "item_id": tc_item_id, "output_index": 0,
                                "delta": tc["function"]["arguments"]})
                            yield _resp_sse("response.function_call_arguments.done", {
                                "item_id": tc_item_id, "output_index": 0,
                                "arguments": tc["function"]["arguments"],
                                "name": tc["function"]["name"]})
                            final_output.append({
                                "type": "function_call",
                                "id": tc_item_id,
                                "status": "completed",
                                "call_id": tc_item_id,
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            })
                    else:
                        accumulated_text += tool_buffer
                        yield _resp_sse("response.output_text.delta", {
                            "item_id": msg_item_id, "output_index": 0,
                            "content_index": 0, "delta": tool_buffer})

                # Finalize text output
                yield _resp_sse("response.output_text.done", {
                    "item_id": msg_item_id, "output_index": 0,
                    "content_index": 0, "text": accumulated_text})
                yield _resp_sse("response.content_part.done", {
                    "item_id": msg_item_id, "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": accumulated_text,
                             "annotations": []}})

                if not tool_call_active:
                    final_output.append({
                        "type": "message",
                        "id": msg_item_id,
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": accumulated_text,
                                     "annotations": []}],
                    })

                yield _resp_sse("response.output_item.done", {
                    "output_index": 0,
                    "item": final_output[0] if final_output else {
                        "type": "message", "id": msg_item_id,
                        "status": "completed", "role": "assistant",
                        "content": []}})

                # response.completed
                shell = _resp_shell(response_id, chat_req.model, created_at,
                                     "completed")
                shell["output"] = final_output
                shell["output_text"] = accumulated_text
                shell["usage"] = {
                    "input_tokens": prompt_len,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_len + completion_tokens,
                }
                yield _resp_sse("response.completed", {"response": shell})

        return StreamingResponse(sse(), media_type="text/event-stream")

    def _resp_sse(event_type: str, data: dict) -> str:
        """Format a Responses API SSE event."""
        data["type"] = event_type
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def _resp_shell(resp_id: str, model: str, created_at: int,
                    status: str) -> dict:
        """Minimal response shell for SSE lifecycle events."""
        return {
            "id": resp_id,
            "object": "response",
            "created_at": created_at,
            "status": status,
            "model": model or MODEL_NAME,
        }

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--draft",  type=Path, default=DEFAULT_DRAFT_ROOT)
    ap.add_argument("--bin",    type=Path, default=DEFAULT_BIN)
    ap.add_argument("--budget", type=int,  default=DEFAULT_BUDGET)
    default_ctx = 16384
    ap.add_argument("--max-ctx", type=int, default=default_ctx,
                    help=f"Maximum context length (default: {default_ctx}; "
                         "oversizing this — e.g. 131072 on short prompts — "
                         "can slow attention 20×+ until issue #10 is fixed)")
    ap.add_argument("--kv-f16", action="store_true",
                    help="Force F16 KV cache.")
    ap.add_argument("--cache-type-k", "--ctk", dest="cache_type_k", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"])
    ap.add_argument("--cache-type-v", "--ctv", dest="cache_type_v", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"])
    ap.add_argument("--fa-window", type=int, default=None,
                    help="Sliding window for FA layers. 0 = full attention.")
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--prefix-cache-slots", type=int, default=4)
    ap.add_argument("--prefill-cache-slots", type=int, default=4)
    ap.add_argument("--daemon", action="store_true")
    add_cli_flags(ap)
    args = ap.parse_args()
    prefill_cfg = config_from_args(args)

    if args.cache_type_k:
        os.environ["DFLASH27B_KV_K"] = args.cache_type_k
    if args.cache_type_v:
        os.environ["DFLASH27B_KV_V"] = args.cache_type_v
    if args.max_ctx > 6144 and not args.kv_f16 and not args.cache_type_k and not args.cache_type_v:
        os.environ.setdefault("DFLASH27B_KV_TQ3", "1")

    if args.fa_window is not None:
        os.environ["DFLASH27B_FA_WINDOW"] = str(args.fa_window)

    if args.prefill_compression != "off":
        os.environ.setdefault("DFLASH27B_LM_HEAD_FIX", "0")
        os.environ.setdefault("DFLASH27B_FA_WINDOW", "0")
        os.environ.setdefault("DFLASH_FP_USE_BSA", "1")
        os.environ.setdefault("DFLASH_FP_ALPHA",   "0.85")
        if prefill_cfg.skip_park:
            os.environ["DFLASH_COMPRESS_NO_PARK"] = "1"

    if not args.target.is_file():
        raise SystemExit(f"target GGUF not found at {args.target}")

    # Architecture detection. test_dflash itself dispatches by GGUF arch at
    # main() entry, so server.py just needs to know enough to omit --draft +
    # DFlash/DDTree flags on archs that lack a spec-decode draft. Same
    # binary serves every arch.
    arch = _arch_from_gguf(args.target)

    if not args.bin.is_file():
        raise SystemExit(f"binary not found at {args.bin} (arch={arch})")

    if arch in _LAGUNA_ARCHES:
        # No DFlash draft model exists for laguna yet; test_dflash'́s
        # internal arch dispatch reads general.architecture, accepts the
        # no-draft argv layout, and routes to run_laguna_daemon(). PFlash
        # compression and prefix-cache SNAPSHOT/RESTORE are both wired
        # through the laguna daemon now, so --prefill-compression and
        # --prefix-cache-slots behave the same as on the qwen35 path.
        draft = None
    else:
        draft = resolve_draft(args.draft) if args.draft.is_dir() else args.draft
        if not draft.is_file():
            raise SystemExit(f"draft safetensors not found at {args.draft}")

    tokenizer_id = args.tokenizer or _tokenizer_id_from_gguf(args.target)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    stop_ids = set()
    for s in ("<|im_end|>", "<|endoftext|>"):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids: stop_ids.add(ids[0])

    drafter_tokenizer = None
    if prefill_cfg.enabled:
        drafter_tokenizer = AutoTokenizer.from_pretrained(
            prefill_cfg.drafter_tokenizer_id, trust_remote_code=True)

    app = build_app(args.target, draft, args.bin, args.budget, args.max_ctx,
                    tokenizer, stop_ids,
                    prefill_cfg=prefill_cfg if prefill_cfg.enabled else None,
                    drafter_tokenizer=drafter_tokenizer,
                    prefix_cache_slots=args.prefix_cache_slots,
                    prefill_cache_slots=args.prefill_cache_slots,
                    arch=arch)

    import uvicorn
    print(f"Luce DFlash OpenAI server on http://{args.host}:{args.port}")
    print(f"  arch      = {arch}")
    print(f"  target    = {args.target}")
    print(f"  draft     = {draft}")
    print(f"  bin       = {args.bin}")
    print(f"  budget    = {args.budget}")
    print(f"  max_ctx   = {args.max_ctx}")
    print(f"  tokenizer = {tokenizer_id}")
    if prefill_cfg.enabled:
        print(f"  pflash    = {prefill_cfg.mode} · threshold={prefill_cfg.threshold} "
              f"keep={prefill_cfg.keep_ratio} drafter={prefill_cfg.drafter_gguf}")
    else:
        print("  pflash    = off")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
