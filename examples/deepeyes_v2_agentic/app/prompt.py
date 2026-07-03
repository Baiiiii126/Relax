# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Prompts for DeepEyesV2: agent system prompt + code execution & search tool
response templates + sandbox initialization snippet that injects ``image_1``
and patches matplotlib to persist any plotted figure to a known directory."""

import re


# ---------------------------------------------------------------------------
# Unified agent schema (single source of truth, shared by smoke + data convert)
# ---------------------------------------------------------------------------
# All tool calls funnel through <tool_call>{"name": ..., "arguments": ...}</tool_call>.
# This deliberately collapses upstream V2's separate <code>...</code> action
# channel into a single tool_call entry (name="python_exec"), so Instruct-line
# checkpoints (Qwen3-VL etc.) can drive the agent without SFT — they already
# know tool_call format, and dropping the non-standard <code> tag removes the
# main source of schema confusion. See `app.env_deepeyes_v2.extract_code` for
# the receiver side that accepts both shapes (compat with any future legacy
# data); but newly-converted training parquets emit only the unified form.

UNIFIED_SYSTEM_PROMPT = """You are a vision agent that solves multimodal questions by reasoning step-by-step and using tools.

The i-th provided image is pre-loaded as the global variable `image_i` (a `PIL.Image`). The first image is `image_1`. You do not need to re-read these files.

On every turn, first reason inside <think>...</think>. Then emit EXACTLY ONE of the following blocks:

- <tool_call>{"name": "python_exec", "arguments": {"code": "<python source>"}}</tool_call>
    Run Python in the persistent Jupyter kernel. Put the entire program in the "code" string; escape newlines as \\n. Do NOT wrap the code in markdown fences.
- <tool_call>{"name": "search", "arguments": {"query": "<query>"}}</tool_call>
    Issue a text web search.
- <tool_call>{"name": "image_search"}</tool_call>
    Reverse image search on `image_1`.
- <answer>your final answer</answer>
    Terminate. Emit this AS SOON AS you know the answer — do not keep calling tools. The final answer MUST be inside <answer>...</answer>; plain text without the tags is rejected as a format error.

How to return a processed image to yourself so you can see it next turn:
- For a `PIL.Image`: call `_relax_save_image(img)` — saves it to the per-turn capture dir so it is attached to the next user message.
- For matplotlib: just call `plt.show()` — the backend is patched to persist the figure.
- Do NOT use `img.show()` (PIL's GUI viewer) or `img.save("any_path.png")` — those silently drop the image; you will receive a blank response and waste a turn.

Notes:
1. python_exec runs in a persistent kernel — functions and variables carry across turns. It times out after 300 seconds.
2. Programs must return in finite time; no infinite loops.
3. Writing arbitrary files to disk is not allowed.
4. For search-style splits: call `search` at most once per turn; `image_search` at most once per trajectory.

Reminder: as soon as you have the answer, emit <answer>...</answer>. Do NOT keep calling tools after you know the answer.
"""

# Replaces the upstream user-suffix "Format strictly as <think> </think> <code>
# </code> (if code is needed) or <think> <think> <answer> </answer>." sentence
# (and the search-split variant that also mentions <tool_call>). The new wording
# folds the two action channels into one.
UNIFIED_FORMAT_REMINDER = (
    "Format strictly as <think> </think> <tool_call> </tool_call> (if a tool is needed) "
    "or <think> </think> <answer> </answer>."
)

# Match the upstream "Format strictly as ...</answer>." sentence in either the
# 2-action (perception/reason/vstar_test) or 3-action (search) variant. Anchored
# on the unique opener + the trailing </answer>. so we don't over-replace.
FORMAT_REMINDER_RE = re.compile(r"Format strictly as[^.]*?</answer>\s*\.")


def rewrite_user_format_reminder(text: str) -> str:
    """Swap the upstream "Format strictly as ..." sentence for the unified
    form.

    Idempotent: re-running on already-rewritten text matches the new sentence
    and substitutes the same template back in, so the convert step is safe to
    re-run on a partially-rewritten dataset.
    """
    return FORMAT_REMINDER_RE.sub(UNIFIED_FORMAT_REMINDER, text)


# ---------------------------------------------------------------------------
# Tool response templates
# ---------------------------------------------------------------------------

RETURN_CODE_PROMPT = """Code execution result:
stdout:
```
{stdout}
```

stderr:
```
{stderr}
```

{image}
"""

RETURN_SEARCH_PROMPT = """<tool_response>
{search_result}
</tool_response>
"""


# ---------------------------------------------------------------------------
# Sandbox initialization snippet
# ---------------------------------------------------------------------------
# Injects ``image_1`` (PIL.Image) into the sandbox kernel, chdirs into the
# persistent inputs directory, and monkey-patches matplotlib so any
# ``plt.show()`` invocation persists the figure to the per-turn images
# directory for later retrieval by the env.
#
# The two container paths below are FIXED on purpose — they correspond to
# the ``bind_paths`` keys in ``apptainer_env/apptainer_config.yaml``, which
# the backend rewrites at session creation time to per-session host
# subdirectories under its own ``session_tmp``. ``writable_tmpfs=True`` means
# anything written to a non-bound container path lands in the container's
# private overlay and is invisible to the host — so list_files/read_bytes
# would silently see nothing. Per-session isolation is therefore handled by
# the backend's bind layer, NOT by changing these container-side keys.

RELAX_IMAGES_DIR = "/tmp/_relax_imgs"

# Persistent input directory for files that must survive every turn (e.g.
# ``image_1.png``).  ``_run_code_in_sandbox`` cleans ``RELAX_IMAGES_DIR`` after
# each turn to avoid image accumulation, so anything we want the model to be
# able to ``Image.open(...)`` across turns MUST live OUTSIDE that directory.
RELAX_INPUTS_DIR = "/tmp/_relax_inputs"

INITIALIZATION_CODE_TEMPLATE = """
import os, base64, uuid as _uuid
from io import BytesIO
from PIL import Image

# 1) Inject the original sample image as the in-memory variable image_1
_img_b64 = "{base64_image}"
image_1 = Image.open(BytesIO(base64.b64decode(_img_b64)))

# 2) Set up the per-turn image-capture directory (cleared between turns).
_RELAX_IMG_DIR = "{relax_img_dir}"
os.makedirs(_RELAX_IMG_DIR, exist_ok=True)

# 3) Set up the persistent inputs directory and ALSO persist image_1 there as
#    a real PNG file so the model can use either form interchangeably:
#       - Python variable:   image_1            (PIL.Image, in kernel globals)
#       - File on disk:      "{relax_inputs_dir}/image_1.png"
#
#    Empirically the LLM frequently writes ``Image.open("image_1")`` /
#    ``Image.open("image_1.png")`` because the system prompt under-specifies
#    that ``image_1`` is already a PIL.Image. Providing a real file makes
#    those calls succeed instead of FileNotFoundError, drastically reducing
#    sandbox-side errors in the multi-turn rollout (~half of code turns
#    used to die on this).
_RELAX_INPUTS_DIR = "{relax_inputs_dir}"
os.makedirs(_RELAX_INPUTS_DIR, exist_ok=True)
_image_1_path = os.path.join(_RELAX_INPUTS_DIR, "image_1.png")
try:
    image_1.save(_image_1_path)
except Exception as _e:
    print(f"[WARN] failed to persist image_1 to disk: {{_e}}")

# 4) Patch matplotlib so plt.show() persists figures to disk
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    def _relax_show(*args, **kwargs):
        for _num in _plt.get_fignums():
            _fig = _plt.figure(_num)
            _path = os.path.join(_RELAX_IMG_DIR, f"plt_{{_uuid.uuid4().hex}}.png")
            _fig.savefig(_path, dpi=100, bbox_inches="tight")
        _plt.close("all")

    _plt.show = _relax_show
except Exception as _e:
    print(f"[WARN] matplotlib patch failed: {{_e}}")

# 5) Helper for users to explicitly persist a PIL.Image
def _relax_save_image(img):
    _p = os.path.join(_RELAX_IMG_DIR, f"img_{{_uuid.uuid4().hex}}.png")
    img.save(_p)
    return _p

# 6) Convenience: alias the persisted PNG under several common extension
#    variants so the model's various ``Image.open(...)`` formulations all
#    resolve. Empirical distribution observed in rollout logs (top hits):
#       'image_1'      (no extension)         ~186 / round
#       'image_1.jpg'  (wrong extension)      ~22 / round
#       'image_1.png'  (canonical)            handled by save() above
#    Aliases live ONLY in the per-session inputs dir; relative-path opens
#    work because step 7 chdirs the kernel into that dir.
_IMAGE_1_ALIASES = ("image_1", "image_1.png", "image_1.jpg", "image_1.jpeg", "image_1.webp")
for _name in _IMAGE_1_ALIASES:
    _link = os.path.join(_RELAX_INPUTS_DIR, _name)
    if os.path.abspath(_link) == os.path.abspath(_image_1_path):
        continue  # don't symlink a file onto itself
    try:
        if os.path.lexists(_link):
            if os.path.islink(_link) or os.path.isfile(_link):
                os.remove(_link)
            else:
                continue  # leave directories etc. alone
        os.symlink(_image_1_path, _link)
    except Exception as _e:
        print(f"[WARN] failed to alias {{_name}} -> image_1.png: {{_e}}")

# 7) chdir into the inputs dir so model relative-path file ops
#    (``Image.open('image_1.png')``, ``img.save('foo.png')``, etc.) land in
#    the per-session host subdir that the backend binds here — never the
#    apptainer kernel's default cwd which inherits from the host launcher
#    (in smoke that's the source tree, in training it's the rollout dir).
os.chdir(_RELAX_INPUTS_DIR)

print(
    "relax sandbox initialized;"
    f" image_1 size: {{image_1.size}};"
    f" image_1 path: {{_image_1_path}};"
    f" cwd: {{os.getcwd()}};"
    f" aliases: {{list(_IMAGE_1_ALIASES)}}"
)
"""
