"""Safe-ish executor for HumanEval-style model-generated code.

WARNING — this module runs UNTRUSTED, MODEL-GENERATED code. Every candidate
completion is executed as a real Python program. The isolation here is a
separate subprocess with a hard wall-clock timeout (and stdin closed), which
is the same pragmatic standard OpenAI's original ``human-eval`` harness uses
on a research box. It is NOT a security boundary: model code can still read
the filesystem, open sockets, or burn CPU/memory up to the OS limits. For any
production or adversarial setting this MUST run inside a container/VM/gVisor
sandbox with network and filesystem locked down. On this single-user research
machine, subprocess + timeout is the accepted trade-off.

The public surface:

- ``extract_code(response, entry_point)`` pulls the completion out of a raw
  model response: strips ```python fences and, when the model re-emits the
  function signature, keeps only the body so that ``prompt + completion`` is a
  single valid function definition.
- ``check_solution(problem, completion, timeout)`` assembles the standard
  HumanEval program ``prompt + completion + test + check(entry_point)`` and
  runs it, returning ``{"passed", "error", "timed_out"}``.
"""

import re
import subprocess
import sys
from typing import Optional, Union


def _field(problem: Union[dict, object], name: str):
    """Read a field from a problem given as a dict or an attribute object."""
    if isinstance(problem, dict):
        return problem[name]
    return getattr(problem, name)


def _strip_fences(text: str) -> str:
    """Return the contents of the first Markdown code block, or the raw text.

    Handles ```python / ```py / bare ``` fences. If an opening fence has no
    matching close (a truncated response), returns everything after it.
    """
    if "```" not in text:
        return text
    match = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Opening fence but no closing fence (truncated generation).
    match = re.search(r"```[^\n]*\n(.*)", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def _strip_reemitted_signature(code: str, entry_point: str) -> str:
    """If ``code`` re-emits ``def entry_point(...):``, keep only the body.

    The HumanEval prompt already supplies the signature and docstring, so a
    model that returns the whole function would otherwise duplicate the
    ``def`` line when concatenated with the prompt. We locate the entry-point
    definition, walk to the end of its (possibly multi-line) header — the
    first ``:`` at parenthesis-depth zero — and return everything after it.

    Anything the model emitted *before* the def (stray imports/helpers) is
    dropped; that is rare in HumanEval and keeping it would break the
    concatenation. Body-only responses (no re-emitted signature) pass through
    unchanged.
    """
    lines = code.split("\n")
    def_idx = None
    for i, line in enumerate(lines):
        if re.match(rf"\s*def\s+{re.escape(entry_point)}\s*\(", line):
            def_idx = i
            break
    if def_idx is None:
        return code

    header = "\n".join(lines[def_idx:])
    depth = 0
    end_pos = None
    for pos, ch in enumerate(header):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ":" and depth == 0:
            end_pos = pos
            break
    if end_pos is None:
        return code  # malformed header; leave as-is
    return header[end_pos + 1 :]


def extract_code(response: Optional[str], entry_point: str) -> str:
    """Extract the completion body from a raw model response.

    Strips Markdown fences, drops a re-emitted function signature, and trims
    surrounding blank lines while preserving the body's indentation (so it can
    be appended directly to the HumanEval prompt).
    """
    text = response or ""
    code = _strip_fences(text)
    code = _strip_reemitted_signature(code, entry_point)
    return code.strip("\n")


def check_solution(
    problem: Union[dict, object], completion: str, timeout: float = 10.0
) -> dict:
    """Run ``completion`` against the problem's unit tests in a subprocess.

    Assembles the standard HumanEval program shape::

        prompt + completion + "\\n" + test + "\\ncheck(entry_point)\\n"

    and executes it via ``sys.executable -c`` with a hard timeout and stdin
    closed. Returns ``{"passed": bool, "error": str|None, "timed_out": bool}``.
    A non-zero exit (assertion failure, exception, syntax error) is
    ``passed=False`` with the captured stderr; exceeding the timeout is
    ``timed_out=True``.
    """
    prompt = _field(problem, "prompt")
    test = _field(problem, "test")
    entry_point = _field(problem, "entry_point")

    program = (
        prompt + completion + "\n" + test + "\ncheck(" + entry_point + ")\n"
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "timed out", "timed_out": True}
    except (OSError, ValueError) as exc:  # e.g. program too large / null bytes
        return {"passed": False, "error": str(exc), "timed_out": False}

    if proc.returncode == 0:
        return {"passed": True, "error": None, "timed_out": False}
    stderr = (proc.stderr or "").strip()
    return {
        "passed": False,
        "error": stderr[-2000:] or f"exit code {proc.returncode}",
        "timed_out": False,
    }
