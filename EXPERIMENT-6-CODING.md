# Experiment 6 — Coding (verifiable-executable query type)

## What it is

The second query type in the council study. The first (HLE) is **judged
multiple-choice**: the chairman/judge can only *reason* about candidate
answers, never check them. This one is **verifiable-executable**: HumanEval
problems ship hidden unit tests, so the judge can actually *run* candidate
solutions.

- **Benchmark**: OpenAI HumanEval (`openai_humaneval`, 164 problems), loaded by
  `backend/evaluation/humaneval.py` with the same seed-42 shuffle HLE uses.
- **Grading**: `backend/execution.py` runs `prompt + completion + test +
  check(entry_point)` in a subprocess with a hard timeout — pass/fail is
  ground truth, not a judgment.
- **Council job = selection**: on each problem the council picks *which
  member's solution to submit*; correctness = whether that solution passes.
  We work on the **disagreement** subset (≥1 member passed AND ≥1 failed) —
  the only place selection matters. On that subset the union-of-members
  ceiling is 100% by construction (some member always passed); the best single
  member sits below it.

## The `--verify` hypothesis

The whole experiment turns on one flag:

- **read-only (default)**: the chairman/judge/reviewers only read the code and
  reason — pure judgment, exactly like HLE. On HLE, no council structure beat
  the best member.
- **`--verify`**: each candidate solution is additionally annotated with its
  ground-truth unit-test result (`[VERIFIED TEST RESULT: PASSED/FAILED]`) — the
  council can "run the tests."

**Hypothesis**: verification is the missing ingredient. In `--verify` mode the
council should approach the union-of-members ceiling (≈100% selection on the
disagreement subset) and thereby beat its best member — the result judged MCQ
could not produce. Read-only mode is the control: judgment without an oracle.

Structures implemented (mirroring the MCQ runner): **cabinet** (one chairman
selects), **court** (one advocate per distinct solution, then a judge selects),
**peer_review** (members rank, Borda picks). **Jury is deliberately omitted** —
a majority vote needs discrete matching options, and open-ended code has no
natural "same answer" equivalence class, so it is ill-defined for code. We do
not fake it.

## Run commands

Stage-1 first (generate + execute solo baselines), then the council. The
Fable-free / near-equals council makes the "beat the best member" comparison
sharpest (no single dominant chair):

```bash
# 0. Smoke / cost check (2 problems, no real spend gate needed to eyeball)
python -m experiments.run_coding_stage1 --smoke
python -m experiments.run_coding_stage1 --dry-run

# 1. Stage-1 pass@1 on full HumanEval (default council = COUNCIL_V2_MODELS)
python -m experiments.run_coding_stage1 --n 164

# ...or the Fable-free council (GPT + Gemini + Grok + Opus)
python -m experiments.run_coding_stage1 --n 164 \
  --council "openai/gpt-5.6-sol,google/gemini-3.1-pro-preview,x-ai/grok-4.5,anthropic/claude-opus-4.8"

# 2. Council — dry-run both modes to see disagreement count + cost
python -m experiments.run_coding_council --dry-run                 # read-only
python -m experiments.run_coding_council --dry-run --verify        # verify

# 3. Council read-only (control) and verify (treatment), Fable-free
COUNCIL="openai/gpt-5.6-sol,google/gemini-3.1-pro-preview,x-ai/grok-4.5,anthropic/claude-opus-4.8"
python -m experiments.run_coding_council --council "$COUNCIL"            # read-only
python -m experiments.run_coding_council --council "$COUNCIL" --verify   # verify

# 4. Analyze (per structure: selection accuracy vs best-member vs union,
#    split by verify mode)
python -m experiments.run_coding_council --analyze-only --council "$COUNCIL"
```

Both stage-1 and council are checkpointed (atomic write + `.bak`, resume from
completed cells, errored records retry). The verify/read-only mode is part of
the council checkpoint key, so the two runs coexist in one results file.

## Cost estimate

**~$50–70** total, gated on user approval, scaling with the disagreement count:

- Stage-1: 3 paid (non-Anthropic) members × 164 problems ≈ 490 paid calls.
  Anthropic members (Fable/Opus) are $0 on research credits.
- Council: per disagreement problem per mode, court + peer_review each buy a
  few paid calls (cabinet and all chairs/judges are anthropic-direct = $0).
  Two modes (read-only + verify) × 3 structures.

Use the `--dry-run` output (paid OpenRouter calls @ $0.10/call for coding,
since outputs are longer than MCQ) for the exact number before spending.

## Safety caveat — executes untrusted code

`backend/execution.py` runs **model-generated code**. Isolation is a separate
subprocess with a hard wall-clock timeout and stdin closed — the same
pragmatic standard as OpenAI's original `human-eval` harness. **This is not a
security sandbox**: model code can still touch the filesystem, open sockets, or
consume CPU/memory. On this single-user research machine that trade-off is
accepted; any production or adversarial use MUST run inside a
container/VM/gVisor with network and filesystem locked down.

## Merge plan

1. Land stage-1 + executor + loader (`backend/execution.py`,
   `backend/evaluation/humaneval.py`, `experiments/run_coding_stage1.py`) with
   their tests — these are self-contained and touch no existing code.
2. Land the council runner (`experiments/run_coding_council.py`) — it reuses
   `run_phase2_scout`'s checkpoint helpers and mirrors `run_council_types`
   patterns; no changes to the MCQ path.
3. Run stage-1, eyeball pass@1 and the disagreement count via `--analyze-only`
   / `--dry-run`, then run the council in both modes.
4. Report the read-only-vs-verify contrast alongside the HLE result: HLE showed
   no structure beats the best member under judgment; coding tests whether
   verification changes that.

Tests: `pytest tests/test_humaneval.py tests/test_execution.py
tests/test_coding_stage1.py tests/test_coding_council.py -v` (51 tests). The
existing `tests/test_council_types.py` (31 tests) is unaffected.
