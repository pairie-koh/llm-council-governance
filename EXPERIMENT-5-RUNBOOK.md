# Experiment 5: Fable-free council (near-equals regime) — runbook

Branch `experiment-5-fable-free`. This is the parallel to the Fable-inclusive
council-types experiment. It removes the dominant member (Fable 5) to test
whether councils flip from harmful to helpful when the field is near-equal.

**This is a *complement* to the Fable-inclusive results, not a replacement.**
The contrast between the two regimes is the finding: dominant-member (Fable in)
vs near-equals (Fable out). Report both side by side.

## Council

`{GPT-5.6 Sol, Gemini 3.1 Pro, Grok 4.5, Opus 4.8}` — the four non-Fable models.
Opus chairs/judges (free on the Anthropic research key). Opus's high error
correlation was specifically with Fable (same-wrong 88%); with Fable removed,
Opus is a normal member (phi 0.53-0.63 with the others). Field flattens to
near-equals: GPT 50.8%, Opus 50.8%, Gemini 53.6%, Grok 48.5% on the shared
clean set.

## Status

| Structure | Cost | State |
|-----------|------|-------|
| jury | $0 (offline) | DONE — 49.8% vs best member 53.6% (near break-even; dominance penalty removed, correlation penalty ~4pt remains) |
| cabinet (Opus chair) | $0 (Anthropic key) | DONE / running free — see results_council_fablefree/ |
| court (Opus judge) | ~$26-31 | BLOCKED on funded OpenRouter key |
| peer_review | ~$35-42 | BLOCKED on funded OpenRouter key |

Free structures already computed to `experiments/results_council_fablefree/`.

## To finish (when a funded OpenRouter key is in `.env`)

The `--council` flag is already merged to main. From the repo root:

```bash
# court + peer review on the Fable-free council; ~$60-75, ~2-4 hours
PYTHONUTF8=1 python -m experiments.run_council_types \
  --council "openai/gpt-5.6-sol,google/gemini-3.1-pro-preview,x-ai/grok-4.5,anthropic/claude-opus-4.8" \
  --types court,peer_review \
  --output-dir experiments/results_council_fablefree

# then the full analysis (all 4 structures)
PYTHONUTF8=1 python -m experiments.run_council_types \
  --council "openai/gpt-5.6-sol,google/gemini-3.1-pro-preview,x-ai/grok-4.5,anthropic/claude-opus-4.8" \
  --output-dir experiments/results_council_fablefree \
  --analyze-only
```

Launch the paid run via the Task Scheduler pattern (see
`experiments/stage1_auto_resume.ps1`) so a reboot doesn't lose it. Checkpoint
is atomic + `.bak`, saves every 5 records; a crash re-buys at most 5 cells.

## Cost note

Dry-run prints ~$162 at the conservative $0.223/call estimate; real per-call
cost measured at ~$0.084 (medium-length revision/ballot/brief), so budget
~$60-75. 728 paid calls (420 peer-review ballots + 308 court briefs); the
420 Opus cabinet/judge/ballot calls are free.

## Interpretation caveat

In cabinet and court, Opus is *both* a member and the chair/judge (mirrors
Fable's dual role in the original). A lift there is partly recognition and
partly Opus picking its own answer back out of the slate. Peer review is the
cleaner recognition test since Borda aggregates all four ballots symmetrically.

## Optional follow-on

A 3-vendor council `{GPT, Gemini, Grok}` (one model per lab, odd-sized so the
jury cannot tie) would test genuine cross-lab diversity harder. Needs a code
tweak: the anonymization labels (`COUNCILOR_LABELS`) and the RANKING regex are
hardwired to 4 members. Separate experiment, not required for Experiment 5.

## Merge plan

When the paid run completes: commit results here, fold an "Experiment 5:
Fable-free council" section into `FINDINGS-2026-07-17.md` (dominance penalty
~9pt vs correlation penalty ~4pt decomposition), then merge this branch to
`frontier-rerun` and `main`.
