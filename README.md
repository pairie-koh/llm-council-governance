# LLM Council Governance Study

Which governance procedure should a council of LLMs use? Extends [Karpathy's llm-council](https://github.com/karpathy/llm-council) with multiple governance implementations and an evaluation harness.

The study has two phases:

- **Phase 1** — seven governance structures on MMLU-Pro Math / GSM8K / TruthfulQA-class benchmarks, first with small (7-9B) models, then rerun with 2026 frontier models. Result: councils beat the best member at the small-model tier, but **the jury premise fails at the frontier** — majority vote cannot beat the best member when member errors are strongly correlated, and at the frontier they are.
- **Phase 2** — updating-vs-herding with a frontier council ("council v2") on Humanity's Last Exam (HLE), the one benchmark hard enough that frontier models still disagree. Instead of asking "does the council beat the baseline on average," phase 2 asks what happens on the specific questions where members disagree: does deliberation propagate correct minority answers, or does the council herd?

---

## Phase 1: Small-Model Results (MMLU-Pro Math)

We evaluated 7 governance structures on **MMLU-Pro Math** with four 7-9B models. The baseline is the **best-performing individual model** (Qwen 2.5 7B at 71.7% accuracy).

![Structure Performance](paper/structure_performance.png)

### Structure Performance

| Structure | Accuracy | vs Baseline | N |
|-----------|----------|-------------|---|
| **Deliberate → Vote** | **80.9%** | **+9.2pp** | 136 |
| Majority Vote | 76.8% | +5.1pp | 142 |
| Deliberate → Synthesize | 75.4% | +3.7pp | 142 |
| Agenda Setter + Veto | 74.1% | +2.4pp | 147 |
| *Baseline: Qwen 2.5 7B* | *71.7%* | — | *847* |
| Rank → Synthesize | 70.7% | -1.0pp | 147 |
| Weighted Majority Vote | 69.0% | -2.7pp | 142 |
| Self-Consistency Vote | 68.9% | -2.8pp | 148 |

*1,004 valid trials from 1,050 total (95.6% completion rate). Baseline accuracy computed from stage-1 responses across all council structures.*

### Individual Model Performance

![Model Performance](paper/model_performance.png)

| Model | Accuracy | N |
|-------|----------|---|
| Qwen 2.5 7B | 71.7% | 847 |
| Llama 3.1 8B | 62.7% | 839 |
| Gemma 2 9B | 60.4% | 848 |
| Mistral 7B | 44.0% | 75 |

*Mistral 7B had significantly fewer valid responses due to API timeout issues.*

### Key Findings (small-model tier)

1. **Four of seven structures beat the best individual model** — by 2.4-9.2pp.
2. **Deliberate → Vote won (+9.2pp)**: answer independently, see each other's reasoning and optionally revise, then vote.
3. **Simple majority vote was second (+5.1pp)** — model diversity pays even without deliberation.
4. **Self-consistency (-2.8pp) and weighted voting (-2.7pp) fell below the best member.** Sampling one model 9x doesn't add the diversity different models provide; historical accuracy weights don't transfer.
5. **Synthesis works after deliberation (+3.7pp), not after ranking (-1.0pp).**

### The Seven Governance Structures

| Structure | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| **Deliberate→Vote** | 4 models answer independently | Each model sees others' answers, can revise | Take majority vote |
| **Majority Vote** | 4 models answer independently | — | Take majority vote (equal weights) |
| **Deliberate→Synthesize** | 4 models answer independently | Each model sees others' answers, can revise | Chairman synthesizes |
| **Agenda Setter + Veto** | 4 models answer independently | Chairman proposes answer | Council votes ACCEPT/VETO; fallback to majority |
| **Rank→Synthesize** | 4 models answer independently | Each model ranks all answers | Chairman synthesizes based on rankings |
| **Weighted Vote** | 4 models answer independently | — | Weighted majority vote (by historical accuracy) |
| **Self-Consistency Vote** | Single model sampled 9× with temp=0.7 | — | Take majority vote across samples |

Small-model council (via OpenRouter): `meta-llama/llama-3.1-8b-instruct`, `mistralai/mistral-7b-instruct`, `google/gemma-2-9b-it`, `qwen/qwen-2.5-7b-instruct`. Self-Consistency Vote uses Llama 3.1 8B, 9 samples at temperature 0.7.

### Pilot Study (GSM8K + TruthfulQA)

1,680 supporting trials on easier benchmarks (~85% baseline accuracy — little headroom). No structure separated significantly from the best member. Two useful negatives:

- **Prompt and persona diversity did not help.** A single model with varied prompts (84.1%) or personas (83.0%) did not beat its own plain baseline (84.5%). The diversity that matters comes from different models.
- **Deliberation flipped 10-15% of answers, net +50 fixed vs broken.** Weaker models deferred to stronger ones.

### Phase 1 at the Frontier: the Jury Premise Fails

We reran all seven structures with 2026 frontier models (GPT-5.5, Gemini 3.1 Pro, Claude Opus 4.8 as chairman, Grok 4.5) on MMLU-Pro Math, GSM8K, TruthfulQA, and AIMO level-5 olympiad math (`experiments/run_frontier_full.py`, results in `experiments/results_frontier_full/`).

The small-model result does not transfer. Majority vote — and council structures generally — cannot beat the best member at the frontier because member errors are strongly correlated: when frontier models are wrong, they tend to be wrong together, and often on the same wrong answer. The Condorcet jury theorem needs independent errors; frontier councils don't have them. `experiments/analyze_frontier.py` computes the pairwise error-correlation matrix and correct-minority survival directly.

This is what motivates phase 2: average accuracy over a full benchmark is the wrong lens. The action is entirely inside the disagreement cells.

---

## Phase 2: Updating vs Herding on HLE

Phase 2 moves to **Humanity's Last Exam** (Phan et al. 2025, text-only multiple-choice subset, 560 questions) — hard enough (frontier models score ~45-53%) that disagreement cells still exist. The question: when council members disagree and the lone dissenter is right, which governance structure preserves the correct minority answer instead of herding to the confident majority?

### Council v2

Strongest model per vendor as of July 2026 (slugs verified on OpenRouter 2026-07-15):

| Role | Model | Slug |
|------|-------|------|
| Member | GPT-5.6 Sol | `openai/gpt-5.6-sol` |
| Member | Gemini 3.1 Pro | `google/gemini-3.1-pro-preview` |
| Member + chairman | Claude Fable 5 | `anthropic/claude-fable-5` |
| Member | Grok 4.5 | `x-ai/grok-4.5` |
| Side-arm (not a member) | Claude Opus 4.8 | `anthropic/claude-opus-4.8` |

Opus 4.8 rides along solely to measure **intra-vendor error correlation** (same vendor, different model generation) against the cross-vendor pairs.

### Scout Findings (2026-07-16, 80 questions, $28.85)

`experiments/run_phase2_scout.py` ran one independent answer per (model, question) on 80 HLE questions — no council structures — to check whether the full study is worth buying. Numbers below reproduce with `python -m experiments.run_phase2_scout --analyze-only` (offline).

**Per-model accuracy** (refusals and non-answers count against accuracy):

| Model | Accuracy | Notes |
|-------|----------|-------|
| Claude Fable 5 | 52.5% | 18/80 refused (see exclusion below) |
| Gemini 3.1 Pro | 50.0% | |
| GPT-5.6 Sol | 48.8% | |
| Grok 4.5 | 43.8% | |
| Claude Opus 4.8 (side-arm) | 40.0% | zero refusals |

**Disagreement exists.** Of the 54 questions where all four council members produced a clean answer: 23 all-correct, 11 all-wrong, and **20 split — a 37% disagreement rate on clean questions**. 7 cells were minority-correct (exactly one member right); Fable 5 was the lone dissenter on 5 of them.

**Errors correlate, everywhere.** Pairwise phi on correct/wrong outcomes runs +0.43 to +0.68 across all pairs, and when two models are both wrong they frequently pick the *same* wrong letter (same-wrong rates of 71-100%). The phase-1 frontier failure mode is fully present on HLE.

**The intra-vendor probe hit its ceiling.** Fable 5 × Opus 4.8: phi = +0.601, and on every one of the 17 questions where both were wrong, they chose the **same wrong answer (17/17, 100%)** — the highest same-wrong rate of any pair. Two model generations from one vendor are not two voters.

**Pre-registered exclusion: Biology/Medicine and Chemistry.** Fable 5's bio/chem safety classifiers refuse 67-71% of scout questions in those two HLE categories, versus 0-11% everywhere else. Since Fable is the chairman, those categories are excluded up front (`HLEBenchmark.PHASE2_EXCLUDED_CATEGORIES`), leaving a **374-question eligible pool** out of the 560 text-only MCQs. The filter is applied after the seeded shuffle, so existing checkpoints stay valid. Refusals are recorded as their own outcome category — never silently retried on a fallback model, which would contaminate the measurement.

### Stage 1: Full-Pool Solo Baselines

`experiments/run_phase2_stage1.py` extends the scout measurement (one independent answer per (model, question), 5 models) to all 374 eligible questions. Every council arm re-aggregates or builds on this dataset.

- **Provider routing** (`backend/model_router.py`): `anthropic/*` models go direct to the Anthropic API on research credits ($0 marginal cost to the study budget, `ANTHROPIC_API_KEY` required); GPT-5.6 Sol, Gemini 3.1 Pro, and Grok 4.5 go through OpenRouter. Both paths return identical record shapes, so analysis code never cares who served a call. Direct Anthropic calls use adaptive thinking with 48k max tokens (adaptive thinking can burn the scout's 24k budget inside the thinking block) and a tighter concurrency cap (4 vs 8).
- **Scout-record seeding**: the scout ran the same prompt and models on a prefix of the same seeded shuffle, so its clean records are valid stage-1 records and are adopted rather than re-bought.
- **Checkpointing**: incremental atomic writes (tmp + fsync + `.bak` + atomic replace); rerunning resumes from completed (model, question) pairs and retries errored ones.

### Council Types Experiment

`experiments/run_council_types.py` evaluates four governance types on the **disagreement subset** — the stage-1 questions where council members split. Full-pool accuracy is already known from stage 1; only the contested cells discriminate between structures.

| Type | Mechanism | New API calls |
|------|-----------|---------------|
| **Jury** | Majority vote over stage-1 answers | None (offline re-aggregation) |
| **Cabinet** | Fable 5 chairman reads all four anonymized council answers, then picks | Chairman only |
| **Court** | One advocate argues for each distinct stage-1 answer; Fable 5 judges | Advocates + judge |
| **Peer review** | Each member anonymously ranks all answers; Borda count decides | Members |

Anonymization matters: the chairman and reviewers must not know which vendor wrote which answer, or brand priors contaminate the updating-vs-herding measurement. Results land in `experiments/results_council_types_v2/`.

---

## Setup

```bash
git clone https://github.com/pairie-koh/llm-council-governance.git
cd llm-council-governance

pip install -e .           # package
pip install -e ".[dev]"    # + pytest

cp .env.example .env       # then edit .env
python scripts/check_setup.py
```

Required in `.env`:

- `OPENROUTER_API_KEY` — all non-Anthropic calls (get one at https://openrouter.ai/keys).
- `ANTHROPIC_API_KEY` — optional but expected for phase 2: routes `anthropic/*` models direct to the Anthropic API on research credits. Without it, Anthropic models fall back to OpenRouter (paid).
- `USE_CHEAP_MODELS` — `true` = 7-9B models for phase-1 testing (~$2-5/run), `false` = frontier models (~$150-350/run). Phase-2 runners ignore this; council v2 is hard-coded in `backend/config.py`.

## Usage

Phase 2 (HLE):

```bash
# Scout (solo baselines, default 150 Q)
python -m experiments.run_phase2_scout --smoke          # 5 Q cost/sanity check
python -m experiments.run_phase2_scout --n 150          # scouting pass
python -m experiments.run_phase2_scout --analyze-only   # reprint analysis, offline, free

# Stage 1 (full 374-question eligible pool, seeds clean scout records)
python -m experiments.run_phase2_stage1 --smoke         # 2 Q sanity check (skips seeding)
python -m experiments.run_phase2_stage1                 # full pool
python -m experiments.run_phase2_stage1 --analyze-only  # offline, free

# Council types (jury / cabinet / court / peer review on the disagreement subset)
python -m experiments.run_council_types --smoke         # few cells, cost check
python -m experiments.run_council_types --dry-run       # no API calls, plan only
python -m experiments.run_council_types                 # full run
python -m experiments.run_council_types --analyze-only  # offline, free
```

Phase 1:

```bash
# Frontier rerun (all 7 structures, MMLU-Pro Math + GSM8K + TruthfulQA + AIMO + GPQA)
python -m experiments.run_frontier_full                 # --smoke / --skip-gpqa available
python -m experiments.analyze_frontier --results-dir experiments/results_frontier_full

# Original small-model experiments
python -m experiments.run_mmlu_pro_final
python -m experiments.run_pilot
python -m experiments.analyze_pilot
python -m experiments.run_prompt_experiment
python -m experiments.run_persona_experiment
```

Tests:

```bash
pytest tests/ -v
```

## Project Structure

```
├── backend/
│   ├── config.py              # Model rosters (v1 + council v2), API keys
│   ├── openrouter.py          # OpenRouter API client
│   ├── model_router.py        # Phase-2 provider routing (Anthropic direct vs OpenRouter)
│   ├── governance/            # 7 governance structure implementations
│   └── evaluation/            # Benchmark loaders (GSM8K, TruthfulQA, MMLU-Pro, AIMO, GPQA, HLE)
├── experiments/
│   ├── run_phase2_scout.py    # Phase-2 scouting pass (HLE solo baselines)
│   ├── run_phase2_stage1.py   # Phase-2 full-pool stage 1
│   ├── run_council_types.py   # Phase-2 council types (jury/cabinet/court/peer review)
│   ├── run_frontier_full.py   # Phase-1 frontier rerun
│   ├── analyze_frontier.py    # Error-correlation matrix + minority survival
│   ├── run_mmlu_pro_final.py  # Phase-1 main experiment
│   ├── run_pilot.py           # Phase-1 pilot
│   └── results*/              # Output data (committed deliberately)
├── paper/                     # Figures and charts
└── tests/                     # Test suite
```

## Benchmarks

| Benchmark | Format | N | Role |
|-----------|--------|---|------|
| [HLE](https://huggingface.co/datasets/cais/hle) (text-only MCQ, via ungated mirror `macabdul9/hle_text_only`) | Multiple choice, letter answer | 374 eligible (560 minus bio/chem) | Phase-2 primary |
| [MMLU-Pro Math](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | 10-option MCQ (A-J) | 150 | Phase-1 primary |
| AIMO level 5 | Olympiad math, integer answers | 100 | Phase-1 frontier hard arm |
| GPQA-Diamond | 4-option PhD science (HF-gated, needs `HF_TOKEN`) | 198 | Phase-1 frontier hard arm |
| [GSM8K](https://github.com/openai/grade-school-math) | Word problems, numeric | 40 | Phase-1 pilot (ceilinged at frontier) |
| [TruthfulQA](https://github.com/sylinrl/TruthfulQA) | Binary A/B | 40 | Phase-1 pilot (ceilinged at frontier) |

## Limitations

- Phase-1 small-model results do not generalize to frontier models — that is the phase-1 frontier finding, not a caveat.
- Phase-2 scout N is small (80 questions); the minority-correct count (7) and per-pair phi values carry wide intervals. Stage 1 on the full 374-question pool is what the council-types analysis actually runs on.
- The bio/chem exclusion is a property of the chosen chairman (Fable 5), not of HLE; a different chairman would need its own refusal scout.
- Council overhead (multiple API calls, deliberation rounds) increases latency and cost.
- Statistical significance was not achieved for most pairwise structure comparisons in phase 1.

## License

MIT

## Acknowledgments

- Inspired by [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council); original phase-1 framework by [Andy Hall](https://github.com/andybhall/llm-council-governance)
- Uses [OpenRouter](https://openrouter.ai/) for multi-model API access
- Benchmarks: [HLE](https://arxiv.org/abs/2501.14249), [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [GSM8K](https://github.com/openai/grade-school-math), [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- Related work: [Du et al. (2023)](https://arxiv.org/abs/2305.14325) on multi-agent debate
