# River AI GSM8K LoRA demo

This repository contains a small, observable reinforcement-learning (RL) experiment using River AI and the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) grade-school-math dataset. The official demo is `RL-Train-Demo.py`.

The experiment creates a LoRA adapter for the configured River model, measures its initial performance on held-out questions, performs a bounded number of group-relative RL updates on separate training questions, saves a checkpoint, and evaluates the same held-out questions again.

## Prerequisites

1. Create or sign in to a [River AI account](https://river.ai/).
2. Open the [River API console](https://api.river.ai/) and create an API key. Treat the key like a password: do not commit it or share it.

## Setup

Use Python 3.12 or newer and create a virtual environment at the repository root. Install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create `.env` by copying `.env.example`, then set `RIVERAI_API_KEY` to your River API key. Keep `.env` private: it is intentionally ignored by Git.

```powershell
Copy-Item .env.example .env
```

Run the experiment from the repository root:

```powershell
.\.venv\Scripts\python.exe .\RL-Train-Demo.py
```

The settings in `.env` keep this a deliberately bounded demo: 30 evaluation questions, five training steps, 16 questions per step, and four completions per question. River API usage and checkpoint storage may incur account usage.

## Outputs

- `LoRA-results/results.json` — configuration, baseline and post-training held-out evaluation records, step summaries, checkpoint path, and accuracy delta.
- River checkpoint — a durable remote LoRA artifact, saved under `CHECKPOINT_NAME`.

## Architecture

```text
.env
  │ configuration and River API key
  ▼
RL-Train-Demo.py
  ├─ River client ──► remote session ──► LoRA model ──► remote checkpoint
  ├─ Hugging Face tokenizer ──► engine-native chat prompts
  └─ GSM8K dataset
       ├─ test split ──► baseline evaluation ────────────┐
       └─ train split ──► grouped samples ─► rewards     │
                                │                         │
                                └─ advantages ─► LoRA update
                                                          │
                                   post-training evaluation
                                                          ▼
                                           LoRA-results/results.json
```

### Experiment flow

1. Load the configured engine, tokenizer, and separate GSM8K `train` and `test` splits.
2. Evaluate one deterministic completion for each of the first `EVAL_SIZE` test questions before any optimizer update.
3. For each training step, sample `GROUP_SIZE` completions for each previously unused training question.
4. Score a completion `1` only when its final `\\boxed{}` answer matches the GSM8K reference; otherwise score `0`.
5. Center rewards within each group to produce advantages. Uniformly correct or incorrect groups have zero advantage and are skipped.
6. Submit token-aligned data to River for an importance-sampling forward/backward pass and optimizer step when nonzero advantages exist.
7. Save the LoRA weights, reevaluate the identical held-out indices, and write the result report.

The comparison is an educational smoke test, not a benchmark: it is small, stochastic in training, and does not establish model quality.
