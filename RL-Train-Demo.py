"""Small, observable River GSM8K reinforcement-learning demonstration.

Required .env settings: RIVERAI_API_KEY, ENGINE, REPO_ROOT, CHECKPOINT_NAME.
Training uses GSM8K's train split only; baseline and post-training evaluation
use the same held-out GSM8K test questions.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Silences HF messages when downloading datasets/transformers packages.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import logging as huggingface_logging
import river_client as river
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

huggingface_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()


# --- Configuration ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")
REPO_ROOT = Path(os.getenv("REPO_ROOT", SCRIPT_DIR)).expanduser()
if not REPO_ROOT.is_absolute():
    REPO_ROOT = (SCRIPT_DIR / REPO_ROOT).resolve()
load_dotenv(REPO_ROOT / ".env")


def required_setting(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required setting {name!r} in {REPO_ROOT / '.env'}.")
    return value


API_KEY = required_setting("RIVERAI_API_KEY")
ENGINE = required_setting("ENGINE")
CHECKPOINT_NAME = required_setting("CHECKPOINT_NAME")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
GROUP_SIZE = int(os.getenv("GROUP_SIZE", "4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "5"))
EVAL_SIZE = int(os.getenv("EVAL_SIZE", "30"))
LORA_RANK = int(os.getenv("LORA_RANK", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "4e-5"))
STOP_TOKEN = os.getenv("STOP_TOKEN", "<|im_end|>")
EVAL_SEED_BASE = int(os.getenv("EVAL_SEED_BASE", "20260921"))
RESULTS_DIR = REPO_ROOT / "LoRA-results"
RESULTS_PATH = RESULTS_DIR / "results.json"


# --- GSM8K answer scoring and engine-specific prompt rendering ------------
def extract_boxed(text: str) -> str | None:
    """Return the contents of the final LaTeX \boxed{...} expression."""
    found: list[str] = []
    opening_braces: list[int] = []
    for index, character in enumerate(text):
        if character == "{":
            opening_braces.append(index)
        elif character == "}" and opening_braces:
            start = opening_braces.pop()
            if text[:start].endswith("\\boxed"):
                found.append(text[start + 1 : index])
    return found[-1] if found else None


def extract_gsm8k_answer(answer: str) -> str:
    """Read GSM8K's conventional final-answer line, for example ``#### 42``."""
    for line in reversed(answer.splitlines()):
        if line.strip().startswith("####"):
            return line.strip()[4:].strip().lstrip(":").replace(",", "").strip()
    return answer.strip()


def normalize_answer(value: str) -> str:
    """Normalize ordinary numeric formatting before answer comparison."""
    value = value.replace(",", "").replace("$", "").replace(" ", "")
    match = re.search(r"-?\d+\.?\d*", value)
    return match.group(0) if match else value


def reward(response: str, reference_answer: str) -> float:
    """Reward only a boxed numerical answer that matches the GSM8K reference."""
    boxed = extract_boxed(response)
    if boxed is None:
        return 0.0
    return float(normalize_answer(boxed) == normalize_answer(extract_gsm8k_answer(reference_answer)))


def make_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    """Render a question with the configured engine's native chat template."""
    instruction = (
        " Solve the problem. Provide a numerical answer without units, "
        "written inside \\boxed{} at the end of your response."
    )
    messages = [{"role": "user", "content": question + instruction}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# --- Held-out evaluation ---------------------------------------------------
def evaluate(
    model: object,
    tokenizer: AutoTokenizer,
    test_split: object,
    indices: list[int],
) -> dict[str, object]:
    """Evaluate one deterministic completion per held-out question in one batch."""
    stop = [STOP_TOKEN] if STOP_TOKEN else None
    records: list[dict[str, object]] = []
    rows = test_split.select(indices)
    prompts = [make_prompt(tokenizer, question) for question in rows["question"]]
    sample_args: dict[str, object] = {
        "prompts": prompts,
        "num_samples": 1,
        "max_tokens": MAX_TOKENS,
        "seed": EVAL_SEED_BASE,
    }
    if stop:
        sample_args["stop"] = stop
    groups = model.sample(**sample_args)
    if len(groups) != len(indices):
        raise RuntimeError(f"Expected {len(indices)} evaluation groups, received {len(groups)}.")

    for index, question, answer, samples in zip(indices, rows["question"], rows["answer"], groups):
        if len(samples) != 1:
            raise RuntimeError(f"Test index {index} returned {len(samples)} samples, expected 1.")
        response = samples[0].text
        extracted = extract_boxed(response)
        reference = extract_gsm8k_answer(answer)
        records.append(
            {
                "test_index": index,
                "reference_answer": reference,
                "extracted_answer": extracted,
                "reward": reward(response, answer),
            }
        )

    correct = sum(record["reward"] for record in records)
    missing_boxed = sum(record["extracted_answer"] is None for record in records)
    count = len(records)
    return {
        "correct": int(correct),
        "evaluated": count,
        "accuracy": correct / count if count else 0.0,
        "missing_boxed": missing_boxed,
        "records": records,
    }


def print_evaluation(label: str, evaluation: dict[str, object]) -> None:
    print(
        f"{label}: {evaluation['correct']} / {evaluation['evaluated']} = "
        f"{evaluation['accuracy']:.3f}; missing_boxed={evaluation['missing_boxed']}"
    )


# --- Training and observability -------------------------------------------
def print_example(label: str, example: dict[str, object]) -> None:
    print(
        f"example_{label}: step={example['step']}, train_index={example['train_index']}, "
        f"reference={example['reference_answer']!r}, extracted={example['extracted_answers']}, "
        f"rewards={example['rewards']}"
    )


def main() -> None:
    # Keep local experiment artifacts together, regardless of the shell cwd.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = river.Client(api_key=API_KEY)
    capabilities = client.get_capabilities()
    if ENGINE not in capabilities:
        available = "\n  ".join(capabilities) or "(none)"
        raise RuntimeError(f"ENGINE={ENGINE!r} is not enabled for this API key. Available engines:\n  {available}")
    if not client.health_check():
        raise RuntimeError("River health check returned false.")

    tokenizer = AutoTokenizer.from_pretrained(ENGINE)
    train_split = load_dataset("openai/gsm8k", "main", split="train")
    test_split = load_dataset("openai/gsm8k", "main", split="test")
    eval_indices = list(range(min(EVAL_SIZE, len(test_split))))
    total_steps = min(MAX_STEPS, (len(train_split) + BATCH_SIZE - 1) // BATCH_SIZE)
    stop = [STOP_TOKEN] if STOP_TOKEN else None

    configuration = {
        "batch_size": BATCH_SIZE,
        "group_size": GROUP_SIZE,
        "max_tokens": MAX_TOKENS,
        "max_steps": total_steps,
        "eval_size": len(eval_indices),
        "lora_rank": LORA_RANK,
        "learning_rate": LEARNING_RATE,
        "stop_token": STOP_TOKEN,
        "eval_seed_base": EVAL_SEED_BASE,
    }
    print("healthy: True")
    print("engine:", ENGINE)
    print("configuration:", configuration)
    print(f"dataset separation: train={len(train_split)}, held_out_test={len(test_split)}")

    step_summaries: list[dict[str, object]] = []
    examples: dict[str, dict[str, object] | None] = {"mixed": None, "zero_advantage": None}

    # The model object combines frozen base weights with trainable LoRA weights.
    # Session provisioning can queue remotely.  Surface that state and bound it
    # so an unavailable service does not leave an invisible day-long process.
    with client.session(
        project="gsm8k-learning-run",
        timeout=300,
        on_session_creation_attempted=lambda: print("requesting River training session...", flush=True),
        on_session_created=lambda: print("River training session ready.", flush=True),
        on_session_closed=lambda: print("River training session closed.", flush=True),
    ) as session:
        model = session.create_model(
            base_model=ENGINE,
            lora=river.LoraConfig(rank=LORA_RANK, train_attn=True, train_mlp=True, train_unembed=True),
        )
        print("model_id:", model.model_id)

        ## 1. Baseline evaluation

        print("\nBaseline evaluation (held-out test split; no weight updates):")
        baseline = evaluate(model, tokenizer, test_split, eval_indices)
        print_evaluation("Baseline", baseline)

        # Each step uses a disjoint contiguous slice of the train split.
        for step in range(total_steps):
            start = step * BATCH_SIZE
            train_indices = list(range(start, min(start + BATCH_SIZE, len(train_split))))
            rows = train_split.select(train_indices)
            prompts = [make_prompt(tokenizer, question) for question in rows["question"]]
            prompt_tokens = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
            sample_args: dict[str, object] = {
                "prompts": prompts,
                "num_samples": GROUP_SIZE,
                "max_tokens": MAX_TOKENS,
                "seed": step * len(prompts) * GROUP_SIZE,
            }
            if stop:
                sample_args["stop"] = stop

            # 2. Model Training 
            groups = model.sample(**sample_args)
            if len(groups) != len(prompts):
                raise RuntimeError(f"Expected {len(prompts)} sampled groups, received {len(groups)}.")

            train_data: list[dict[str, object]] = []
            all_rewards: list[float] = []
            mixed_groups = all_wrong_groups = all_correct_groups = 0
            for train_index, tokens, samples, answer in zip(train_indices, prompt_tokens, groups, rows["answer"]):
                if len(samples) != GROUP_SIZE:
                    raise RuntimeError(f"Train index {train_index} returned {len(samples)} samples, expected {GROUP_SIZE}.")

                rewards = [reward(sample.text, answer) for sample in samples]
                extracted = [extract_boxed(sample.text) for sample in samples]
               
                all_rewards.extend(rewards)
                baseline_reward = sum(rewards) / len(rewards)
                example = {
                    "step": step + 1,
                    "train_index": train_index,
                    "reference_answer": extract_gsm8k_answer(answer),
                    "extracted_answers": extracted,
                    "rewards": rewards,
                }
                if all(value == 0.0 for value in rewards):
                    all_wrong_groups += 1
                elif all(value == 1.0 for value in rewards):
                    all_correct_groups += 1
                else:
                    mixed_groups += 1
                    if examples["mixed"] is None:
                        examples["mixed"] = example
                        print_example("mixed", example)
                if all(value == baseline_reward for value in rewards) and examples["zero_advantage"] is None:
                    examples["zero_advantage"] = example
                    print_example("zero_advantage", example)
                if all(value == baseline_reward for value in rewards):
                    continue

                prompt_length = len(tokens)
                for sample, value in zip(samples, rewards):
                    advantage = value - baseline_reward
                    train_data.append(
                        {
                            "input_ids": tokens + sample.tokens,
                            "attention_mask": [1] * (prompt_length + len(sample.tokens)),
                            "old_logprobs": [0.0] * (prompt_length - 1) + sample.logprobs + [0.0],
                            "advantages": [0.0] * (prompt_length - 1) + [advantage] * len(sample.tokens) + [0.0],
                        }
                    )

            optimizer_applied = False
            loss: float | None = None
            if train_data:
                # 3. Use advantages and log probabilities to calculate loss and gradients 
                result = model.forward_backward(train_data, loss_fn="importance_sampling")
                loss = float(result.metrics["loss"])

                # 4. Applies those gradients to the trainable LoRA weights
                model.optim_step(lr=LEARNING_RATE, beta1=0.9, beta2=0.95, eps=1e-8)
                optimizer_applied = True

            summary = {
                "step": step + 1,
                "questions": len(rows),
                "completions": len(all_rewards),
                "correct_completions": int(sum(all_rewards)),
                "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
                "mixed_groups": mixed_groups,
                "all_wrong_groups": all_wrong_groups,
                "all_correct_groups": all_correct_groups,
                "training_records": len(train_data),
                "optimizer_applied": optimizer_applied,
                "loss": loss,
            }
            step_summaries.append(summary)
            print(
                f"step={summary['step']} questions={summary['questions']} completions={summary['completions']} "
                f"correct_completions={summary['correct_completions']} mean_reward={summary['mean_reward']:.3f} "
                f"mixed_groups={summary['mixed_groups']} all_wrong_groups={summary['all_wrong_groups']} "
                f"all_correct_groups={summary['all_correct_groups']} training_records={summary['training_records']} "
                f"optimizer_applied={str(summary['optimizer_applied']).lower()} "
                f"loss={summary['loss'] if summary['loss'] is not None else 'n/a'}"
            )

        checkpoint = model.save_weights(CHECKPOINT_NAME, mode="inference")
        print("saved checkpoint:", checkpoint.path)

        # 5. Post Training Evaluation. 
        print("\nPost-training evaluation (the same held-out test indices):")
        post_training = evaluate(model, tokenizer, test_split, eval_indices)
        print_evaluation("Post-training", post_training)

    delta = post_training["accuracy"] - baseline["accuracy"]
    print(f"Delta: {delta:+.3f} (small-sample comparison; not a benchmark)")
    results = {
        "engine": ENGINE,
        "configuration": configuration,
        "baseline_evaluation": baseline,
        "training_steps": step_summaries,
        "post_training_evaluation": post_training,
        "checkpoint": checkpoint.path,
        "accuracy_delta": delta,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("results written:", RESULTS_PATH)


if __name__ == "__main__":
    main()
