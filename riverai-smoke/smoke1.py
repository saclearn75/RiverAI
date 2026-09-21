"""Run a small, configurable River RL smoke test against GSM8K.

The parent directory's .env file must define RIVERAI_API_KEY, ENGINE, and
REPO_ROOT. Optional .env settings tune the workload; defaults stay small.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Keep routine dependency downloads quiet; River's own progress and errors stay visible.
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
# Load the parent .env first, then respect its optional REPO_ROOT override.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent
load_dotenv(DEFAULT_REPO_ROOT / ".env")
REPO_ROOT = Path(os.getenv("REPO_ROOT", DEFAULT_REPO_ROOT)).expanduser()
if not REPO_ROOT.is_absolute():
    REPO_ROOT = (DEFAULT_REPO_ROOT / REPO_ROOT).resolve()
load_dotenv(REPO_ROOT / ".env")


def required_setting(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required setting {name!r} in {REPO_ROOT / '.env'}.")
    return value


API_KEY = required_setting("RIVERAI_API_KEY")
ENGINE = required_setting("ENGINE")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
GROUP_SIZE = int(os.getenv("GROUP_SIZE", "2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "64"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "1"))
LORA_RANK = int(os.getenv("LORA_RANK", "8"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "4e-5"))
STOP_TOKEN = os.getenv("STOP_TOKEN", "<|im_end|>")
CHECKPOINT_NAME = os.getenv("CHECKPOINT_NAME")


# --- Reward and prompt formatting -----------------------------------------
def extract_boxed(text: str) -> str | None:
    """Return the contents of the last LaTex \boxed{...} expression."""
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
    """Read GSM8K's conventional final-answer line: ``#### 42``."""
    for line in reversed(answer.splitlines()):
        if line.strip().startswith("####"):
            return line.strip()[4:].strip().lstrip(":").replace(",", "").strip()
    return answer.strip()


def normalize_answer(value: str) -> str:
    """Normalize ordinary numeric answer formatting before comparison."""
    value = value.replace(",", "").replace("$", "").replace(" ", "")
    match = re.search(r"-?\d+\.?\d*", value)
    return match.group(0) if match else value


def reward(response: str, reference_answer: str) -> float:
    """Reward a response only when its boxed result matches GSM8K's answer."""
    boxed = extract_boxed(response)
    if boxed is None:
        return 0.0
    return float(normalize_answer(boxed) == normalize_answer(extract_gsm8k_answer(reference_answer)))


def make_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    """Render one question through the selected engine's chat template."""
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
        # Older tokenizer templates do not accept enable_thinking.
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# --- River session and one-or-more RL steps --------------------------------
def main() -> None:
    client = river.Client(api_key=API_KEY)
    capabilities = client.get_capabilities()
    if ENGINE not in capabilities:
        available = "\n  ".join(capabilities) or "(none)"
        raise RuntimeError(f"ENGINE={ENGINE!r} is not enabled for this API key. Available engines:\n  {available}")

    print("healthy:", client.health_check())
    print("engine:", ENGINE)
    print(f"smoke configuration: batch={BATCH_SIZE}, group={GROUP_SIZE}, max_tokens={MAX_TOKENS}, steps={MAX_STEPS}")

    tokenizer = AutoTokenizer.from_pretrained(ENGINE)
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    total_steps = min(MAX_STEPS, (len(gsm8k) + BATCH_SIZE - 1) // BATCH_SIZE)
    stop = [STOP_TOKEN] if STOP_TOKEN else None

    with client.session(project="gsm8k-smoke") as session:
        model = session.create_model(
            base_model=ENGINE,
            lora=river.LoraConfig(rank=LORA_RANK, train_attn=True, train_mlp=True, train_unembed=True),
        )
        print("model_id:", model.model_id)

        for step in range(total_steps):
            start = step * BATCH_SIZE
            rows = gsm8k.select(range(start, min(start + BATCH_SIZE, len(gsm8k))))
            prompts = [make_prompt(tokenizer, question) for question in rows["question"]]
            prompt_tokens = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]

            sample_args = {
                "prompts": prompts,
                "num_samples": GROUP_SIZE,
                "max_tokens": MAX_TOKENS,
                "seed": step * len(prompts) * GROUP_SIZE,
            }
            if stop:
                sample_args["stop"] = stop
            groups = model.sample(**sample_args)

            # Build the token-aligned fields needed for importance-sampling RL.
            train_data = []
            rewards_for_step = []
            for tokens, samples, answer in zip(prompt_tokens, groups, rows["answer"]):
                rewards = [reward(sample.text, answer) for sample in samples]
                rewards_for_step.extend(rewards)
                baseline = sum(rewards) / len(rewards)
                if all(value == baseline for value in rewards):
                    continue

                prompt_length = len(tokens)
                for sample, value in zip(samples, rewards):
                    advantage = value - baseline
                    train_data.append(
                        {
                            "input_ids": tokens + sample.tokens,
                            "attention_mask": [1] * (prompt_length + len(sample.tokens)),
                            "old_logprobs": [0.0] * (prompt_length - 1) + sample.logprobs + [0.0],
                            "advantages": [0.0] * (prompt_length - 1) + [advantage] * len(sample.tokens) + [0.0],
                        }
                    )

            if train_data:
                result = model.forward_backward(train_data, loss_fn="importance_sampling")
                model.optim_step(lr=LEARNING_RATE, beta1=0.9, beta2=0.95, eps=1e-8)
                print(f"step {step + 1}: loss={result.metrics['loss']:.4f}")
            else:
                print(f"step {step + 1}: skipped (all groups had zero advantage)")

            mean_reward = sum(rewards_for_step) / len(rewards_for_step)
            print(f"step {step + 1}: mean_reward={mean_reward:.3f}")

        # Saving creates a durable remote artifact, so it is opt-in.
        if CHECKPOINT_NAME:
            checkpoint = model.save_weights(CHECKPOINT_NAME, mode="inference")
            print("saved checkpoint:", checkpoint.path)


if __name__ == "__main__":
    main()
