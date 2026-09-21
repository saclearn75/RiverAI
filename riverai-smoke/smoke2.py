"""Inspect a small, reproducible sample of the GSM8K dataset.

Install the only required dependency first:
    python -m pip install datasets
"""

from __future__ import annotations

from statistics import mean, median

from datasets import load_dataset


DATASET = "openai/gsm8k"
CONFIG = "main"
SPLIT = "train"
SAMPLE_SIZE = 5
SEED = 42


def describe_lengths(values: list[str]) -> str:
    lengths = [len(value) for value in values]
    return (
        f"min={min(lengths)}, mean={mean(lengths):.1f}, "
        f"median={median(lengths):.1f}, max={max(lengths)}"
    )


def final_answer(answer: str) -> str | None:
    """Extract GSM8K's conventional final-answer marker, e.g. '#### 42'."""
    for line in reversed(answer.splitlines()):
        if line.strip().startswith("####"):
            return line.strip()[4:].strip()
    return None


def main() -> None:
    dataset = load_dataset(DATASET, CONFIG, split=SPLIT)
    questions = dataset["question"]
    answers = dataset["answer"]
    marked_answers = sum(final_answer(answer) is not None for answer in answers)

    print("GSM8K dataset inspection")
    print(f"dataset: {DATASET}")
    print(f"config: {CONFIG}")
    print(f"split: {SPLIT}")
    print(f"rows: {len(dataset):,}")
    print(f"columns: {dataset.column_names}")
    print(f"features: {dataset.features}")
    print(f"question length (characters): {describe_lengths(questions)}")
    print(f"answer length (characters):   {describe_lengths(answers)}")
    print(f"answers with a '####' final-answer marker: {marked_answers:,}/{len(dataset):,}")

    examples = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))
    print(f"\n{SAMPLE_SIZE} examples (deterministic seed={SEED})")
    for index, row in enumerate(examples, start=1):
        print(f"\n--- Example {index} ---")
        print("Question:")
        print(row["question"])
        print("\nAnswer / reasoning:")
        print(row["answer"])
        print(f"\nExtracted final answer: {final_answer(row['answer'])!r}")


if __name__ == "__main__":
    main()
