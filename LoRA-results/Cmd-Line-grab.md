(.venv) PS C:\sac\dev\RiverAI> python .\RL-Train-Demo.py

healthy: True
engine: Qwen/Qwen3.6-35B-A3B-FP8
configuration: {'batch_size': 16, 'group_size': 4, 'max_tokens': 1024, 'max_steps': 5, 'eval_size': 30, 'lora_rank': 8, 'learning_rate': 4e-05, 'stop_token': '<|im_end|>', 'eval_seed_base': 20260921}
dataset separation: train=7473, held_out_test=1319
requesting River training session...
River training session ready.
model_id: 40c60338-a6bd-42a4-bed5-331c2583c257:model:1

Baseline evaluation (held-out test split; no weight updates):
Baseline: 29 / 30 = 0.967; missing_boxed=1
example_zero_advantage: step=1, train_index=0, reference='72', extracted=['72', '72', '72', '72'], rewards=[1.0, 1.0, 1.0, 1.0]
example_mixed: step=1, train_index=15, reference='448000', extracted=['448000', '448000', '448000', None], rewards=[1.0, 1.0, 1.0, 0.0]
step=1 questions=16 completions=64 correct_completions=63 mean_reward=0.984 mixed_groups=1 all_wrong_groups=0 all_correct_groups=15 training_records=4 optimizer_applied=true loss=89.47932434082031
step=2 questions=16 completions=64 correct_completions=64 mean_reward=1.000 mixed_groups=0 all_wrong_groups=0 all_correct_groups=16 training_records=0 optimizer_applied=false loss=n/a
step=3 questions=16 completions=64 correct_completions=61 mean_reward=0.953 mixed_groups=2 all_wrong_groups=0 all_correct_groups=14 training_records=8 optimizer_applied=true loss=82.20068359375
step=4 questions=16 completions=64 correct_completions=57 mean_reward=0.891 mixed_groups=2 all_wrong_groups=1 all_correct_groups=13 training_records=8 optimizer_applied=true loss=82.12725830078125
step=5 questions=16 completions=64 correct_completions=64 mean_reward=1.000 mixed_groups=0 all_wrong_groups=0 all_correct_groups=16 training_records=0 optimizer_applied=false loss=n/a
saved checkpoint: river://b0716687-6a1f-4627-a218-6ea0cb840cb4/sampler_weights/gsm8k-qwen-learning-run

Post-training evaluation (the same held-out test indices):
Post-training: 29 / 30 = 0.967; missing_boxed=0
River training session closed.
Delta: +0.000 (small-sample comparison; not a benchmark)
results written: C:\sac\dev\RiverAI\LoRA-results\results.json


(.venv) PS C:\sac\dev\RiverAI>