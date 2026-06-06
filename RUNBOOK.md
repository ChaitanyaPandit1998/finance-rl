# Qwen3-8B Finance Fine-Tuning & Benchmarking

Fine-tune Qwen3-8B on Finance Alpaca + FinQA, evaluate with ROUGE-L and BERTScore.
Inspired by [Fin-R1](https://arxiv.org/abs/2503.16252) — same two-stage SFT → GRPO pipeline.

---

## Datasets

| Dataset | Used in | License |
|---|---|---|
| [gbharti/finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca) | SFT + GRPO (fallback) | MIT |
| [ibm/finqa](https://huggingface.co/datasets/ibm/finqa) | SFT + GRPO (default) | MIT |
| [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | Benchmark (optional) | MIT |
| [BeIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa) | Benchmark (optional) | CC-BY-SA-4.0 |

> FinanceBench (`PatronusAI/financebench`) is **non-commercial only** (CC-BY-NC-4.0). Excluded from this pipeline.

---

## Requirements

- Nvidia GPU with **≥ 24 GB VRAM** — recommended: **RTX PRO 4500** ($0.74/hr, high availability)
  - Use `--use-qlora` on all scripts for GPUs with 16 GB (e.g. T4, A10)
- Python 3.11+
- CUDA 12.1 or 12.4

---

## Setup

Run these once on your cloud GPU instance.

```bash
# 1. Clone the project
git clone https://github.com/ChaitanyaPandit1998/finance-rl.git && cd finance-rl

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Configure environment variables
cp .env.example .env
# Edit .env — fill in HF_TOKEN, and confirm HF_HOME / CHECKPOINT_DIR paths

# 4. PyTorch with CUDA (match your driver — check with: nvcc --version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. Unsloth (efficient LoRA training)
pip install unsloth

# 6. bitsandbytes (only needed for --use-qlora)
pip install bitsandbytes

# 7. Project dependencies
pip install -e .
```

> **Tip:** On subsequent sessions, just run `source .venv/bin/activate` from the project root before running any script.

Verify the environment:

```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
python3 benchmark.py --help
```

---

## Workflow Overview

```
Phase 1: Baseline benchmark (zero-shot)
          ↓
Phase 2: SFT — Finance Alpaca (~68k) + FinQA (~6k)
          ↓
Phase 3: Benchmark SFT checkpoint
          ↓
Phase 4: GRPO — FinQA with dual reward (format + accuracy)
          ↓
Phase 5: Benchmark GRPO checkpoint
          ↓
Phase 6: Compare all three results
```

Results land in `results/` as JSON files.

---

## Phase 1 — Baseline Benchmark

Evaluate Qwen3-8B zero-shot on 500 Finance Alpaca samples. Establishes the pre-training baseline.

```bash
python benchmark.py --tag baseline
```

| Flag | Default | Description |
|---|---|---|
| `--tag` | `baseline` | Label for the output file (`results/benchmark_{tag}.json`) |
| `--num-samples` | `500` | Number of examples to evaluate |
| `--batch-size` | `4` | Inference batch size |
| `--checkpoint` | none | Path to a LoRA adapter (omit for zero-shot baseline) |
| `--no-bertscore` | off | Skip BERTScore to save ~2 GB VRAM and time |
| `--use-qlora` | off | Load model in 4-bit (for GPUs < 24 GB) |
| `--seed` | `42` | Random seed for sample selection |

---

## Phase 2 — SFT Training

Trains Qwen3-8B with LoRA on **Finance Alpaca + FinQA combined**:

- **Finance Alpaca** (~68k): broad financial instruction-following and terminology
- **FinQA** (~6k): numerical reasoning over SEC filing tables — teaches the `#### <answer>` format that GRPO's dual reward targets

```bash
python train_sft.py
```

Adapter saved to `checkpoints/sft/`.

| Flag | Default | Description |
|---|---|---|
| `--finqa-samples` | `-1` | FinQA examples to mix in (`-1` = all ~6k, `0` = skip FinQA) |
| `--rank` | `16` | LoRA rank (`32` for higher quality, uses more VRAM) |
| `--lora-alpha` | `16` | LoRA alpha |
| `--epochs` | `1` | Training epochs (1 epoch ≈ 4–6 hrs on A100/RTX PRO 4500) |
| `--batch-size` | `2` | Per-device batch size |
| `--grad-accum` | `4` | Gradient accumulation (effective batch = batch × accum) |
| `--lr` | `2e-4` | Learning rate |
| `--max-seq-len` | `2048` | Max sequence length |
| `--max-steps` | off | Override epochs with a fixed step count |
| `--output-dir` | `checkpoints/sft` | Where to save the adapter |
| `--use-qlora` | off | 4-bit QLoRA for GPUs < 24 GB |

Quick smoke test (100 steps, ~5 min):

```bash
python train_sft.py --max-steps 100
```

Finance Alpaca only (skip FinQA):

```bash
python train_sft.py --finqa-samples 0
```

---

## Phase 3 — Benchmark SFT Checkpoint

```bash
python benchmark.py --checkpoint checkpoints/sft --tag sft
```

---

## Phase 4 — GRPO Training

Reinforcement learning with a **dual reward signal** — a significant improvement over single ROUGE-L:

| Reward component | Weight | What it checks |
|---|---|---|
| Format reward | 0.2 | Does the completion contain `#### <number>`? |
| Accuracy reward | 0.8 | Is the extracted number within 1% of the reference? |

A fluent but numerically wrong answer scores **0.2 max**. Both format and accuracy correct scores **1.0**.
Accuracy credit is only given when the `####` marker is present — preventing the model from hiding correct numbers in unstructured prose to game the reward.

Default dataset is **FinQA** (verifiable numerical answers). Use `--dataset alpaca` to fall back to Finance Alpaca with ROUGE-L reward.

**Start from the SFT checkpoint** — faster convergence, more stable than cold GRPO.

```bash
python train_grpo.py --base-checkpoint checkpoints/sft
```

Adapter saved to `checkpoints/grpo/`.

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `finqa` | `finqa` = dual reward, `alpaca` = ROUGE-L fallback |
| `--base-checkpoint` | none | LoRA adapter to start from (**strongly recommended**: use SFT checkpoint) |
| `--num-generations` | `4` | Completions per prompt — minimum 2, higher = more stable gradients + more VRAM |
| `--lr` | `5e-6` | Learning rate (kept small — policy updates should be incremental) |
| `--max-train-samples` | `-1` | Cap dataset size (`-1` = all examples) |
| `--max-prompt-len` | `1024` | Raised vs Alpaca default — FinQA prompts include table context |
| `--max-completion-len` | `512` | Max tokens generated per completion |
| `--rank` | `16` | LoRA rank (only applies when no `--base-checkpoint`) |
| `--epochs` | `1` | Training epochs |
| `--batch-size` | `2` | Per-device batch size |
| `--grad-accum` | `4` | Gradient accumulation steps |
| `--output-dir` | `checkpoints/grpo` | Where to save the adapter |
| `--use-qlora` | off | 4-bit QLoRA for GPUs < 24 GB |

Quick smoke test (50 steps, ~10 min):

```bash
python train_grpo.py --base-checkpoint checkpoints/sft --max-steps 50
```

GRPO on Finance Alpaca with ROUGE-L (weaker signal, for comparison):

```bash
python train_grpo.py --base-checkpoint checkpoints/sft --dataset alpaca
```

---

## Phase 5 — Benchmark GRPO Checkpoint

```bash
python benchmark.py --checkpoint checkpoints/grpo --tag grpo
```

---

## Phase 6 — Compare Results

All benchmark outputs land in `results/`. Run this to compare all three at once:

```bash
python3 -c "
import json, glob
for path in sorted(glob.glob('results/benchmark_*.json')):
    d = json.load(open(path))['summary']
    tag = d['tag'].ljust(10)
    rl  = d['rouge_l']['mean']
    bs  = d.get('bert_score_f1', {}).get('mean', 'n/a')
    print(f'{tag}  ROUGE-L: {rl:.4f}   BERTScore: {bs}')
"
```

Example output:

```
baseline    ROUGE-L: 0.1823   BERTScore: 0.8541
sft         ROUGE-L: 0.2614   BERTScore: 0.8893
grpo        ROUGE-L: 0.2891   BERTScore: 0.9012
```

---

## Multi-GPU (RTX PRO 4500 × 2 or × 4)

Multiple GPUs cut training time roughly in half per additional GPU at similar total cost.
For GRPO, more GPUs also let you increase `--num-generations` for a better reward signal.

```bash
# Configure accelerate once (select "multi-GPU" and number of processes)
accelerate config

# Then replace `python` with `accelerate launch --num_processes=N`
accelerate launch --num_processes=2 train_sft.py
accelerate launch --num_processes=2 train_grpo.py --base-checkpoint checkpoints/sft --num-generations 8
```

| GPUs | `--num-generations` | GRPO reward quality |
|---|---|---|
| 1 | 4 (default) | Baseline |
| 2 | 8 | More stable advantage estimates |
| 4 | 16 | Significantly better policy gradient |

---

## Budget GPU (< 24 GB VRAM)

Add `--use-qlora` to every command. 4-bit quantisation keeps VRAM under 10 GB.

```bash
python benchmark.py --tag baseline --use-qlora
python train_sft.py               --use-qlora
python benchmark.py --tag sft     --use-qlora --checkpoint checkpoints/sft
python train_grpo.py              --use-qlora --base-checkpoint checkpoints/sft
python benchmark.py --tag grpo    --use-qlora --checkpoint checkpoints/grpo
```

---

## Output Structure

```
checkpoints/
  sft/               # LoRA adapter + tokenizer (Finance Alpaca + FinQA SFT)
  grpo/              # LoRA adapter + tokenizer (FinQA GRPO, dual reward)

results/
  benchmark_baseline.json   # Zero-shot Qwen3-8B
  benchmark_sft.json        # After SFT
  benchmark_grpo.json       # After GRPO
```

Each JSON has a `summary` block at the top and a `records` array with per-example scores.
