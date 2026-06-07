# Finance RL — Qwen3-8B Fine-Tuning & Benchmarking

Fine-tune [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) on financial datasets using a two-stage SFT → GRPO pipeline, then benchmark the results. Inspired by [Fin-R1](https://arxiv.org/abs/2503.16252).

---

## Pipeline Overview

```
Baseline benchmark (zero-shot)
        ↓
SFT — Finance Alpaca (~68k) + FinQA (~6k)
        ↓
Benchmark SFT checkpoint
        ↓
GRPO — FinQA with dual reward (format + accuracy)
        ↓
Benchmark GRPO checkpoint
        ↓
Compare all three results
```

---

## Datasets

| Dataset | Role | License |
|---|---|---|
| [gbharti/finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca) | SFT training | MIT |
| [dreamerdeo/finqa](https://huggingface.co/datasets/dreamerdeo/finqa) | SFT + GRPO training | MIT |
| [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | Optional benchmark | MIT |
| [BeIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa) | Optional benchmark | CC-BY-SA-4.0 |

---

## Scripts

| File | Description |
|---|---|
| `benchmark.py` | Evaluate baseline or LoRA checkpoint — ROUGE-1/2/L, BERTScore P/R/F1, response length, throughput |
| `train_sft.py` | SFT with Unsloth LoRA on Finance Alpaca + FinQA combined |
| `train_grpo.py` | GRPO with dual reward (format 0.2 + accuracy 0.8) on FinQA |
| `generate_dataset.py` | Synthetic finance Q&A dataset generator (used for early experiments) |
| `RUNBOOK.md` | Step-by-step setup and usage instructions |

---

## Benchmark Metrics

Each run produces a JSON and a Markdown report in `results/`:

| Metric | What it measures |
|---|---|
| ROUGE-1 F1 | Unigram word overlap |
| ROUGE-2 F1 | Bigram / phrase overlap |
| ROUGE-L F1 | Longest common subsequence |
| BERTScore P/R/F1 | Semantic similarity via BERT embeddings |
| Avg response length | Words in prediction vs reference |
| Throughput | Samples/sec, total evaluation time |

---

## GRPO Dual Reward

Replaces single ROUGE-L with two coupled signals:

| Component | Weight | Condition |
|---|---|---|
| Format reward | 0.2 | Completion contains `#### <number>` marker |
| Accuracy reward | 0.8 | Extracted number within 1% of reference |

Accuracy credit is only awarded when the format marker is present — preventing the model from hiding correct numbers in unstructured prose.

---

## Requirements

- Nvidia GPU ≥ 24 GB VRAM (recommended: RTX PRO 4500 at $0.74/hr)
- Python 3.11+, CUDA 12.1 or 12.4

---

## Quick Start

```bash
# Clone and enter the project
git clone https://github.com/ChaitanyaPandit1998/finance-rl.git && cd finance-rl

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Configure environment variables
cp .env.example .env   # then fill in HF_TOKEN, HF_HOME, CHECKPOINT_DIR

# Install dependencies (see RUNBOOK.md for full GPU setup)
pip install python-dotenv                                              # install first so scripts can load .env
pip install unsloth bitsandbytes                                       # unsloth installs torch automatically
pip install -r requirements.txt

# 1. Baseline benchmark
python benchmark.py --tag baseline

# 2. SFT (Finance Alpaca + FinQA)
python train_sft.py

# 3. Benchmark SFT
python benchmark.py --checkpoint checkpoints/sft --tag sft

# 4. GRPO (FinQA, dual reward)
python train_grpo.py --base-checkpoint checkpoints/sft

# 5. Benchmark GRPO
python benchmark.py --checkpoint checkpoints/grpo --tag grpo

# 6. Compare all results
python3 -c "
import json, glob
for path in sorted(glob.glob('results/benchmark_*.json')):
    d = json.load(open(path))['summary']
    print(f\"{d['tag']:<10} ROUGE-L: {d['rougeL']['f1']['mean']:.4f}  BERTScore: {d.get('bert_score', {}).get('f1', {}).get('mean', 'n/a')}\")
"
```

See [RUNBOOK.md](RUNBOOK.md) for the full setup guide including multi-GPU usage, QLoRA fallback, and smoke test commands.

---

## Related Work

- [Fin-R1](https://arxiv.org/abs/2503.16252) — SFT + GRPO on Qwen2.5-7B, 76.0 on FinQA
- [FEVO](https://arxiv.org/abs/2507.06057) — CPT + SFT + RL on Qwen2.5-32B, beats GPT-4o on finance benchmarks
