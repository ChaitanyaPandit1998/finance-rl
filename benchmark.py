#!/usr/bin/env python3
"""
Benchmark Qwen3-8B (baseline or LoRA checkpoint) on gbharti/finance-alpaca.

Metrics computed per run:
  - ROUGE-1, ROUGE-2, ROUGE-L  (precision, recall, F1 for each)
  - BERTScore                   (precision, recall, F1)
  - Response length             (avg words in prediction vs reference)
  - Throughput                  (samples/sec, total time)

Two output files are written per run:
  results/benchmark_{tag}.json   full stats + per-sample records
  results/benchmark_{tag}.md     human-readable summary for GitHub

Usage:
    # Baseline (no fine-tuning)
    python benchmark.py --tag baseline

    # After SFT
    python benchmark.py --checkpoint checkpoints/sft --tag sft

    # After GRPO
    python benchmark.py --checkpoint checkpoints/grpo --tag grpo

    # Skip BERTScore (faster, saves ~2 GB VRAM)
    python benchmark.py --tag baseline --no-bertscore

    # Budget GPU
    python benchmark.py --tag baseline --use-qlora
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import torch
    from datasets import load_dataset
    from rouge_score import rouge_scorer as rs
except ImportError as e:
    if "--help" not in sys.argv and "-h" not in sys.argv:
        raise

BASE_MODEL = "Qwen/Qwen3-8B"
MAX_NEW_TOKENS = 512
SYSTEM_PROMPT = "You are a helpful financial assistant. Answer concisely and accurately."


def format_prompt(instruction: str, input_text: str, tokenizer) -> str:
    """Build a chat-formatted prompt string ready for tokenization.

    Combines the instruction and optional input into a single user message,
    applies Qwen3's chat template, and appends the generation prompt marker
    so the model knows to start its response. The /no_think prefix suppresses
    Qwen3's internal chain-of-thought reasoning, which speeds up inference
    without affecting answer quality for straightforward Q&A.

    Args:
        instruction: The finance question or task description.
        input_text: Optional additional context (empty string if not provided).
        tokenizer: Qwen3 tokenizer used to apply the chat template.

    Returns:
        Fully formatted prompt string including special tokens.
    """
    user_content = instruction
    if input_text.strip():
        user_content += f"\n\n{input_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "/no_think\n" + user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(checkpoint: str | None, use_qlora: bool):
    """Load Qwen3-8B via Unsloth and optionally apply a LoRA adapter.

    Loads the base model in bfloat16 by default, or in 4-bit if use_qlora
    is set. If a checkpoint path is provided, the saved LoRA adapter weights
    are loaded on top of the base model. The model is then switched to
    inference mode (disables dropout, fuses layers for speed).

    Args:
        checkpoint: Path to a saved LoRA adapter directory, or None for
            zero-shot baseline evaluation.
        use_qlora: If True, load in 4-bit quantization (for GPUs < 24GB VRAM).

    Returns:
        Tuple of (model, tokenizer) ready for inference.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        dtype=None if use_qlora else torch.bfloat16,
        load_in_4bit=use_qlora,
    )
    if checkpoint:
        print(f"Loading adapter from {checkpoint}...")
        model.load_adapter(checkpoint)

    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: list[str], device) -> list[str]:
    """Run greedy decoding on a batch of prompts and return the new tokens only.

    Pads the batch to a uniform length, runs a single forward + decode pass,
    then strips the prompt tokens from the output so only the model's response
    is returned. Uses greedy decoding (do_sample=False) for deterministic,
    reproducible benchmark results.

    Args:
        model: The loaded Qwen3 model in inference mode.
        tokenizer: Corresponding tokenizer with left-side padding.
        prompts: List of fully formatted prompt strings.
        device: torch.device the model is on.

    Returns:
        List of decoded response strings, one per prompt.
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, list[dict]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L for each prediction/reference pair.

    Returns all three variants together in one pass so the scorer is only
    instantiated once. Each variant includes precision, recall, and F1 so
    callers can report whichever combination is most informative.

    ROUGE-1: unigram overlap — broad vocabulary match
    ROUGE-2: bigram overlap — captures phrasing similarity
    ROUGE-L: longest common subsequence — captures sentence-level structure

    Args:
        predictions: Model-generated responses.
        references: Ground truth answers from the dataset.

    Returns:
        Dict with keys 'rouge1', 'rouge2', 'rougeL'. Each value is a list of
        dicts with keys 'precision', 'recall', 'f1', one per prediction/reference pair.
    """
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in results:
            s = scores[key]
            results[key].append({
                "precision": round(s.precision, 4),
                "recall":    round(s.recall, 4),
                "f1":        round(s.fmeasure, 4),
            })
    return results


def compute_bertscore(predictions: list[str], references: list[str]) -> dict[str, list[float]]:
    """Compute BERTScore precision, recall, and F1 for each prediction/reference pair.

    BERTScore embeds both texts with a pretrained BERT model and measures
    cosine similarity between token embeddings, making it sensitive to
    semantic meaning rather than surface word overlap. This catches cases
    where the model uses different but correct phrasing that ROUGE would
    penalise. All three components are returned so the caller can assess
    whether the model is precise, comprehensive, or both.

    Args:
        predictions: Model-generated responses.
        references: Ground truth answers from the dataset.

    Returns:
        Dict with keys 'precision', 'recall', 'f1', each a list of floats in [0, 1].
    """
    from bert_score import score as bs_score
    P, R, F1 = bs_score(predictions, references, lang="en", verbose=False)
    return {
        "precision": [round(v, 4) for v in P.tolist()],
        "recall":    [round(v, 4) for v in R.tolist()],
        "f1":        [round(v, 4) for v in F1.tolist()],
    }


def response_length_stats(predictions: list[str], references: list[str]) -> dict:
    """Compute average word counts for predictions and references.

    A large gap between prediction and reference length indicates the model
    is either truncating answers (prediction << reference) or being verbose
    (prediction >> reference). Both are useful diagnostics alongside ROUGE scores.

    Args:
        predictions: Model-generated responses.
        references: Ground truth answers from the dataset.

    Returns:
        Dict with avg/min/max word counts for both predictions and references.
    """
    pred_lens = [len(p.split()) for p in predictions]
    ref_lens  = [len(r.split()) for r in references]
    return {
        "prediction_words": {
            "mean": round(sum(pred_lens) / len(pred_lens), 1),
            "min":  min(pred_lens),
            "max":  max(pred_lens),
        },
        "reference_words": {
            "mean": round(sum(ref_lens) / len(ref_lens), 1),
            "min":  min(ref_lens),
            "max":  max(ref_lens),
        },
    }


def percentile(data: list[float], p: int) -> float:
    """Return the p-th percentile of a list of floats, rounded to 4 decimal places."""
    data = sorted(data)
    idx = int(len(data) * p / 100)
    return round(data[min(idx, len(data) - 1)], 4)


def aggregate(scores: list[float]) -> dict:
    """Return mean, p25, p50, p75 for a list of scores."""
    return {
        "mean": round(sum(scores) / len(scores), 4),
        "p25":  percentile(scores, 25),
        "p50":  percentile(scores, 50),
        "p75":  percentile(scores, 75),
    }


def write_markdown_report(summary: dict, out_path: str) -> None:
    """Write a human-readable markdown summary for GitHub display.

    Produces a clean table-based report that renders directly on GitHub,
    making it easy to compare baseline / SFT / GRPO runs at a glance
    without opening the raw JSON.

    Args:
        summary: The summary dict built in main().
        out_path: File path to write the .md report to.
    """
    tag       = summary["tag"]
    checkpoint = summary.get("checkpoint") or "none (zero-shot baseline)"
    n         = summary["num_samples"]
    ts        = summary.get("timestamp", "")
    elapsed   = summary.get("elapsed_seconds", 0)
    sps       = summary.get("samples_per_second", 0)

    lines = [
        f"# Benchmark: `{tag}`",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Model | {BASE_MODEL} |",
        f"| Checkpoint | `{checkpoint}` |",
        f"| Samples evaluated | {n} |",
        f"| Run timestamp | {ts} |",
        f"| Total time | {elapsed:.0f}s ({sps:.1f} samples/s) |",
        "",
        "## ROUGE Scores",
        "",
        "| Metric | Precision | Recall | F1 (mean) | F1 (p50) |",
        "|---|---|---|---|---|",
    ]

    for key, label in [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")]:
        if key in summary:
            s = summary[key]
            lines.append(
                f"| {label} | {s['precision']['mean']:.4f} | {s['recall']['mean']:.4f} "
                f"| {s['f1']['mean']:.4f} | {s['f1']['p50']:.4f} |"
            )

    if "bert_score" in summary:
        bs = summary["bert_score"]
        lines += [
            "",
            "## BERTScore",
            "",
            "| Metric | Precision | Recall | F1 (mean) | F1 (p50) |",
            "|---|---|---|---|---|",
            f"| BERTScore | {bs['precision']['mean']:.4f} | {bs['recall']['mean']:.4f} "
            f"| {bs['f1']['mean']:.4f} | {bs['f1']['p50']:.4f} |",
        ]

    if "response_length" in summary:
        rl = summary["response_length"]
        lines += [
            "",
            "## Response Length (words)",
            "",
            "| | Mean | Min | Max |",
            "|---|---|---|---|",
            f"| Prediction | {rl['prediction_words']['mean']} "
            f"| {rl['prediction_words']['min']} | {rl['prediction_words']['max']} |",
            f"| Reference  | {rl['reference_words']['mean']} "
            f"| {rl['reference_words']['min']} | {rl['reference_words']['max']} |",
        ]

    lines += ["", "---", f"_Generated by benchmark.py — {ts}_", ""]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tag", type=str, default="baseline")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path("results").mkdir(exist_ok=True)

    print("Loading finance-alpaca...")
    dataset = load_dataset("gbharti/finance-alpaca", split="train")
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    samples = dataset.select(indices)
    print(f"Sampled {len(samples)} examples from {len(dataset)} total")

    print(f"Loading {BASE_MODEL}...")
    model, tokenizer = load_model(args.checkpoint, args.use_qlora)
    device = next(model.parameters()).device

    all_predictions, all_references, all_instructions = [], [], []
    t0 = time.time()

    for i in range(0, len(samples), args.batch_size):
        batch = samples[i : i + args.batch_size]
        instructions = batch["instruction"]
        inputs = batch.get("input") or [""] * len(instructions)
        refs = batch["output"]

        prompts = [format_prompt(instr, inp, tokenizer) for instr, inp in zip(instructions, inputs)]
        preds = generate_batch(model, tokenizer, prompts, device)

        all_predictions.extend(preds)
        all_references.extend(refs)
        all_instructions.extend(instructions)

        done = min(i + args.batch_size, len(samples))
        elapsed = time.time() - t0
        print(f"  [{done}/{len(samples)}] {done/elapsed:.1f} samples/s")

    total_elapsed = time.time() - t0

    print("\nScoring ROUGE-1 / ROUGE-2 / ROUGE-L...")
    rouge = compute_rouge(all_predictions, all_references)

    bert = None
    if not args.no_bertscore:
        print("Scoring BERTScore precision / recall / F1 (this may take a moment)...")
        bert = compute_bertscore(all_predictions, all_references)

    length_stats = response_length_stats(all_predictions, all_references)

    # --- Build summary ---
    summary = {
        "tag":                args.tag,
        "checkpoint":         args.checkpoint,
        "model":              BASE_MODEL,
        "num_samples":        len(samples),
        "timestamp":          datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "elapsed_seconds":    round(total_elapsed, 1),
        "samples_per_second": round(len(samples) / total_elapsed, 2),
        "rouge1": {
            "precision": aggregate([s["precision"] for s in rouge["rouge1"]]),
            "recall":    aggregate([s["recall"]    for s in rouge["rouge1"]]),
            "f1":        aggregate([s["f1"]        for s in rouge["rouge1"]]),
        },
        "rouge2": {
            "precision": aggregate([s["precision"] for s in rouge["rouge2"]]),
            "recall":    aggregate([s["recall"]    for s in rouge["rouge2"]]),
            "f1":        aggregate([s["f1"]        for s in rouge["rouge2"]]),
        },
        "rougeL": {
            "precision": aggregate([s["precision"] for s in rouge["rougeL"]]),
            "recall":    aggregate([s["recall"]    for s in rouge["rougeL"]]),
            "f1":        aggregate([s["f1"]        for s in rouge["rougeL"]]),
        },
        "response_length": length_stats,
    }

    if bert:
        summary["bert_score"] = {
            "precision": aggregate(bert["precision"]),
            "recall":    aggregate(bert["recall"]),
            "f1":        aggregate(bert["f1"]),
        }

    # --- Build per-sample records ---
    records = []
    for i, (instr, ref, pred) in enumerate(zip(all_instructions, all_references, all_predictions)):
        record = {
            "instruction": instr,
            "reference":   ref,
            "prediction":  pred,
            "rouge1_f1":   rouge["rouge1"][i]["f1"],
            "rouge2_f1":   rouge["rouge2"][i]["f1"],
            "rougeL_f1":   rouge["rougeL"][i]["f1"],
            "pred_words":  len(pred.split()),
            "ref_words":   len(ref.split()),
        }
        if bert:
            record["bert_precision"] = bert["precision"][i]
            record["bert_recall"]    = bert["recall"][i]
            record["bert_f1"]        = bert["f1"][i]
        records.append(record)

    # --- Write JSON ---
    json_path = f"results/benchmark_{args.tag}.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    # --- Write Markdown ---
    md_path = f"results/benchmark_{args.tag}.md"
    write_markdown_report(summary, md_path)

    # --- Print to terminal ---
    print(f"\n{'='*55}")
    print(f"Tag:        {args.tag}")
    print(f"Samples:    {len(samples)} in {total_elapsed:.0f}s ({len(samples)/total_elapsed:.1f}/s)")
    print(f"ROUGE-1 F1: mean={summary['rouge1']['f1']['mean']}  p50={summary['rouge1']['f1']['p50']}")
    print(f"ROUGE-2 F1: mean={summary['rouge2']['f1']['mean']}  p50={summary['rouge2']['f1']['p50']}")
    print(f"ROUGE-L F1: mean={summary['rougeL']['f1']['mean']}  p50={summary['rougeL']['f1']['p50']}")
    if bert:
        print(f"BERTScore:  P={summary['bert_score']['precision']['mean']}  "
              f"R={summary['bert_score']['recall']['mean']}  "
              f"F1={summary['bert_score']['f1']['mean']}")
    print(f"Avg words:  pred={length_stats['prediction_words']['mean']}  "
          f"ref={length_stats['reference_words']['mean']}")
    print(f"JSON:       {json_path}")
    print(f"Markdown:   {md_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
