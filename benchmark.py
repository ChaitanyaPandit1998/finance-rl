#!/usr/bin/env python3
"""
Benchmark Qwen3-8B (baseline or LoRA checkpoint) on FinQA or Finance Alpaca.

Default dataset is FinQA test split — a proper held-out evaluation set the
model never sees during training. Finance Alpaca (--dataset alpaca) is available
for comparison but suffers from train/test contamination after SFT.

Metrics:
  FinQA (default):
    - Exact match accuracy  PRIMARY — extracted number within 1% of reference
    - ROUGE-1/2/L, BERTScore  secondary
  Finance Alpaca:
    - ROUGE-1/2/L, BERTScore  only (no verifiable ground truth)

Two output files per run:
  results/benchmark_{tag}.json   full stats + per-sample records
  results/benchmark_{tag}.md     human-readable summary for GitHub

Usage:
    # Proper FinQA benchmark (recommended)
    python benchmark.py --tag baseline
    python benchmark.py --checkpoint checkpoints/sft --tag sft
    python benchmark.py --checkpoint checkpoints/grpo --tag grpo

    # Finance Alpaca (ROUGE/BERTScore only, contaminated after SFT)
    python benchmark.py --dataset alpaca --tag baseline_alpaca

    # Skip BERTScore (faster)
    python benchmark.py --tag baseline --no-bertscore

    # Budget GPU
    python benchmark.py --tag baseline --use-qlora
"""
import argparse
import json
import os
import re
import random
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads HF_TOKEN and HF_HOME from .env before any HF imports
except ImportError:
    print("Warning: python-dotenv not installed. Set HF_TOKEN, HF_HOME, "
          "CHECKPOINT_DIR manually in your shell if needed.")

HF_CACHE: str | None = os.getenv("HF_HOME")

try:
    import torch
    from datasets import load_dataset
    from rouge_score import rouge_scorer as rs
except ImportError as e:
    if "--help" not in sys.argv and "-h" not in sys.argv:
        raise

BASE_MODEL = "Qwen/Qwen3-8B"
SYSTEM_PROMPT = "You are a helpful financial assistant. Answer concisely and accurately."


# ── Text utilities ─────────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from generated output.

    Qwen3's thinking tokens are plain text, not special tokens, so
    skip_special_tokens=True does not remove them. Leaving them in inflates
    response length and contaminates ROUGE/BERTScore with reasoning chains
    that have nothing to do with the final answer.

    Handles two cases:
    - Complete blocks: <think>...</think> — both tags present, strip contents
    - Truncated blocks: <think>... with no closing tag — max_new_tokens cut the
      generation mid-thought; strip everything from <think> to end of string

    Args:
        text: Raw decoded model output.

    Returns:
        Text with all thinking content removed and whitespace stripped.
    """
    # Complete blocks first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Truncated blocks — <think> opened but </think> never arrived
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def extract_final_answer(text: str) -> str | None:
    """Extract a numerical answer from model output.

    Looks for the '#### <number>' marker that SFT training establishes.
    Falls back to the last number in the text for outputs that don't use
    the structured format.

    Args:
        text: Model completion with thinking stripped.

    Returns:
        Numeric string with commas and % stripped, or None if not found.
    """
    match = re.search(r"####\s*(-?[\d,]+\.?\d*%?)", text)
    if match:
        return match.group(1).replace(",", "").replace("%", "")
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else None


# ── Prompt formatting ──────────────────────────────────────────────────────────

def format_alpaca_prompt(instruction: str, input_text: str, tokenizer) -> str:
    """Build a chat-formatted prompt for a Finance Alpaca example.

    Injects /no_think to suppress Qwen3's chain-of-thought for faster
    inference. Thinking is stripped from the output regardless, but suppressing
    it here saves generation time.

    Args:
        instruction: The finance question or task.
        input_text: Optional additional context (empty string if absent).
        tokenizer: Qwen3 tokenizer.

    Returns:
        Fully formatted prompt string with generation marker appended.
    """
    user_content = instruction
    if input_text.strip():
        user_content += f"\n\n{input_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "/no_think\n" + user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_finqa_prompt(example: dict, tokenizer) -> str:
    """Build a chat-formatted prompt for a FinQA example.

    Includes the financial context (table + surrounding text from the SEC
    filing) so the model has the source data needed to compute the answer.
    The model must produce a '#### <number>' answer to receive full credit
    from the exact-match scorer.

    Args:
        example: FinQA dataset row with pre_text, table, post_text, question.
        tokenizer: Qwen3 tokenizer.

    Returns:
        Fully formatted prompt string with generation marker appended.
    """
    pre = " ".join(example.get("pre_text") or [])
    post = " ".join(example.get("post_text") or [])
    table = example.get("table") or []
    if isinstance(table, list) and table:
        table_str = "\n".join(" | ".join(str(cell) for cell in row) for row in table)
    else:
        table_str = str(table) if table else ""

    context = "\n\n".join(part for part in [pre, table_str, post] if part.strip())
    user_content = (
        f"/no_think\n"
        f"Financial Context:\n{context}\n\n"
        f"Question: {example['question']}\n\n"
        f"Provide step-by-step reasoning and end with '#### <number>'."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Model loading & inference ──────────────────────────────────────────────────

def load_model(checkpoint: str | None, use_qlora: bool):
    """Load Qwen3-8B via Unsloth and optionally apply a LoRA adapter.

    Args:
        checkpoint: Path to a LoRA adapter directory, or None for zero-shot.
        use_qlora: Load in 4-bit quantization (for GPUs < 24 GB VRAM).

    Returns:
        Tuple of (model, tokenizer) ready for inference.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        dtype=None if use_qlora else torch.bfloat16,
        load_in_4bit=use_qlora,
        cache_dir=HF_CACHE,
    )
    if checkpoint:
        print(f"Loading adapter from {checkpoint}...")
        model.load_adapter(checkpoint)

    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: list[str], device,
                   max_prompt_len: int = 1024,
                   max_new_tokens: int = 1024) -> list[str]:
    """Run greedy decoding on a batch and return clean responses.

    Strips prompt tokens and Qwen3 thinking blocks from outputs so only
    the final answer text is returned for scoring.

    Args:
        model: Qwen3 model in inference mode.
        tokenizer: Corresponding tokenizer with left-side padding.
        prompts: Fully formatted prompt strings.
        device: Torch device the model is on.
        max_prompt_len: Truncation limit for input tokens. FinQA prompts
            include table context and need more room than Alpaca prompts.

    Returns:
        List of clean response strings, one per prompt.
    """
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=None,  # clear Qwen3's baked-in default (40960) to silence warning
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [strip_thinking(t) for t in decoded]


# ── Scoring ────────────────────────────────────────────────────────────────────

def compute_exact_match(predictions: list[str], references: list[str]) -> list[bool]:
    """Check whether each prediction's extracted number matches the reference.

    Uses 1% relative tolerance so minor floating-point differences don't
    count as wrong. Percentage signs are stripped before comparison since
    some FinQA answers are expressed as '14.2%' while others are plain '14.2'.
    This is the primary metric for FinQA — unambiguous and contamination-free.

    Args:
        predictions: Model-generated responses (thinking already stripped).
        references: Ground truth answer strings from the FinQA test split.

    Returns:
        List of booleans, True if the extracted answer is within 1% of reference.
    """
    results = []
    for pred, ref in zip(predictions, references):
        pred_str = extract_final_answer(pred)
        ref_str = str(ref).replace(",", "").replace("%", "").strip()
        if pred_str is None:
            results.append(False)
            continue
        try:
            p, r = float(pred_str), float(ref_str)
            if r == 0.0:
                results.append(abs(p) < 1e-6)
            else:
                results.append(abs(p - r) / abs(r) < 0.01)
        except ValueError:
            results.append(pred_str.strip() == ref_str)
    return results


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, list[dict]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L (precision, recall, F1) per pair.

    All three variants are computed in one scorer pass for efficiency.
    ROUGE-1: unigram overlap; ROUGE-2: bigram/phrase overlap;
    ROUGE-L: longest common subsequence / sentence structure.

    Args:
        predictions: Model-generated responses.
        references: Ground truth answers.

    Returns:
        Dict with keys 'rouge1', 'rouge2', 'rougeL', each a list of
        {'precision', 'recall', 'f1'} dicts.
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
    """Compute BERTScore precision, recall, and F1 per pair.

    BERTScore measures semantic similarity via BERT token embeddings rather
    than surface word overlap, catching cases where correct but differently-
    worded answers would be penalised by ROUGE.

    Args:
        predictions: Model-generated responses.
        references: Ground truth answers.

    Returns:
        Dict with keys 'precision', 'recall', 'f1', each a list of floats.
    """
    from bert_score import score as bs_score
    P, R, F1 = bs_score(predictions, references, lang="en", verbose=False)
    return {
        "precision": [round(v, 4) for v in P.tolist()],
        "recall":    [round(v, 4) for v in R.tolist()],
        "f1":        [round(v, 4) for v in F1.tolist()],
    }


def response_length_stats(predictions: list[str], references: list[str]) -> dict:
    """Compute avg/min/max word counts for predictions vs references.

    A large gap flags truncated answers (pred << ref) or verbosity (pred >> ref).

    Args:
        predictions: Model-generated responses.
        references: Ground truth answers.

    Returns:
        Dict with word count stats for predictions and references.
    """
    pred_lens = [len(p.split()) for p in predictions]
    ref_lens  = [len(r.split()) for r in references]
    return {
        "prediction_words": {"mean": round(sum(pred_lens)/len(pred_lens), 1),
                              "min": min(pred_lens), "max": max(pred_lens)},
        "reference_words":  {"mean": round(sum(ref_lens)/len(ref_lens), 1),
                              "min": min(ref_lens),  "max": max(ref_lens)},
    }


# ── Aggregation helpers ────────────────────────────────────────────────────────

def percentile(data: list[float], p: int) -> float:
    """Return the p-th percentile of a list, rounded to 4 decimal places."""
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


# ── Reporting ──────────────────────────────────────────────────────────────────

def write_markdown_report(summary: dict, out_path: str) -> None:
    """Write a GitHub-renderable markdown summary of the benchmark run.

    For FinQA, exact match accuracy is shown at the top as the headline
    metric. ROUGE and BERTScore follow as secondary context.

    Args:
        summary: The summary dict built in main().
        out_path: Path to write the .md file to.
    """
    tag        = summary["tag"]
    checkpoint = summary.get("checkpoint") or "none (zero-shot baseline)"
    dataset    = summary.get("dataset", "finqa")
    n          = summary["num_samples"]
    ts         = summary.get("timestamp", "")
    elapsed    = summary.get("elapsed_seconds", 0)
    sps        = summary.get("samples_per_second", 0)

    lines = [
        f"# Benchmark: `{tag}`",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Model | {BASE_MODEL} |",
        f"| Dataset | {dataset} |",
        f"| Checkpoint | `{checkpoint}` |",
        f"| Samples evaluated | {n} |",
        f"| Run timestamp | {ts} |",
        f"| Total time | {elapsed:.0f}s ({sps:.1f} samples/s) |",
        "",
    ]

    if "exact_match" in summary:
        em = summary["exact_match"]
        lines += [
            "## Exact Match Accuracy (Primary — FinQA)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Accuracy | **{em['accuracy']:.1%}** |",
            f"| Correct | {em['correct']} / {em['total']} |",
            "",
        ]

    lines += [
        "## ROUGE Scores (Secondary)",
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
            "## BERTScore (Secondary)",
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["finqa", "alpaca"], default="finqa",
                        help="finqa = FinQA test split (proper held-out eval, recommended); "
                             "alpaca = Finance Alpaca train sample (contaminated after SFT)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tag", type=str, default="baseline")
    parser.add_argument("--num-samples", type=int, default=-1,
                        help="Number of examples to evaluate (-1 = full test split)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max tokens the model can generate per example. "
                             "FinQA needs ~800-1500 tokens for reasoning + answer; "
                             "lower values risk cutting off before '#### <number>'.")
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-every", type=int, default=50,
                        help="Print a sample Q/answer every N samples (0 = disable)")
    args = parser.parse_args()

    random.seed(args.seed)
    Path("results").mkdir(exist_ok=True)

    # --- Load dataset ---
    if args.dataset == "finqa":
        print("Loading FinQA test split (held-out, no contamination)...")
        from utils import load_finqa
        raw = load_finqa(split="test", cache_dir=HF_CACHE)
        max_prompt_len = 2048  # FinQA prompts include table context
    else:
        print("Loading Finance Alpaca (warning: contaminated after SFT)...")
        raw = load_dataset("gbharti/finance-alpaca", split="train", cache_dir=HF_CACHE)
        max_prompt_len = 1024

    if args.num_samples > 0 and args.num_samples < len(raw):
        indices = random.sample(range(len(raw)), args.num_samples)
        samples = raw.select(indices)
    else:
        samples = raw
    print(f"Evaluating on {len(samples)} examples")

    # --- Load model ---
    print(f"Loading {BASE_MODEL}...")
    model, tokenizer = load_model(args.checkpoint, args.use_qlora)
    device = next(model.parameters()).device

    all_predictions, all_references, all_questions = [], [], []
    t0 = time.time()

    for i in range(0, len(samples), args.batch_size):
        batch = samples[i : i + args.batch_size]

        if args.dataset == "finqa":
            # Dataset slice returns a dict-of-lists, not a list-of-dicts
            questions = batch["question"]
            refs      = [str(a) for a in batch["answer"]]
            n         = len(questions)
            examples  = [{k: batch[k][j] for k in batch} for j in range(n)]
            prompts   = [format_finqa_prompt(ex, tokenizer) for ex in examples]
        else:
            questions = batch["instruction"]
            refs      = batch["output"]
            inputs_   = batch.get("input") or [""] * len(questions)
            prompts   = [format_alpaca_prompt(q, inp, tokenizer)
                         for q, inp in zip(questions, inputs_)]

        preds = generate_batch(model, tokenizer, prompts, device,
                               max_prompt_len, args.max_new_tokens)
        all_predictions.extend(preds)
        all_references.extend(refs)
        all_questions.extend(questions)

        done    = min(i + args.batch_size, len(samples))
        elapsed = time.time() - t0
        print(f"  [{done}/{len(samples)}] {done/elapsed:.1f} samples/s")

        if args.preview_every > 0 and done % args.preview_every == 0:
            idx = len(preds) - 1
            print(f"\n  {'─'*60}")
            print(f"  Sample [{done}/{len(samples)}]")
            print(f"  Q:   {all_questions[-1][:120]}{'...' if len(all_questions[-1]) > 120 else ''}")
            print(f"  Ref: {refs[idx][:200]}{'...' if len(refs[idx]) > 200 else ''}")
            print(f"  Ans: {preds[idx][:200]}{'...' if len(preds[idx]) > 200 else ''}")
            print(f"  {'─'*60}\n")

    total_elapsed = time.time() - t0

    # --- Score ---
    print("\nScoring ROUGE-1 / ROUGE-2 / ROUGE-L...")
    rouge = compute_rouge(all_predictions, all_references)

    exact_matches = None
    if args.dataset == "finqa":
        print("Scoring exact match accuracy...")
        exact_matches = compute_exact_match(all_predictions, all_references)

    bert = None
    if not args.no_bertscore:
        print("Scoring BERTScore (this may take a moment)...")
        bert = compute_bertscore(all_predictions, all_references)

    length_stats = response_length_stats(all_predictions, all_references)

    # --- Build summary ---
    summary = {
        "tag":                args.tag,
        "dataset":            args.dataset,
        "checkpoint":         args.checkpoint,
        "model":              BASE_MODEL,
        "num_samples":        len(samples),
        "max_new_tokens":     args.max_new_tokens,
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

    if exact_matches is not None:
        correct = sum(exact_matches)
        summary["exact_match"] = {
            "accuracy": round(correct / len(exact_matches), 4),
            "correct":  correct,
            "total":    len(exact_matches),
        }

    if bert:
        summary["bert_score"] = {
            "precision": aggregate(bert["precision"]),
            "recall":    aggregate(bert["recall"]),
            "f1":        aggregate(bert["f1"]),
        }

    # --- Per-sample records ---
    records = []
    for i, (q, ref, pred) in enumerate(zip(all_questions, all_references, all_predictions)):
        record = {
            "question":  q,
            "reference": ref,
            "prediction": pred,
            "rouge1_f1": rouge["rouge1"][i]["f1"],
            "rouge2_f1": rouge["rouge2"][i]["f1"],
            "rougeL_f1": rouge["rougeL"][i]["f1"],
            "pred_words": len(pred.split()),
            "ref_words":  len(ref.split()),
        }
        if exact_matches is not None:
            record["exact_match"] = exact_matches[i]
            record["extracted_answer"] = extract_final_answer(pred)
        if bert:
            record["bert_precision"] = bert["precision"][i]
            record["bert_recall"]    = bert["recall"][i]
            record["bert_f1"]        = bert["f1"][i]
        records.append(record)

    # --- Write outputs ---
    json_path = f"results/benchmark_{args.tag}.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    md_path = f"results/benchmark_{args.tag}.md"
    write_markdown_report(summary, md_path)

    # --- Terminal summary ---
    print(f"\n{'='*55}")
    print(f"Tag:        {args.tag}  |  Dataset: {args.dataset}")
    print(f"Samples:    {len(samples)} in {total_elapsed:.0f}s ({len(samples)/total_elapsed:.1f}/s)")
    if exact_matches is not None:
        print(f"Exact Match:{summary['exact_match']['correct']}/{summary['exact_match']['total']} "
              f"= {summary['exact_match']['accuracy']:.1%}  ← PRIMARY METRIC")
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
