#!/usr/bin/env python3
"""
Benchmark Qwen3-8B (baseline or LoRA checkpoint) on gbharti/finance-alpaca.

Usage:
    # Baseline (no fine-tuning)
    python benchmark.py --tag baseline

    # After SFT
    python benchmark.py --checkpoint checkpoints/sft --tag sft

    # After GRPO
    python benchmark.py --checkpoint checkpoints/grpo --tag grpo

    # On a budget GPU
    python benchmark.py --tag baseline --use-qlora
"""
import argparse
import json
import random
import sys
import time
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


def rouge_l(predictions: list[str], references: list[str]) -> list[float]:
    """Compute ROUGE-L F1 score for each prediction/reference pair.

    ROUGE-L measures the longest common subsequence (LCS) between two texts.
    Unlike ROUGE-1/2 which count n-gram overlaps, LCS captures sentence-level
    structure — words don't need to be adjacent to contribute. The F1 score
    balances precision (fraction of prediction covered by LCS) and recall
    (fraction of reference covered by LCS).

    Args:
        predictions: Model-generated responses.
        references: Ground truth answers from the dataset.

    Returns:
        List of ROUGE-L F1 scores in [0, 1], one per pair.
    """
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]


def bertscore(predictions: list[str], references: list[str]) -> list[float]:
    """Compute BERTScore F1 for each prediction/reference pair.

    BERTScore embeds both texts with a pretrained BERT model and measures
    cosine similarity between token embeddings, making it sensitive to
    semantic meaning rather than surface word overlap. This catches cases
    where the model uses different but correct phrasing that ROUGE-L would
    penalise.

    Args:
        predictions: Model-generated responses.
        references: Ground truth answers from the dataset.

    Returns:
        List of BERTScore F1 values in [0, 1], one per pair.
    """
    from bert_score import score as bs_score
    _, _, f1 = bs_score(predictions, references, lang="en", verbose=False)
    return f1.tolist()


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
        sps = done / elapsed
        print(f"  [{done}/{len(samples)}] {sps:.1f} samples/s")

    print("\nScoring ROUGE-L...")
    rouge_scores = rouge_l(all_predictions, all_references)

    bert_scores = None
    if not args.no_bertscore:
        print("Scoring BERTScore (this may take a moment)...")
        bert_scores = bertscore(all_predictions, all_references)

    def percentile(data, p):
        data = sorted(data)
        idx = int(len(data) * p / 100)
        return round(data[min(idx, len(data) - 1)], 4)

    summary = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "num_samples": len(samples),
        "rouge_l": {
            "mean": round(sum(rouge_scores) / len(rouge_scores), 4),
            "p25": percentile(rouge_scores, 25),
            "p50": percentile(rouge_scores, 50),
            "p75": percentile(rouge_scores, 75),
        },
    }
    if bert_scores:
        summary["bert_score_f1"] = {
            "mean": round(sum(bert_scores) / len(bert_scores), 4),
            "p25": percentile(bert_scores, 25),
            "p50": percentile(bert_scores, 50),
            "p75": percentile(bert_scores, 75),
        }

    records = []
    for i, (instr, ref, pred, r) in enumerate(
        zip(all_instructions, all_references, all_predictions, rouge_scores)
    ):
        record = {
            "instruction": instr,
            "reference": ref,
            "prediction": pred,
            "rouge_l": round(r, 4),
        }
        if bert_scores:
            record["bert_score_f1"] = round(bert_scores[i], 4)
        records.append(record)

    out = {"summary": summary, "records": records}
    out_path = f"results/benchmark_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Tag:         {args.tag}")
    print(f"ROUGE-L:     mean={summary['rouge_l']['mean']}  p50={summary['rouge_l']['p50']}")
    if bert_scores:
        print(f"BERTScore:   mean={summary['bert_score_f1']['mean']}  p50={summary['bert_score_f1']['p50']}")
    print(f"Results:     {out_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
