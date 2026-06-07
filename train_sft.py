#!/usr/bin/env python3
"""
SFT fine-tuning of Qwen3-8B on Finance Alpaca + FinQA using Unsloth + LoRA.

Both datasets are combined before training so the model learns:
  - Finance Alpaca (~68k): broad financial instruction-following and terminology
  - FinQA (~6k):          numerical reasoning over financial tables/documents

The FinQA examples establish the '#### <answer>' output format that the
dual reward in train_grpo.py later reinforces.

Usage:
    python train_sft.py

    # Skip FinQA supplementation (Finance Alpaca only)
    python train_sft.py --finqa-samples 0

    # QLoRA fallback for GPUs with < 24GB VRAM
    python train_sft.py --use-qlora

    # Custom hyperparameters
    python train_sft.py --rank 32 --epochs 2 --batch-size 4
"""
import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads HF_TOKEN and HF_HOME from .env into os.environ before any HF imports
except ImportError:
    print("Warning: python-dotenv not installed. Set HF_TOKEN, HF_HOME, "
          "CHECKPOINT_DIR manually in your shell if needed.")

import os
HF_CACHE: str | None = os.getenv("HF_HOME")
CHECKPOINT_DIR: str = os.getenv("CHECKPOINT_DIR", "checkpoints")

try:
    import unsloth  # must be first — patches transformers/trl/peft before they load
    import torch
    from datasets import Dataset, concatenate_datasets, load_dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
except ImportError as e:
    if "--help" not in sys.argv and "-h" not in sys.argv:
        raise

from utils import extract_final_answer, strip_thinking

BASE_MODEL = "Qwen/Qwen3-8B"
SYSTEM_PROMPT = "You are a helpful financial assistant. Answer concisely and accurately."


def format_alpaca_example(example: dict, tokenizer) -> str:
    """Format a single Finance Alpaca example into a full training string.

    Applies Qwen3's native chat template to produce a string containing the
    system prompt, user message, and expected assistant response. The tokenizer's
    apply_chat_template handles all special tokens (<|im_start|>, <|im_end|>)
    so the model trains on the same format it was pre-trained on.

    Args:
        example: Dict with keys 'instruction', 'input' (optional), 'output'.
        tokenizer: Qwen3 tokenizer used to apply the chat template.

    Returns:
        Single string representing the full conversation, ready for tokenization.
    """
    user_content = example["instruction"]
    if example.get("input", "").strip():
        user_content += f"\n\n{example['input']}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def format_finqa_example(example: dict, tokenizer) -> str:
    """Format a single FinQA example into a full training string.

    FinQA examples contain a financial table plus surrounding text extracted
    from real SEC filings, paired with a numerical question and answer.
    The table is formatted as pipe-separated rows so the model can parse
    column relationships. The answer is wrapped in '#### <answer>' to
    establish the extraction format that the GRPO dual reward later targets.

    Args:
        example: FinQA dataset row with 'pre_text', 'table', 'post_text',
                 'question', and 'answer' keys.
        tokenizer: Qwen3 tokenizer used to apply the chat template.

    Returns:
        Single string representing the full conversation, ready for tokenization.
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
        f"Financial Context:\n{context}\n\nQuestion: {example['question']}\n\n"
        f"Provide step-by-step reasoning, then wrap your final numerical answer "
        f"in <answer></answer> tags. Example: <answer>42.5</answer>"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": f"<answer>{example['answer']}</answer>"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_combined_dataset(alpaca_raw, finqa_raw, tokenizer) -> Dataset:
    """Pre-format and merge Finance Alpaca and FinQA into a single text dataset.

    Both datasets are mapped to a common single-column schema ('text') so
    SFTTrainer can consume them uniformly. Pre-formatting here rather than
    using a batch formatting_func avoids schema mismatches between datasets
    and makes the combined shuffle straightforward.

    Args:
        alpaca_raw: Raw Finance Alpaca HuggingFace dataset split.
        finqa_raw: Raw FinQA HuggingFace dataset split, or None to skip.
        tokenizer: Qwen3 tokenizer passed through to the formatting functions.

    Returns:
        Shuffled Dataset with a single 'text' column containing all examples.
    """
    print(f"  Formatting {len(alpaca_raw)} Finance Alpaca examples...")
    alpaca_formatted = alpaca_raw.map(
        lambda ex: {"text": format_alpaca_example(ex, tokenizer)},
        remove_columns=alpaca_raw.column_names,
    )

    if finqa_raw is None:
        return alpaca_formatted

    print(f"  Formatting {len(finqa_raw)} FinQA examples...")
    finqa_formatted = finqa_raw.map(
        lambda ex: {"text": format_finqa_example(ex, tokenizer)},
        remove_columns=finqa_raw.column_names,
    )

    combined = concatenate_datasets([alpaca_formatted, finqa_formatted])
    return combined.shuffle(seed=42)


def _build_inference_prompt(example: dict, tokenizer) -> str:
    """Build an inference-only prompt for a FinQA example (no assistant turn)."""
    pre = " ".join(example.get("pre_text") or [])
    post = " ".join(example.get("post_text") or [])
    table = example.get("table") or []
    if isinstance(table, list) and table:
        table_str = "\n".join(" | ".join(str(cell) for cell in row) for row in table)
    else:
        table_str = str(table) if table else ""
    context = "\n\n".join(p for p in [pre, table_str, post] if p.strip())
    user_content = (
        f"/no_think\n"
        f"Financial Context:\n{context}\n\n"
        f"Question: {example['question']}\n\n"
        f"Provide step-by-step reasoning, then wrap your final numerical answer "
        f"in <answer></answer> tags. Example: <answer>42.5</answer>"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class SamplePreviewCallback(TrainerCallback):
    """Runs inference on a small probe set at the end of each epoch and prints previews."""

    def __init__(self, probe_examples: list, tokenizer, device):
        self.probe_examples = probe_examples
        self.tokenizer = tokenizer
        self.device = device

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not self.probe_examples or model is None:
            return

        print(f"\n{'─'*60}")
        print(f"  Sample previews — end of epoch {int(state.epoch)}")
        print(f"{'─'*60}")

        self.tokenizer.padding_side = "left"
        model.eval()

        for i, ex in enumerate(self.probe_examples):
            prompt = _build_inference_prompt(ex, self.tokenizer)
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = output[:, inputs["input_ids"].shape[1]:]
            pred = strip_thinking(self.tokenizer.decode(new_tokens[0], skip_special_tokens=True))
            extracted = extract_final_answer(pred)
            ref = str(ex["answer"])
            q = ex["question"]

            print(f"\n  [{i+1}/{len(self.probe_examples)}] Q: {q[:100]}{'...' if len(q) > 100 else ''}")
            print(f"  Ref:       {ref}")
            print(f"  Extracted: {extracted or '(none)'}")
            if len(pred) > 250:
                print(f"  Ans start: {pred[:125]}...")
                print(f"  Ans end:   ...{pred[-125:]}")
            else:
                print(f"  Ans:       {pred}")

        self.tokenizer.padding_side = "right"
        model.train()
        print(f"\n{'─'*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-qlora", action="store_true", help="4-bit QLoRA (for GPUs with < 24GB VRAM)")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--output-dir", type=str, default=f"{CHECKPOINT_DIR}/sft")
    parser.add_argument("--max-steps", type=int, default=-1, help="Override epochs with fixed step count")
    parser.add_argument("--finqa-samples", type=int, default=-1,
                        help="FinQA examples to add to SFT (-1 = all ~6k, 0 = skip FinQA)")
    parser.add_argument("--preview-samples", type=int, default=5,
                        help="FinQA test examples to preview after each epoch (0 = disable)")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    from unsloth import FastLanguageModel

    print(f"Loading {BASE_MODEL} ({'4-bit QLoRA' if args.use_qlora else 'bfloat16 LoRA'})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=args.max_seq_len,
        dtype=None if args.use_qlora else torch.bfloat16,
        load_in_4bit=args.use_qlora,
        cache_dir=HF_CACHE,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading datasets...")
    alpaca_raw = load_dataset("gbharti/finance-alpaca", split="train", cache_dir=HF_CACHE)

    finqa_raw = None
    if args.finqa_samples != 0:
        from utils import load_finqa
        finqa_raw = load_finqa(split="train", cache_dir=HF_CACHE)
        if args.finqa_samples > 0 and args.finqa_samples < len(finqa_raw):
            finqa_raw = finqa_raw.select(range(args.finqa_samples))

    dataset = build_combined_dataset(alpaca_raw, finqa_raw, tokenizer)
    print(f"Combined dataset: {len(dataset)} examples "
          f"({len(alpaca_raw)} Alpaca + {len(finqa_raw) if finqa_raw else 0} FinQA)")

    probe_examples = []
    if args.preview_samples > 0:
        from utils import load_finqa
        probe_raw = load_finqa(split="test", cache_dir=HF_CACHE)
        probe_examples = [probe_raw[i] for i in range(min(args.preview_samples, len(probe_raw)))]
        print(f"Loaded {len(probe_examples)} FinQA test examples for epoch-end previews")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=not args.use_qlora,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=42,
    )

    callbacks = []
    if probe_examples:
        device = next(model.parameters()).device
        callbacks.append(SamplePreviewCallback(probe_examples, tokenizer, device))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=True,
        callbacks=callbacks or None,
    )

    print(f"\nStarting SFT training...")
    print(f"  Rank: {args.rank} | Alpha: {args.lora_alpha} | LR: {args.lr}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | Grad accum: {args.grad_accum}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")

    trainer.train()

    print(f"\nSaving adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
