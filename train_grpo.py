#!/usr/bin/env python3
"""
GRPO (RL) fine-tuning of Qwen3-8B using Unsloth + TRL.

Dual reward signal (replaces single ROUGE-L):
  - Format reward  (0.2): completion contains a '#### <number>' answer marker
  - Accuracy reward (0.8): extracted number matches the reference within 1% tolerance

Together these fix the "fluent but wrong" problem: ROUGE-L rewards verbose,
well-worded answers even when the number is incorrect. The dual reward only
gives full credit when the answer is both structured AND numerically right.

Default dataset is FinQA (numerical, verifiable) rather than Finance Alpaca
(prose, unverifiable) because GRPO needs a reward signal it can actually trust.
Use --dataset alpaca to fall back to Finance Alpaca with ROUGE-L reward.

Usage:
    # Default: FinQA with dual reward (recommended)
    python train_grpo.py --base-checkpoint checkpoints/sft

    # Finance Alpaca with ROUGE-L reward (weaker signal)
    python train_grpo.py --base-checkpoint checkpoints/sft --dataset alpaca

    # QLoRA fallback
    python train_grpo.py --use-qlora --base-checkpoint checkpoints/sft
"""
import argparse
import re
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
    import torch
    from datasets import load_dataset
    from rouge_score import rouge_scorer as rs
    from trl import GRPOConfig, GRPOTrainer
except ImportError as e:
    if "--help" not in sys.argv and "-h" not in sys.argv:
        raise

BASE_MODEL = "Qwen/Qwen3-8B"
SYSTEM_PROMPT = "You are a helpful financial assistant. Answer concisely and accurately."

# Reward weights — must sum to 1.0
FORMAT_REWARD = 0.2
ACCURACY_REWARD = 0.8


def format_alpaca_prompt(instruction: str, input_text: str, tokenizer) -> str:
    """Build a chat-formatted prompt for a Finance Alpaca example.

    Args:
        instruction: The finance question or task description.
        input_text: Optional additional context (empty string if not provided).
        tokenizer: Qwen3 tokenizer used to apply the chat template.

    Returns:
        Fully formatted prompt string with generation marker appended.
    """
    user_content = instruction
    if input_text.strip():
        user_content += f"\n\n{input_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_finqa_prompt(example: dict, tokenizer) -> str:
    """Build a chat-formatted prompt for a FinQA example.

    Includes the financial context (table + surrounding text) so the model
    has the source data needed to compute the answer. Unlike the benchmark
    version, this does NOT inject /no_think — GRPO benefits from the model's
    full reasoning capacity when generating candidate completions.

    Args:
        example: FinQA dataset row with 'pre_text', 'table', 'post_text', 'question'.
        tokenizer: Qwen3 tokenizer used to apply the chat template.

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
        f"Financial Context:\n{context}\n\nQuestion: {example['question']}\n\n"
        f"Provide step-by-step reasoning, then wrap your final numerical answer "
        f"in <answer></answer> tags. Example: <answer>42.5</answer>"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_final_answer(text: str) -> str | None:
    """Extract the answer from a completion.

    Captures any content from <answer>...</answer> tags — numbers, 'yes',
    'no', or any text — matching how the model is trained during SFT.
    Falls back to the last number in the text for partially-formatted outputs.

    Args:
        text: Raw model completion string.

    Returns:
        Extracted answer string, or None if not found.
    """
    match = re.search(r"<answer>\s*(.+?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else None


def score_format(completion: str) -> float:
    """Return FORMAT_REWARD if the completion contains a valid <answer>X</answer> tag.

    Rewards the model for producing a structured, extractable answer in
    Fin-R1's format rather than burying the result in free-form prose.

    Args:
        completion: Raw model completion string.

    Returns:
        FORMAT_REWARD (0.2) or 0.0.
    """
    return FORMAT_REWARD if re.search(r"<answer>\s*-?[\d,]+\.?\d*%?\s*</answer>", completion) else 0.0


def _scale_match(p: float, r: float, tol: float = 0.001) -> bool:
    """Check numerical equality at the same scale or with a ×100 correction.

    FinQA annotations are inconsistent — some percentage answers are stored
    as raw decimals (0.1822) while others are stored as whole percentages
    (18.22). Same-scale comparison uses decimal-place exact match so 18.21
    does NOT match 18.22. Cross-scale comparison uses relative tolerance so
    18.21 / 100 = 0.1821 DOES match ref 0.1822 (0.055% error < 0.1% tol).

    Args:
        p: Predicted float value.
        r: Reference float value.
        tol: Relative tolerance for cross-scale check (default 0.001 = 0.1%).

    Returns:
        True if values match at same scale (exact) or cross scale (within tol).
    """
    if r == 0:
        return abs(p) < tol

    ref_str = f"{r}"
    ref_decimals = len(ref_str.split(".")[-1]) if "." in ref_str else 0
    ref_decimals = min(ref_decimals, 4)
    if round(p, ref_decimals) == round(r, ref_decimals):
        return True
    if abs(p / 100 - r) / abs(r) < tol:
        return True
    if abs(p * 100 - r) / abs(r) < tol:
        return True
    return False


def score_accuracy(completion: str, reference: str) -> float:
    """Return ACCURACY_REWARD if the extracted answer matches the reference.

    Deliberately requires an <answer> tag before checking the number, matching
    Fin-R1's coupled format+accuracy reward design. A correct number buried in
    prose without the tag scores 0.0, not 0.8 — this prevents the model from
    gaming the reward by omitting the structured format.

    Uses _scale_match() to handle FinQA's percentage/decimal annotation
    inconsistency so the model is not penalised for correctly computing 18.21%
    when the reference happens to be stored as 0.1822.

    Args:
        completion: Raw model completion string.
        reference: Ground truth answer string from the dataset.

    Returns:
        ACCURACY_REWARD (0.8) if format marker present and answer correct, else 0.0.
    """
    if not re.search(r"<answer>", completion):
        return 0.0
    pred_str = extract_final_answer(completion)
    if pred_str is None:
        return 0.0
    ref_str = str(reference).replace(",", "").replace("%", "").strip()
    try:
        return ACCURACY_REWARD if _scale_match(float(pred_str), float(ref_str)) else 0.0
    except ValueError:
        return ACCURACY_REWARD if pred_str.strip() == ref_str else 0.0


def build_dual_reward_fn():
    """Create the dual reward function (format + accuracy) for FinQA GRPO.

    The two components are additive:
      - Format reward  (0.2): was a structured '#### <answer>' produced?
      - Accuracy reward (0.8): does the extracted number match the reference?

    A fluent but numerically wrong answer scores 0.2 at most. A correct
    answer in the right format scores 1.0. This directly penalises the
    "fluent but wrong" failure mode of ROUGE-L.

    Returns:
        reward_dual: Callable consumed by GRPOTrainer at each training step.
    """
    def reward_dual(completions: list[str], **kwargs) -> list[float]:
        """Score each completion with format + accuracy rewards.

        Called by GRPOTrainer once per training step. The 'reference' column
        comes from the dataset prepared by prepare_finqa().

        Args:
            completions: Model-generated response strings for the current batch.
            **kwargs: Dataset columns; must include 'reference' (ground truth).

        Returns:
            List of scores in [0.0, 1.0], one per completion.
        """
        references = kwargs["reference"]
        return [
            score_format(comp) + score_accuracy(comp, ref)
            for comp, ref in zip(completions, references)
        ]

    return reward_dual


def build_rouge_reward_fn():
    """Create a ROUGE-L reward function for Finance Alpaca GRPO (fallback).

    Used when --dataset alpaca is specified. Weaker than dual reward because
    ROUGE-L cannot distinguish correct from incorrect numerical answers.
    The RougeScorer is created once via closure and reused across all steps.

    Returns:
        reward_rouge: Callable consumed by GRPOTrainer at each training step.
    """
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)

    def reward_rouge(completions: list[str], **kwargs) -> list[float]:
        """Score each completion with ROUGE-L F1 against the reference answer.

        Args:
            completions: Model-generated response strings for the current batch.
            **kwargs: Dataset columns; must include 'reference' (ground truth).

        Returns:
            List of ROUGE-L F1 scores in [0, 1], one per completion.
        """
        references = kwargs["reference"]
        return [
            scorer.score(ref, comp)["rougeL"].fmeasure
            for comp, ref in zip(completions, references)
        ]

    return reward_rouge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["finqa", "alpaca"], default="finqa",
                        help="Training dataset: finqa uses dual reward, alpaca uses ROUGE-L")
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--base-checkpoint", type=str, default=None,
                        help="Start from this LoRA adapter (e.g. checkpoints/sft) instead of base model")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Completions per prompt — minimum 2, higher = more stable, more VRAM")
    parser.add_argument("--max-prompt-len", type=int, default=1024,
                        help="FinQA prompts include table context and need more tokens than Alpaca")
    parser.add_argument("--max-completion-len", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int, default=-1,
                        help="Cap dataset size (-1 = use all examples)")
    parser.add_argument("--output-dir", type=str, default=f"{CHECKPOINT_DIR}/grpo")
    args = parser.parse_args()

    if args.num_generations < 2:
        raise ValueError("--num-generations must be >= 2: GRPO needs at least 2 completions to compute advantages")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    from unsloth import FastLanguageModel

    model_to_load = args.base_checkpoint if args.base_checkpoint else BASE_MODEL
    print(f"Loading {model_to_load} ({'4-bit QLoRA' if args.use_qlora else 'bfloat16 LoRA'})...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_to_load,
        max_seq_length=args.max_prompt_len + args.max_completion_len,
        dtype=None if args.use_qlora else torch.bfloat16,
        load_in_4bit=args.use_qlora,
        cache_dir=HF_CACHE,
    )

    # Only apply PEFT when starting from base model; SFT checkpoint already has LoRA
    if not args.base_checkpoint:
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
    tokenizer.padding_side = "left"

    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "finqa":
        from utils import load_finqa
        raw = load_finqa(split="train", cache_dir=HF_CACHE)
        reward_fn = build_dual_reward_fn()

        def prepare(example):
            """Reshape a FinQA row into the format GRPOTrainer expects.

            GRPOTrainer generates completions from 'prompt' and passes all
            other columns to the reward function via **kwargs. 'reference'
            carries the ground truth numerical answer for the dual reward.

            Args:
                example: FinQA row with pre_text, table, post_text, question, answer.

            Returns:
                Dict with 'prompt' (formatted context+question) and 'reference' (answer).
            """
            return {
                "prompt": format_finqa_prompt(example, tokenizer),
                "reference": str(example["answer"]),
            }

    else:
        raw = load_dataset("gbharti/finance-alpaca", split="train", cache_dir=HF_CACHE)
        reward_fn = build_rouge_reward_fn()

        def prepare(example):
            """Reshape a Finance Alpaca row into the format GRPOTrainer expects.

            Args:
                example: Alpaca row with instruction, input, output keys.

            Returns:
                Dict with 'prompt' and 'reference' (prose reference answer).
            """
            return {
                "prompt": format_alpaca_prompt(
                    example["instruction"], example.get("input") or "", tokenizer
                ),
                "reference": example["output"],
            }

    if args.max_train_samples > 0 and args.max_train_samples < len(raw):
        raw = raw.shuffle(seed=42).select(range(args.max_train_samples))

    dataset = raw.map(prepare, remove_columns=raw.column_names)
    print(f"Training on {len(dataset)} examples | Reward: {'dual (format + accuracy)' if args.dataset == 'finqa' else 'ROUGE-L'}")

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=not args.use_qlora,
        fp16=False,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting GRPO training...")
    print(f"  Base:        {model_to_load}")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Reward:      {'format (0.2) + accuracy (0.8)' if args.dataset == 'finqa' else 'ROUGE-L'}")
    print(f"  Rank: {args.rank} | LR: {args.lr} | Generations/prompt: {args.num_generations}")
    print(f"  Batch: {args.batch_size} | Grad accum: {args.grad_accum}")
    print(f"  Effective GRPO batch: {args.batch_size * args.grad_accum * args.num_generations} completions")

    trainer.train()

    print(f"\nSaving adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
