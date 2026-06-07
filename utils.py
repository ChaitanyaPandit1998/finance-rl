"""
Shared utilities for the finance-rl pipeline.
"""
import json
import re
from pathlib import Path


def strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from generated output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def extract_final_answer(text: str) -> str | None:
    """Extract answer from <answer>X</answer> tag, or fall back to last number."""
    match = re.search(r"<answer>\s*(.+?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else None


FINQA_URLS = {
    "train":      "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "validation": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "test":       "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
}


def load_finqa(split: str = "test", cache_dir: str | None = None):
    """Load FinQA from the original GitHub repo, bypassing HuggingFace dataset scripts.

    All HuggingFace mirrors of FinQA (ibm/finqa, dreamerdeo/finqa, etc.) bundle
    a custom finqa.py loading script. datasets v3.0+ removed support for these
    scripts entirely, causing RuntimeError at load time. This function downloads
    the raw JSON directly from czyssrs/FinQA on GitHub and caches it to disk as
    a HuggingFace Dataset so subsequent calls are instant.

    The returned dataset has these columns (matching the HF FinQA schema):
        question  (str)        — the financial question
        answer    (str)        — the numerical answer (from qa.exe_ans)
        pre_text  (list[str])  — sentences before the table in the SEC filing
        post_text (list[str])  — sentences after the table
        table     (list[list]) — the financial table as rows of cells

    Args:
        split: One of 'train', 'validation', 'test'.
        cache_dir: Root directory for the on-disk cache. If None, defaults to
            ~/.cache/huggingface/finqa_github. Pass HF_HOME to co-locate with
            other HuggingFace downloads.

    Returns:
        HuggingFace Dataset object with the requested split.
    """
    import requests
    from datasets import Dataset

    cache_base = Path(cache_dir) / "finqa_github" if cache_dir else \
                 Path.home() / ".cache" / "huggingface" / "finqa_github"
    cache_path = cache_base / split

    # Return cached version if available
    if cache_path.exists():
        print(f"Loading FinQA {split} from cache ({cache_path})...")
        return Dataset.load_from_disk(str(cache_path))

    if split not in FINQA_URLS:
        raise ValueError(f"split must be one of {list(FINQA_URLS)}, got '{split}'")

    print(f"Downloading FinQA {split} from GitHub...")
    resp = requests.get(FINQA_URLS[split], timeout=60)
    resp.raise_for_status()
    records = resp.json()

    rows = []
    for r in records:
        qa = r.get("qa", {})
        # exe_ans is the cleaned numerical result; fall back to answer string
        exe = qa.get("exe_ans")
        answer = str(exe) if exe is not None else str(qa.get("answer", ""))
        rows.append({
            "question": qa.get("question", ""),
            "answer":   answer,
            "pre_text": r.get("pre_text", []),
            "post_text": r.get("post_text", []),
            "table":    r.get("table", []),
        })

    dataset = Dataset.from_list(rows)
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_path))
    print(f"Cached {len(dataset)} FinQA {split} examples to {cache_path}")
    return dataset
