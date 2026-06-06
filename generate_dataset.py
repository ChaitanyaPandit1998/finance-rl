import re
import requests
import json
import time
import random
from datasets import Dataset

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"

ANSWER_SYSTEM_PROMPT = """/no_think
Solve the problem step by step. Use this exact format:
Step 1: [calculation] ** Step 2: [calculation] ** ... #### [final number only]

Example:
Step 1: SI = 5000 * 0.06 * 3 = <<5000*0.06*3=900>>900 ** Step 2: Total = 5000 + 900 = <<5000+900=5900>>5900 #### 5900
"""

def r(a, b, step=1):
    return random.randrange(a, b, step)

def rf(a, b):
    return round(random.uniform(a, b), 2)

QUESTION_TEMPLATES = [
    (
        "Simple Interest",
        lambda: (
            p := r(1000, 20000, 500),
            rate := r(3, 15),
            t := r(1, 7),
            f"{p} dollars is deposited at {rate}% simple interest per year for {t} years. What is the total interest earned?"
        )[-1]
    ),
    (
        "Compound Interest",
        lambda: (
            p := r(1000, 15000, 500),
            rate := r(4, 12),
            t := r(2, 8),
            f"An investment of ${p} is made at {rate}% annual interest compounded yearly for {t} years. What is the final amount?"
        )[-1]
    ),
    (
        "Rate of Return",
        lambda: (
            invest := r(5000, 50000, 1000),
            final := invest + r(500, 10000, 500),
            t := r(1, 5),
            f"An investor puts in ${invest} and receives ${final} after {t} years. What is the annualized rate of return (as a percentage, rounded to 2 decimal places)?"
        )[-1]
    ),
    (
        "Present Value",
        lambda: (
            fv := r(5000, 30000, 1000),
            rate := r(4, 12),
            t := r(2, 6),
            f"What is the present value of ${fv} to be received {t} years from now, assuming a discount rate of {rate}% per year?"
        )[-1]
    ),
    (
        "Future Value",
        lambda: (
            pv := r(2000, 20000, 500),
            rate := r(4, 10),
            t := r(2, 8),
            f"You invest ${pv} today at an annual interest rate of {rate}%. What will this investment be worth after {t} years?"
        )[-1]
    ),
    (
        "Loan EMI",
        lambda: (
            p := r(10000, 100000, 5000),
            rate := r(6, 18),
            n := r(12, 60, 12),
            f"Calculate the monthly EMI for a loan of ${p} at {rate}% annual interest rate for {n} months."
        )[-1]
    ),
    (
        "Loan Amortization",
        lambda: (
            p := r(10000, 80000, 5000),
            rate := r(6, 15),
            n := r(12, 60, 12),
            f"A loan of ${p} is taken at {rate}% annual interest for {n} months. If the monthly EMI is paid on time, what is the total interest paid over the full loan tenure?"
        )[-1]
    ),
    (
        "Break-even Analysis",
        lambda: (
            fc := r(5000, 30000, 1000),
            vc := r(10, 80),
            sp := vc + r(10, 50),
            f"A business has fixed costs of ${fc}, variable cost of ${vc} per unit, and sells each unit for ${sp}. How many units must be sold to break even?"
        )[-1]
    ),
    (
        "Gross Profit Margin",
        lambda: (
            rev := r(50000, 500000, 10000),
            cogs := r(20000, rev - 5000, 5000),
            f"A company has revenue of ${rev} and cost of goods sold (COGS) of ${cogs}. What is the gross profit margin as a percentage?"
        )[-1]
    ),
    (
        "Dividend Yield",
        lambda: (
            price := r(50, 500, 10),
            div := r(1, 20),
            shares := r(100, 1000, 100),
            f"An investor holds {shares} shares of a company priced at ${price} each. The company pays an annual dividend of ${div} per share. What is the dividend yield as a percentage?"
        )[-1]
    ),
    (
        "Earnings Per Share",
        lambda: (
            net := r(100000, 1000000, 50000),
            pref := r(5000, 20000, 1000),
            shares := r(10000, 100000, 5000),
            f"A company has a net income of ${net}, preferred dividends of ${pref}, and {shares} outstanding shares. What is the earnings per share (EPS)?"
        )[-1]
    ),
    (
        "Straight-Line Depreciation",
        lambda: (
            cost := r(10000, 100000, 5000),
            salvage := r(1000, 5000, 500),
            life := r(3, 10),
            f"A machine costs ${cost} and has a salvage value of ${salvage} after {life} years. Using straight-line depreciation, what is the annual depreciation amount?"
        )[-1]
    ),
    (
        "Payback Period",
        lambda: (
            invest := r(20000, 100000, 5000),
            annual := r(5000, 25000, 1000),
            f"A project requires an initial investment of ${invest} and generates annual cash inflows of ${annual}. What is the payback period in years?"
        )[-1]
    ),
    (
        "Portfolio Return",
        lambda: (
            w1 := r(20, 60),
            w2 := 100 - w1,
            r1 := r(5, 20),
            r2 := r(3, 15),
            f"A portfolio has {w1}% invested in Stock A with a return of {r1}% and {w2}% in Stock B with a return of {r2}%. What is the weighted average portfolio return?"
        )[-1]
    ),
    (
        "Inflation Adjustment",
        lambda: (
            nominal := r(6, 15),
            inflation := r(2, 6),
            f"An investment has a nominal return of {nominal}%. If the inflation rate is {inflation}%, what is the real rate of return (as a percentage, rounded to 2 decimal places)?"
        )[-1]
    ),
    (
        "Currency Exchange",
        lambda: (
            amount := r(500, 10000, 500),
            rate := rf(1.1, 1.5),
            fee := r(1, 4),
            f"Convert ${amount} USD to Euros at an exchange rate of {rate} USD per Euro. The bank charges a {fee}% transaction fee. How many Euros will you receive after the fee?"
        )[-1]
    ),
    (
        "Capital Gains Tax",
        lambda: (
            buy := r(5000, 30000, 1000),
            sell := buy + r(2000, 15000, 500),
            tax_rate := r(10, 30, 5),
            f"An investor bought shares for ${buy} and sold them for ${sell}. If the capital gains tax rate is {tax_rate}%, how much tax is owed?"
        )[-1]
    ),
    (
        "Working Capital",
        lambda: (
            ca := r(50000, 300000, 10000),
            cl := r(20000, ca - 5000, 5000),
            f"A company has current assets of ${ca} and current liabilities of ${cl}. What is the working capital?"
        )[-1]
    ),
    (
        "Net Present Value",
        lambda: (
            invest := r(10000, 50000, 5000),
            cf := r(3000, 15000, 1000),
            rate := r(5, 12),
            n := r(3, 5),
            f"A project requires an initial investment of ${invest} and generates a cash flow of ${cf} per year for {n} years. At a discount rate of {rate}%, what is the NPV?"
        )[-1]
    ),
    (
        "Bond Current Yield",
        lambda: (
            face := r(1000, 10000, 1000),
            coupon_rate := r(4, 10),
            market_price := r(800, face + 200, 50),
            f"A bond with a face value of ${face} has a coupon rate of {coupon_rate}%. If the bond is currently trading at ${market_price}, what is the current yield (as a percentage)?"
        )[-1]
    ),
]


def generate_answer(question: str) -> str:
    print(f"  --> Calling Ollama...", flush=True)
    start = time.time()
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        "stream": True,
        "options": {"temperature": 0.3, "num_ctx": 1024},
        "think": False
    }, timeout=120, stream=True)
    response.raise_for_status()

    full_text = ""
    print("  --> ", end="", flush=True)
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("message", {}).get("content", "")
        full_text += token
        print(token, end="", flush=True)
        if chunk.get("done"):
            break

    elapsed = time.time() - start
    print(f"\n  --> Done in {elapsed:.1f}s", flush=True)
    return full_text.strip()


def extract_final_answer(text: str) -> str | None:
    if "####" in text:
        return text.split("####")[-1].strip().split()[0]
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


def build_dataset(
    num_samples: int = 200,
    save_path: str = "finance_dataset.jsonl",
) -> Dataset:
    records = []
    skipped = 0

    with open(save_path, "w") as f:
        for i in range(num_samples):
            topic, template_fn = random.choice(QUESTION_TEMPLATES)
            print(f"\n[{i + 1}/{num_samples}] Topic: {topic}", flush=True)

            try:
                question = template_fn()
                print(f"  Q: {question}", flush=True)

                answer = generate_answer(question)
                final = extract_final_answer(answer)

                if not final:
                    print(f"  Skipped — could not extract answer", flush=True)
                    print(f"  Raw: {answer[:150]}", flush=True)
                    skipped += 1
                    continue

                record = {"topic": topic, "question": question, "answer": answer}
                records.append(record)
                f.write(json.dumps(record) + "\n")
                print(f"  Saved — #### {final} | Total: {len(records)}", flush=True)

            except Exception as e:
                print(f"  Error: {type(e).__name__}: {e}", flush=True)
                skipped += 1

            time.sleep(0.3)

    print(f"\nDone. Generated: {len(records)} | Skipped: {skipped}")
    dataset = Dataset.from_list(records)
    parquet_path = save_path.replace(".jsonl", ".parquet")
    dataset.to_parquet(parquet_path)
    print(f"Saved to: {parquet_path}")
    print(dataset)
    return dataset


if __name__ == "__main__":
    dataset = build_dataset(num_samples=200, save_path="finance_dataset.jsonl")
