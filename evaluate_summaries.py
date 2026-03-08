#!/usr/bin/env python3
"""
CNN/DM Hallucination Evaluation Pipeline
=========================================
1. Sample 2,000 examples from CNN/DailyMail (test split).
2. Generate summaries with LLaMA via OpenRouter.
3. Evaluate faithfulness with GPT-5.3 via OpenRouter.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python evaluate_summaries.py                       # full 2000-example run
    python evaluate_summaries.py --num-samples 5       # quick test run
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import aiohttp
from datasets import load_dataset
from tqdm import tqdm

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
LLAMA_MODEL = "meta-llama/llama-3.1-8b-instruct"
GPT53_MODEL = "openai/gpt-5.3-chat"

SUMMARISATION_SYSTEM_PROMPT = (
    "You are a helpful assistant. Summarise the following document concisely "
    "and accurately. Do not add any information that is not in the document."
)

EVALUATION_PROMPT_TEMPLATE = """You will be provided with a document and a proposed summary. Your task is to determine if the
proposed summary can be directly inferred from the document. If the summary contains any information
not found in the document, it is considered false. Even if the summary is different from a ground
truth summary, it might still be true, as long as it doesn't contain false information.
For each proposed summary, explain why it is true or false based on the information from the
document. Focus only on the original document's content, disregarding any external context.
After your explanation, give your final conclusion as Conclusion: True if the proposed summary is
completely accurate based on the document, or Conclusion: False if it contains any incorrect or
unsupported information. If your conclusion is 'False', identify the exact phrases or name entities
from the summary that is incorrect by stating Problematic Spans: [the inaccurate text spans from
the summary, in Python list of strings format].
#Document#: {document}
#Ground Truth Summary#: {ground_truth_summary}
#Proposed Summary#: {response}
Write your explanation first, and then give your final conclusion as Conclusion: True if
the proposed summary is completely accurate based on the document, or Conclusion: False if it
contains any incorrect or unsupported information. Add Problematic Spans: [the exact inaccurate
text spans from the summary, in a list of strings] if your conclusion is 'False'."""


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning an empty list if it doesn't exist."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict):
    """Append a single JSON record to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


async def call_openrouter(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 5,
) -> str:
    """Call the OpenRouter chat completions endpoint with retry + exponential backoff."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/hallucination-eval",
        "X-Title": "Hallucination Evaluation Pipeline",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            async with session.post(
                OPENROUTER_BASE_URL, headers=headers, json=payload
            ) as resp:
                if resp.status == 429:
                    # Rate limited — back off
                    wait = 2 ** attempt + 1
                    print(f"  [rate-limited] waiting {wait}s …")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except (aiohttp.ClientError, KeyError) as e:
            wait = 2 ** attempt + 1
            print(f"  [error] {e} — retrying in {wait}s …")
            await asyncio.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} retries (model={model})")


def parse_evaluation(text: str) -> dict:
    """Extract Conclusion and optional Problematic Spans from GPT-5.3 output."""
    result: dict = {"raw_evaluation": text, "conclusion": None, "problematic_spans": None}

    # Parse conclusion
    conclusion_match = re.search(
        r"Conclusion:\s*(True|False)", text, re.IGNORECASE
    )
    if conclusion_match:
        result["conclusion"] = conclusion_match.group(1).capitalize() == "True"

    # Parse problematic spans (if False)
    spans_match = re.search(
        r"Problematic Spans:\s*(\[.*?\])", text, re.DOTALL
    )
    if spans_match:
        try:
            result["problematic_spans"] = json.loads(
                spans_match.group(1).replace("'", '"')
            )
        except json.JSONDecodeError:
            result["problematic_spans"] = spans_match.group(1)

    return result


# ──────────────────────────────────────────────
# Stage 1 — Sample dataset
# ──────────────────────────────────────────────
def stage_sample_dataset(num_samples: int, seed: int, output_dir: Path) -> list[dict]:
    """Load CNN/DM and save a random subset."""
    out_path = output_dir / "cnndm_subset.jsonl"

    existing = load_jsonl(out_path)
    if len(existing) >= num_samples:
        print(f"[Stage 1] Loaded {len(existing)} cached examples from {out_path}")
        return existing[:num_samples]

    print(f"[Stage 1] Loading CNN/DailyMail and sampling {num_samples} examples …")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
    ds_shuffled = ds.shuffle(seed=seed)
    subset = ds_shuffled.select(range(min(num_samples, len(ds_shuffled))))

    records = []
    with open(out_path, "w") as f:
        for i, row in enumerate(subset):
            rec = {
                "id": i,
                "article": row["article"],
                "highlights": row["highlights"],
            }
            f.write(json.dumps(rec) + "\n")
            records.append(rec)

    print(f"[Stage 1] Saved {len(records)} examples to {out_path}")
    return records


# ──────────────────────────────────────────────
# Stage 2 — Generate summaries
# ──────────────────────────────────────────────
async def stage_generate_summaries(
    records: list[dict],
    api_key: str,
    concurrency: int,
    output_dir: Path,
) -> list[dict]:
    """Generate summaries for each example with LLaMA."""
    out_path = output_dir / "llama_summaries.jsonl"
    existing = load_jsonl(out_path)
    done_ids = {r["id"] for r in existing}
    results = list(existing)

    todo = [r for r in records if r["id"] not in done_ids]
    if not todo:
        print(f"[Stage 2] All {len(results)} summaries already generated.")
        return results

    print(f"[Stage 2] Generating summaries for {len(todo)} examples (concurrency={concurrency}) …")
    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(todo), desc="Summarising")

    async def _summarise(session: aiohttp.ClientSession, rec: dict):
        async with sem:
            messages = [
                {"role": "system", "content": SUMMARISATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Document:\n\n{rec['article']}"},
            ]
            summary = await call_openrouter(
                session, api_key, LLAMA_MODEL, messages, temperature=0.0, max_tokens=512
            )
            out = {**rec, "llama_summary": summary}
            append_jsonl(out_path, out)
            results.append(out)
            pbar.update(1)

    async with aiohttp.ClientSession() as session:
        tasks = [_summarise(session, rec) for rec in todo]
        await asyncio.gather(*tasks, return_exceptions=True)

    pbar.close()
    # Check for errors
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        print(f"  ⚠ {len(errors)} errors during summarisation")

    print(f"[Stage 2] {len(results)} summaries ready.")
    return [r for r in results if isinstance(r, dict)]


# ──────────────────────────────────────────────
# Stage 3 — Evaluate faithfulness
# ──────────────────────────────────────────────
async def stage_evaluate(
    summaries: list[dict],
    api_key: str,
    concurrency: int,
    output_dir: Path,
) -> list[dict]:
    """Evaluate each summary for faithfulness with GPT-5.3."""
    out_path = output_dir / "gpt53_evaluations.jsonl"
    existing = load_jsonl(out_path)
    done_ids = {r["id"] for r in existing}
    results = list(existing)

    todo = [r for r in summaries if r["id"] not in done_ids]
    if not todo:
        print(f"[Stage 3] All {len(results)} evaluations already complete.")
        return results

    print(f"[Stage 3] Evaluating {len(todo)} summaries (concurrency={concurrency}) …")
    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(todo), desc="Evaluating")

    async def _evaluate(session: aiohttp.ClientSession, rec: dict):
        async with sem:
            prompt = EVALUATION_PROMPT_TEMPLATE.format(
                document=rec["article"],
                ground_truth_summary=rec["highlights"],
                response=rec["llama_summary"],
            )
            messages = [{"role": "user", "content": prompt}]
            raw = await call_openrouter(
                session, api_key, GPT53_MODEL, messages, temperature=0.0, max_tokens=1024
            )
            parsed = parse_evaluation(raw)
            out = {**rec, **parsed}
            append_jsonl(out_path, out)
            results.append(out)
            pbar.update(1)

    async with aiohttp.ClientSession() as session:
        tasks = [_evaluate(session, rec) for rec in todo]
        await asyncio.gather(*tasks, return_exceptions=True)

    pbar.close()
    print(f"[Stage 3] {len(results)} evaluations ready.")
    return [r for r in results if isinstance(r, dict)]


# ──────────────────────────────────────────────
# Stage 4 — Aggregate and report
# ──────────────────────────────────────────────
def stage_aggregate(evaluations: list[dict], output_dir: Path):
    """Save final merged results and print summary stats."""
    out_path = output_dir / "final_results.jsonl"
    with open(out_path, "w") as f:
        for rec in evaluations:
            f.write(json.dumps(rec) + "\n")

    total = len(evaluations)
    true_count = sum(1 for r in evaluations if r.get("conclusion") is True)
    false_count = sum(1 for r in evaluations if r.get("conclusion") is False)
    unknown = total - true_count - false_count

    print("\n" + "=" * 60)
    print("FAITHFULNESS EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total examples evaluated : {total}")
    print(f"  Faithful   (True)        : {true_count}  ({100*true_count/max(total,1):.1f}%)")
    print(f"  Unfaithful (False)       : {false_count}  ({100*false_count/max(total,1):.1f}%)")
    if unknown:
        print(f"  Unparseable              : {unknown}  ({100*unknown/max(total,1):.1f}%)")
    print("=" * 60)
    print(f"\nFull results saved to {out_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLaMA summaries on CNN/DM for faithfulness."
    )
    parser.add_argument(
        "--num-samples", type=int, default=2000,
        help="Number of CNN/DM examples to sample (default: 2000)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory for all output files (default: output)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for dataset shuffling (default: 42)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set the OPENROUTER_API_KEY environment variable before running.\n"
            "  export OPENROUTER_API_KEY='sk-or-...'"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Stage 1 — Dataset
    records = stage_sample_dataset(args.num_samples, args.seed, output_dir)

    # Stage 2 — Summarise
    summaries = await stage_generate_summaries(records, api_key, args.concurrency, output_dir)

    # Stage 3 — Evaluate
    evaluations = await stage_evaluate(summaries, api_key, args.concurrency, output_dir)

    # Stage 4 — Report
    stage_aggregate(evaluations, output_dir)

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    asyncio.run(main())
