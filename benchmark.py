"""
RepoMemory Benchmark Suite:
Evaluates RepoMemory tool retrieval against naive whole-file baseline across realistic tasks.
Records prompt tokens, completion tokens, latency, correctness, and token reduction percentage.
Outputs results to results.csv and prints a formatted markdown summary.
"""

import os
import csv
import json
import time
import argparse
from typing import Dict, Any, List

from indexer import RepoIndexer
from agent import CodingAgent


def evaluate_correctness(answer: str, task: Dict[str, Any], tool_calls: List[Dict[str, Any]] = None) -> bool:
    """
    Evaluate if the answer or retrieved context contains the expected keywords or ground truth symbol.
    """
    text_to_check = answer.lower()
    
    # Also check retrieved context from tool calls if present
    if tool_calls:
        for call in tool_calls:
            text_to_check += " " + json.dumps(call).lower()

    # Check ground truth symbol
    gt_symbol = task.get("ground_truth_symbol", "").lower()
    if gt_symbol and gt_symbol in text_to_check:
        return True

    # Check keywords (at least 2 matching or 50% of keywords)
    keywords = [k.lower() for k in task.get("keywords", [])]
    if keywords:
        matches = sum(1 for kw in keywords if kw in text_to_check)
        if matches >= max(1, len(keywords) // 2):
            return True

    return False


def run_benchmark(
    repo_path: str = ".",
    tasks_file: str = "tasks.json",
    output_csv: str = "results.csv",
    reindex: bool = False,
    model: str = "qwen2.5-coder:7b-instruct"
) -> Dict[str, Any]:
    """Execute benchmark over all tasks for both modes."""
    repo_path = os.path.abspath(repo_path)
    print(f"Starting RepoMemory Benchmark on: {repo_path}")

    # 1. Ensure indexing is up to date
    indexer = RepoIndexer(repo_path=repo_path)
    print("Verifying repository index...")
    idx_summary = indexer.index_repo(force=reindex)
    print(f"Index ready: {idx_summary['chunks_added']} new chunks added, {idx_summary['files_scanned']} files scanned.")

    # 2. Load tasks
    if not os.path.isabs(tasks_file):
        tasks_file = os.path.join(repo_path, tasks_file)
    with open(tasks_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"Loaded {len(tasks)} benchmark tasks.")

    agent = CodingAgent(repo_path=repo_path, model=model)
    results = []

    for idx, task in enumerate(tasks, start=1):
        task_id = task.get("id", f"task_{idx}")
        question = task["question"]
        print(f"\n[{idx}/{len(tasks)}] Running {task_id}: '{question[:55]}...'")

        # Run RepoMemory Mode
        res_mem = agent.run_query(question, mode="repomemory")
        mem_correct = evaluate_correctness(res_mem["answer"], task, res_mem.get("tool_calls"))

        # Run Baseline Mode
        res_base = agent.run_query(question, mode="baseline")
        base_correct = evaluate_correctness(res_base["answer"], task)

        # Calculate savings
        savings_pct = (
            (res_base["prompt_tokens"] - res_mem["prompt_tokens"])
            / max(1, res_base["prompt_tokens"])
        ) * 100.0

        print(f"  -> Memory Tokens:   {res_mem['prompt_tokens']} prompt, {res_mem['total_tokens']} total ({res_mem['latency_sec']}s)")
        print(f"  -> Baseline Tokens: {res_base['prompt_tokens']} prompt, {res_base['total_tokens']} total ({res_base['latency_sec']}s)")
        print(f"  -> Token Savings:   {savings_pct:.1f}% reduction")

        results.append({
            "task_id": task_id,
            "category": task.get("category", "general"),
            "question": question,
            "difficulty": task.get("difficulty", "medium"),
            "memory_prompt_tokens": res_mem["prompt_tokens"],
            "memory_completion_tokens": res_mem["completion_tokens"],
            "memory_total_tokens": res_mem["total_tokens"],
            "memory_latency_s": res_mem["latency_sec"],
            "memory_correct": mem_correct,
            "baseline_prompt_tokens": res_base["prompt_tokens"],
            "baseline_completion_tokens": res_base["completion_tokens"],
            "baseline_total_tokens": res_base["total_tokens"],
            "baseline_latency_s": res_base["latency_sec"],
            "baseline_correct": base_correct,
            "token_savings_pct": round(savings_pct, 2)
        })

    # Save CSV
    if not os.path.isabs(output_csv):
        output_csv = os.path.join(repo_path, output_csv)

    fieldnames = [
        "task_id", "category", "difficulty", "question",
        "memory_prompt_tokens", "memory_total_tokens", "memory_latency_s", "memory_correct",
        "baseline_prompt_tokens", "baseline_total_tokens", "baseline_latency_s", "baseline_correct",
        "token_savings_pct"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved detailed results to: {output_csv}")

    # Calculate summary metrics
    n = len(results)
    avg_mem_prompt = sum(r["memory_prompt_tokens"] for r in results) / n
    avg_base_prompt = sum(r["baseline_prompt_tokens"] for r in results) / n
    avg_mem_total = sum(r["memory_total_tokens"] for r in results) / n
    avg_base_total = sum(r["baseline_total_tokens"] for r in results) / n
    avg_mem_lat = sum(r["memory_latency_s"] for r in results) / n
    avg_base_lat = sum(r["baseline_latency_s"] for r in results) / n
    mem_accuracy = (sum(1 for r in results if r["memory_correct"]) / n) * 100.0
    base_accuracy = (sum(1 for r in results if r["baseline_correct"]) / n) * 100.0
    overall_token_savings = ((avg_base_prompt - avg_mem_prompt) / max(1, avg_base_prompt)) * 100.0

    summary = {
        "total_tasks": n,
        "avg_memory_prompt_tokens": round(avg_mem_prompt, 1),
        "avg_baseline_prompt_tokens": round(avg_base_prompt, 1),
        "avg_memory_total_tokens": round(avg_mem_total, 1),
        "avg_baseline_total_tokens": round(avg_base_total, 1),
        "avg_memory_latency_s": round(avg_mem_lat, 3),
        "avg_baseline_latency_s": round(avg_base_lat, 3),
        "memory_accuracy_pct": round(mem_accuracy, 1),
        "baseline_accuracy_pct": round(base_accuracy, 1),
        "overall_token_savings_pct": round(overall_token_savings, 1),
        "csv_path": output_csv
    }

    # Print Summary Markdown Table
    print("\n" + "="*70)
    print("                     BENCHMARK SUMMARY RESULTS")
    print("="*70)
    print(f"| Metric                      | RepoMemory Mode | Naive Baseline  | Impact / Diff     |")
    print(f"| :-------------------------- | :-------------- | :-------------- | :---------------- |")
    print(f"| Avg Prompt Tokens           | {summary['avg_memory_prompt_tokens']:<15} | {summary['avg_baseline_prompt_tokens']:<15} | -{summary['overall_token_savings_pct']:.1f}% reduction   |")
    print(f"| Avg Total Tokens            | {summary['avg_memory_total_tokens']:<15} | {summary['avg_baseline_total_tokens']:<15} | -{((avg_base_total - avg_mem_total)/avg_base_total*100):.1f}% reduction   |")
    print(f"| Avg Latency                 | {summary['avg_memory_latency_s']}s{' '*11} | {summary['avg_baseline_latency_s']}s{' '*11} | {round(avg_mem_lat - avg_base_lat, 3):+}s latency      |")
    print(f"| Retrieval Accuracy          | {summary['memory_accuracy_pct']}%{' '*12} | {summary['baseline_accuracy_pct']}%{' '*12} | {round(mem_accuracy - base_accuracy, 1):+}% accuracy     |")
    print("="*70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run RepoMemory Benchmark")
    parser.add_argument("--repo", default=".", help="Target repository directory")
    parser.add_argument("--tasks", default="tasks.json", help="Path to tasks.json")
    parser.add_argument("--output", default="results.csv", help="Output CSV path")
    parser.add_argument("--reindex", action="store_true", help="Force reindexing repository")
    parser.add_argument("--model", default="qwen2.5-coder:7b-instruct", help="Ollama model")
    args = parser.parse_args()

    run_benchmark(
        repo_path=args.repo,
        tasks_file=args.tasks,
        output_csv=args.output,
        reindex=args.reindex,
        model=args.model
    )


if __name__ == "__main__":
    main()
