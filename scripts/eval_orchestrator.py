"""
eval_orchestrator.py — TGToM evaluation orchestrator.

================================================================================
SCRIPT DESCRIPTION
================================================================================
Orchestrates evaluation for the TemporalGraph-ToM benchmark. For
each (story, question, scaffold, model, trial), this script builds a prompt,
submits it to the LLM API (via OpenRouter for unified access across providers),
parses the response, and writes a predictions JSONL.

It does not implement scoring or ground truth derivation — those live in `score_v10.py` 
and `tbg_scorer_v10.py`, which consume the predictions file produced here.

Pipeline:
    Stage 1 (existing):  ground truth from gen_v10.py + question scripts
    Stage 2 (this):      predictions from LLM APIs
    Stage 3 (existing):  scoring with score_v10.py and tbg_scorer_v10.py

================================================================================
INPUT
================================================================================
Reads from /mnt/user-data/outputs/v11/data/:
    stories_v10.jsonl
    higher_order_beliefs_v10.jsonl       (Q0, Q1, Q2, Q3)
    counterfactual_beliefs_v10.jsonl     (Q5, Q6, Q7, Q10)
    causal_beliefs_v10.jsonl             (Q8)
    common_knowledge_v10.jsonl           (Q9, Q11, Q13)

Requires environment variable OPENROUTER_API_KEY.

================================================================================
OUTPUT
================================================================================
predictions.jsonl. Each entry:
    {
      "story_id": int,
      "question_id": str,            # Q0, Q1, Q2, ..., Q13
      "scaffold": str,               # cot | system2 | tbg
      "model": str,                  # OpenRouter model string
      "trial": int,
      "prompt": str,                 # full prompt sent
      "raw_response": str,           # raw LLM response
      "parsed": {                    # extracted prediction (best-effort)
        "answer": str | dict,
        "edges": [...],              # only for TBG
        "final_beliefs": {...},      # only for TBG
        "belief_trajectory": [...],  # only for TBG (optional)
      }
    }

================================================================================
USAGE
================================================================================
    export OPENROUTER_API_KEY=...
    python eval_orchestrator.py --stories 5 --pilot       # pilot on 5 stories
    python eval_orchestrator.py                           # full eval (100 stories)
    python eval_orchestrator.py --models gpt-4o-mini      # one model only
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Install the openai package: pip install openai")


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DATA_DIR = "/mnt/user-data/outputs/v11/data"

GROUND_TRUTH_FILES = {
    "higher": f"{DATA_DIR}/higher_order_beliefs_v10.jsonl",
    "counterfactual": f"{DATA_DIR}/counterfactual_beliefs_v10.jsonl",
    "causal": f"{DATA_DIR}/causal_beliefs_v10.jsonl",
    "knowledge": f"{DATA_DIR}/common_knowledge_v10.jsonl",
}

# Map each question id to ground-truth file.
QUESTION_FILE = {
    "Q0":  "higher",  "Q1":  "higher",  "Q2":  "higher",  "Q3":  "higher",
    "Q5":  "counterfactual", "Q6":  "counterfactual",
    "Q7":  "counterfactual", "Q10": "counterfactual",
    "Q8":  "causal",
    "Q9":  "knowledge", "Q11": "knowledge", "Q13": "knowledge",
}

# OpenRouter model strings. Update as needed.
# Starting with one closed-API model (GPT-4o-mini); add open-weight models
# (e.g. openai/gpt-oss-120b, openai/gpt-oss-20b) after initial run.
DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
]

DEFAULT_SCAFFOLDS = ["cot", "system2", "tbg"]
DEFAULT_TRIALS = 3
DEFAULT_N_STORIES = 100
MAX_PARALLEL = 20      # concurrent API calls
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0    # seconds, exponential


# ----------------------------------------------------------------------------
# Scaffold templates
# ----------------------------------------------------------------------------
# Three reasoning scaffolds tested in TGToM evaluation:
#   COT      — baseline chain-of-thought
#   SYSTEM2  — deliberate/slow reasoning
#   TBG      — temporal belief graph (this paper's contribution)
# All three use {story} and {question} placeholders, filled by str.format().
# ----------------------------------------------------------------------------

COT_TEMPLATE = """\
Use chain-of-thought reasoning. Think step by step to answer the question.

Story:
{story}

Question: {question}

Provide your reasoning, then your final answer on a line beginning with "Answer:".
"""

SYSTEM2_TEMPLATE = """\
Engage System 2 reasoning: slow, deliberate, and effortful. Consider the \
question carefully and reason thoroughly before answering.

Story:
{story}

Question: {question}

Provide your reasoning, then your final answer on a line beginning with "Answer:".
"""

TBG_TEMPLATE = """\
Represent your reasoning as a temporal belief graph. Nodes represent each \
agent's state (belief, observation, intent) at each time step. Beliefs update \
through direct observation and propagate through two layers: communication \
and intent. Track how each agent's belief state updates after every event, \
then use the resulting graph to answer the question.

Story:
{story}

Question: {question}

Output JSON in exactly this format (no commentary outside the JSON):
{{
  "edges": [
    {{"source_agent": "<name>", "target_agent": "<name>", "line": <int>, "relation_type": "<trusted|untrusted|cooperative|deceptive>"}}
  ],
  "final_beliefs": {{"<agent>": "<location>"}},
  "belief_trajectory": [
    {{"<agent>": "<location_or_null>"}}
  ],
  "answer": "<your final answer>"
}}
"""

SCAFFOLDS = {
    "cot": COT_TEMPLATE,
    "system2": SYSTEM2_TEMPLATE,
    "tbg": TBG_TEMPLATE,
}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def load_all_data():
    """Load stories and ground-truth files; return one dict per story."""
    stories = load_jsonl(f"{DATA_DIR}/stories_v10.jsonl")
    ground_truth = {key: load_jsonl(path)
                    for key, path in GROUND_TRUTH_FILES.items()}
    return stories, ground_truth


def questions_for_story(story_id, ground_truth):
    """Return a list of {question_id, question_text} for one story."""
    out = []
    for question_id, gt_key in QUESTION_FILE.items():
        entry = ground_truth[gt_key][story_id]
        if question_id == "Q1":
            # Q1 has multiple sub-questions, one per agent.
            for sub_idx, sub in enumerate(entry["Q1"]["questions"]):
                out.append({
                    "question_id": f"Q1_{sub_idx}",
                    "question_text": sub["question"],
                })
        else:
            block = entry.get(question_id, {})
            text = block.get("question")
            if text is not None:
                out.append({
                    "question_id": question_id,
                    "question_text": text,
                })
    return out


# ----------------------------------------------------------------------------
# API client
# ----------------------------------------------------------------------------

def make_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY environment variable.")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


class FatalAPIError(Exception):
    """Non-retryable error (billing, auth, invalid request). Aborts the run."""


def _is_fatal(exc):
    """Return True if the exception indicates a non-retryable failure."""
    msg = str(exc).lower()
    if any(code in msg for code in ("401", "402", "403", "429")):
        # 401 = auth failure, 402 = out of credits, 403 = forbidden,
        # 429 = rate limit (treat as fatal to avoid retry storm)
        return True
    if any(word in msg for word in ("insufficient", "quota", "billing",
                                     "credit", "unauthorized")):
        return True
    return False


def call_api(client, model, prompt):
    """Submit one prompt to the API. Retry on transient errors. Return raw text."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4000,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            if _is_fatal(exc):
                raise FatalAPIError(f"Fatal API error (will not retry): {exc}")
            last_err = exc
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    raise RuntimeError(f"API call failed after {MAX_RETRIES} retries: {last_err}")


# ----------------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------------

ANSWER_LINE = re.compile(r"^\s*Answer:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_cot_or_system2(raw):
    """Extract 'Answer: <x>' line. Returns dict with 'answer' field."""
    matches = ANSWER_LINE.findall(raw)
    answer = matches[-1].strip() if matches else None
    return {"answer": answer}


def parse_tbg(raw):
    """Extract JSON object from raw response. Best-effort: find first {...} block."""
    start = raw.find("{")
    if start == -1:
        return {"answer": None, "edges": [], "final_beliefs": {},
                "parse_error": "no JSON object found"}
    # Greedy: take from first { to last }, then try to parse.
    end = raw.rfind("}")
    if end == -1 or end <= start:
        return {"answer": None, "edges": [], "final_beliefs": {},
                "parse_error": "unmatched braces"}
    candidate = raw[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return {"answer": None, "edges": [], "final_beliefs": {},
                "parse_error": f"invalid JSON: {exc}"}
    return {
        "answer": parsed.get("answer"),
        "edges": parsed.get("edges", []),
        "final_beliefs": parsed.get("final_beliefs", {}),
        "belief_trajectory": parsed.get("belief_trajectory"),
    }


def parse_response(scaffold, raw):
    if scaffold == "tbg":
        return parse_tbg(raw)
    return parse_cot_or_system2(raw)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def build_jobs(stories, ground_truth, n_stories, models, scaffolds, trials):
    """Yield one job per (story, question, scaffold, model, trial)."""
    for story in stories[:n_stories]:
        questions = questions_for_story(story["id"], ground_truth)
        for q in questions:
            for scaffold in scaffolds:
                template = SCAFFOLDS[scaffold]
                prompt = template.format(
                    story=story["story"],
                    question=q["question_text"],
                )
                for model in models:
                    for trial in range(trials):
                        yield {
                            "story_id": story["id"],
                            "question_id": q["question_id"],
                            "scaffold": scaffold,
                            "model": model,
                            "trial": trial,
                            "prompt": prompt,
                        }


def run_job(client, job):
    """Submit one job; return job dict with raw_response and parsed prediction."""
    raw = call_api(client, job["model"], job["prompt"])
    job["raw_response"] = raw
    job["parsed"] = parse_response(job["scaffold"], raw)
    return job


def run_all(client, jobs, output_path, max_parallel):
    """Run all jobs in parallel; stream results to output JSONL.

    Aborts immediately on FatalAPIError (billing, auth, rate-limit) to prevent
    wasteful retry storms when credits run out or API key is invalid.
    """
    n = len(jobs)
    print(f"Submitting {n} jobs with {max_parallel}-way parallelism.")
    completed = 0
    failed = 0
    fatal = None
    with open(output_path, "w") as out:
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {pool.submit(run_job, client, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    out.write(json.dumps(result) + "\n")
                    out.flush()
                    completed += 1
                except FatalAPIError as exc:
                    fatal = exc
                    job["error"] = str(exc)
                    out.write(json.dumps(job) + "\n")
                    out.flush()
                    failed += 1
                    # Cancel any pending futures; let in-flight ones finish
                    # naturally (they'll either complete or hit the same fatal error).
                    for f in futures:
                        f.cancel()
                    break
                except Exception as exc:
                    job["error"] = str(exc)
                    out.write(json.dumps(job) + "\n")
                    out.flush()
                    failed += 1
                if (completed + failed) % 50 == 0:
                    print(f"  {completed + failed}/{n}  "
                          f"({failed} failed)")
    if fatal:
        print(f"ABORTED after fatal error: {fatal}")
        print(f"  {completed} succeeded, {failed} failed before abort.")
        print(f"  Partial results written to {output_path}.")
        sys.exit(1)
    print(f"Done. {completed} succeeded, {failed} failed. "
          f"Wrote {output_path}.")


# ----------------------------------------------------------------------------
# TBG predictions: post-process to format expected by tbg_scorer_v10.py
# ----------------------------------------------------------------------------

def write_tbg_predictions(predictions_path, tbg_output_path):
    """Read the main predictions file and write a TBG-only predictions file
    in the format expected by `tbg_scorer_v10.py`.

    The TBG graph reconstruction is independent of which question was asked,
    so for each (story_id, model, trial) we take the TBG row from the first
    question encountered. One entry per (story_id, model, trial) is written.
    """
    rows = [json.loads(line) for line in open(predictions_path)]
    tbg_rows = [r for r in rows if r.get("scaffold") == "tbg"]

    seen = set()  # (story_id, model, trial) keys already written
    written = 0
    with open(tbg_output_path, "w") as out:
        for row in tbg_rows:
            key = (row["story_id"], row["model"], row["trial"])
            if key in seen:
                continue
            seen.add(key)
            parsed = row.get("parsed") or {}
            entry = {
                "story_id": row["story_id"],
                "model": row["model"],
                "trial": row["trial"],
                "edges": parsed.get("edges", []),
                "final_beliefs": parsed.get("final_beliefs", {}),
                "belief_trajectory": parsed.get("belief_trajectory"),
            }
            out.write(json.dumps(entry) + "\n")
            written += 1
    print(f"Wrote {written} TBG entries to {tbg_output_path}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stories", type=int, default=DEFAULT_N_STORIES,
                        help="Number of stories to evaluate (default: 100)")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot mode: small sample for prompt validation")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Model strings (OpenRouter format)")
    parser.add_argument("--scaffolds", nargs="+", default=DEFAULT_SCAFFOLDS,
                        choices=list(SCAFFOLDS.keys()))
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--output", default="predictions.jsonl")
    parser.add_argument("--parallel", type=int, default=MAX_PARALLEL)
    args = parser.parse_args()

    if args.pilot:
        args.stories = 5
        args.trials = 1
        args.parallel = 5
        print("Pilot mode: 5 stories, 1 trial.")

    client = make_client()
    stories, ground_truth = load_all_data()
    jobs = list(build_jobs(stories, ground_truth, args.stories,
                           args.models, args.scaffolds, args.trials))
    run_all(client, jobs, args.output, args.parallel)

    # Auto-write TBG predictions in the format tbg_scorer_v10.py expects.
    if "tbg" in args.scaffolds:
        tbg_path = args.output.replace(".jsonl", "_tbg.jsonl")
        if tbg_path == args.output:
            tbg_path = args.output + ".tbg.jsonl"
        write_tbg_predictions(args.output, tbg_path)


if __name__ == "__main__":
    main()
