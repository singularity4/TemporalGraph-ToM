"""
Scorer — grade model predictions against TGToM ground truth.

================================================================================
SCRIPT DESCRIPTION
================================================================================
Reads ground truth (the per-question JSONL files) and a predictions file,
grades each prediction, and produces an accuracy report.

For each (story_id, question), the prediction is correct iff it matches the
stored ground-truth answer per the question's matching rule:

  Q0       — exact match on string (location)
  Q1       — exact match per agent (one (agent, location) item per agent)
  Q2       — exact match on string
  Q3       — exact match on dict {help_target, hide_target} (or None)
  Q5       — exact match on string (or None)
  Q6       — exact match on string (or None)
  Q7       — exact match on dict {help_target, hide_target} (or None)
  Q8       — exact match on `answer.line` (the line number of the cause)
  Q9       — exact match on dict {common_knowledge, shared_location}
  Q10      — exact match on string (or None)
  Q11      — exact match on dict {common_knowledge, shared_location}
  Q13      — exact match on dict {common_knowledge, shared_location}

Trial averaging: if the predictions file has multiple trials per
(story, question), the scorer reports both per-trial accuracy and
mean-over-trials accuracy.

================================================================================
INPUT
================================================================================
- TGToM ground truth files:
    /mnt/user-data/outputs/v10/higher_order_beliefs_v10.jsonl
    /mnt/user-data/outputs/v10/counterfactual_beliefs_v10.jsonl
    /mnt/user-data/outputs/v10/causal_beliefs_v10.jsonl
    /mnt/user-data/outputs/v10/common_knowledge_v10.jsonl

- Predictions file (passed as argument): JSONL with entries of shape
    {
      "story_id": int,
      "question_id": "Q0" | "Q1" | "Q2" | ... | "Q13",
      "agent": str | null,        # required for Q1 (one prediction per agent)
      "trial": int,               # 0..n_trials-1 (0 if single trial)
      "prediction": ...           # type depends on question_id
    }

  The prediction's type per question:
    Q0, Q2, Q5, Q6, Q10  -> string (a container name) or null
    Q1                   -> string (a container name) or null,
                            with `agent` field set
    Q3, Q7               -> dict {help_target, hide_target}, each str|null
    Q8                   -> int (line number)
    Q9, Q11, Q13         -> dict {common_knowledge: bool, shared_location: str|null}

================================================================================
OUTPUT
================================================================================
Prints a per-question accuracy table and a per-(question, trial-count)
summary. Also writes scored_<input_basename>.jsonl with per-prediction
correctness flags.

================================================================================
USAGE
================================================================================
    python score_v10.py path/to/predictions.jsonl
    python score_v10.py path/to/predictions.jsonl --trials 5
    python score_v10.py path/to/predictions.jsonl --by-targeted
    python score_v10.py path/to/predictions.jsonl --by-chain-depth

The --trials flag specifies the expected number of trials per (story,
question). Common values: 3, 4, or 5. Trials are averaged per (story,
question); accuracy is the mean of per-trial 0/1 scores.
"""

import argparse
import json
import os
import sys
from collections import defaultdict


GROUND_TRUTH_PATHS = {
    "higher": "/mnt/user-data/outputs/v10/higher_order_beliefs_v10.jsonl",
    "counterfactual": "/mnt/user-data/outputs/v10/counterfactual_beliefs_v10.jsonl",
    "causal": "/mnt/user-data/outputs/v10/causal_beliefs_v10.jsonl",
    "knowledge": "/mnt/user-data/outputs/v10/common_knowledge_v10.jsonl",
}

# Maps question_id -> (file_key, dict_key_in_entry).
QUESTION_LOCATION = {
    "Q0":  ("higher", "Q0"),
    "Q1":  ("higher", "Q1"),    # has questions: list of {agent, ...}
    "Q2":  ("higher", "Q2"),
    "Q3":  ("higher", "Q3"),
    "Q5":  ("counterfactual", "Q5"),
    "Q6":  ("counterfactual", "Q6"),
    "Q7":  ("counterfactual", "Q7"),
    "Q8":  ("causal", "Q8"),
    "Q9":  ("knowledge", "Q9"),
    "Q10": ("counterfactual", "Q10"),
    "Q11": ("knowledge", "Q11"),
    "Q13": ("knowledge", "Q13"),
}

ALL_QUESTIONS = list(QUESTION_LOCATION.keys())


# ---------------------------------------------------------------------------
# Loading ground truth
# ---------------------------------------------------------------------------

def load_ground_truth():
    """Return a nested dict {story_id: {question_id: ground_truth_value}}.
    For Q1, the value is itself a dict {agent: ground_truth}."""
    gt = defaultdict(dict)
    files = {key: [json.loads(l) for l in open(path)]
             for key, path in GROUND_TRUTH_PATHS.items()}

    for question_id, (file_key, dict_key) in QUESTION_LOCATION.items():
        for entry in files[file_key]:
            story_id = entry["id"]
            sub = entry[dict_key]
            if question_id == "Q1":
                gt[story_id]["Q1"] = {q["agent"]: q["answer"] for q in sub["questions"]}
            else:
                gt[story_id][question_id] = sub["answer"]
    return dict(gt)


# ---------------------------------------------------------------------------
# Per-question grading
# ---------------------------------------------------------------------------

def grade_one(question_id, prediction, ground_truth, agent=None):
    """Return 1 if prediction matches ground_truth, 0 otherwise."""
    if question_id == "Q1":
        # Ground truth is dict {agent: answer}. Caller must pass `agent`.
        if agent is None:
            return 0
        gt = ground_truth.get(agent)
        return int(prediction == gt)

    if question_id in ("Q0", "Q2", "Q5", "Q6", "Q10"):
        return int(prediction == ground_truth)

    if question_id in ("Q3", "Q7"):
        # Dict comparison; both can be None.
        if ground_truth is None:
            return int(prediction is None)
        if not isinstance(prediction, dict):
            return 0
        return int(
            prediction.get("help_target") == ground_truth.get("help_target")
            and prediction.get("hide_target") == ground_truth.get("hide_target")
        )

    if question_id == "Q8":
        # Ground truth is dict {line, kind, description, resulting_belief}.
        # Score on the line number (the canonical answer).
        if ground_truth is None:
            return int(prediction is None)
        gt_line = ground_truth.get("line")
        # Accept either int or dict prediction (be lenient on format).
        if isinstance(prediction, dict):
            return int(prediction.get("line") == gt_line)
        return int(prediction == gt_line)

    if question_id in ("Q9", "Q11", "Q13"):
        # Dict {common_knowledge: bool, shared_location: str|null}.
        if ground_truth is None:
            return int(prediction is None)
        if not isinstance(prediction, dict):
            return 0
        gt_ck = ground_truth.get("common_knowledge")
        gt_loc = ground_truth.get("shared_location")
        return int(
            prediction.get("common_knowledge") == gt_ck
            and prediction.get("shared_location") == gt_loc
        )

    raise ValueError(f"Unknown question_id: {question_id}")


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_predictions(predictions_path, expected_trials=None,
                       group_by_targeted=False, group_by_chain_depth=False):
    """Read predictions, grade them, return a structured results dict."""
    gt = load_ground_truth()
    predictions = [json.loads(l) for l in open(predictions_path)]

    # Bucket: per_q_per_trial[(question_id, trial)] = list of 0/1
    # Also: per_q[question_id] = list of (story_id, agent, trial, score)
    per_q = defaultdict(list)

    # For grouping (targeted / chain depth), we also need per-story metadata.
    counterfactual_meta = {entry["id"]: entry for entry in
                           [json.loads(l) for l in open(GROUND_TRUTH_PATHS["counterfactual"])]}
    knowledge_meta = {entry["id"]: entry for entry in
                      [json.loads(l) for l in open(GROUND_TRUTH_PATHS["knowledge"])]}
    higher_meta = {entry["id"]: entry for entry in
                   [json.loads(l) for l in open(GROUND_TRUTH_PATHS["higher"])]}

    scored = []
    for prediction_entry in predictions:
        story_id = prediction_entry["story_id"]
        question_id = prediction_entry["question_id"]
        trial = prediction_entry.get("trial", 0)
        agent = prediction_entry.get("agent")

        if question_id == "Q1":
            ground_truth = gt[story_id].get("Q1", {})
        else:
            ground_truth = gt[story_id].get(question_id)

        score = grade_one(question_id, prediction_entry["prediction"], ground_truth, agent=agent)
        scored.append({**prediction_entry, "ground_truth": ground_truth, "correct": score})
        per_q[question_id].append((story_id, agent, trial, score))

    # Compute per-question accuracy.
    print(f"\nScored {len(predictions)} predictions from {predictions_path}\n")
    print(f"{'Question':<6} {'N':>5} {'Acc':>8}")
    print("-" * 22)
    overall_correct = 0
    overall_total = 0
    for question_id in ALL_QUESTIONS:
        records = per_q[question_id]
        if not records:
            print(f"{question_id:<6} {'-':>5} {'-':>8}")
            continue
        n = len(records)
        correct = sum(score for _, _, _, score in records)
        overall_correct += correct
        overall_total += n
        accuracy = correct / n
        print(f"{question_id:<6} {n:>5} {accuracy:>7.1%}")
    print("-" * 22)
    if overall_total > 0:
        print(f"{'TOTAL':<6} {overall_total:>5} {overall_correct/overall_total:>7.1%}")

    # Per-trial breakdown if expected_trials specified.
    if expected_trials is not None and expected_trials > 1:
        print(f"\nPer-trial accuracy (n_trials = {expected_trials}):\n")
        print(f"{'Question':<6} ", end="")
        for t in range(expected_trials):
            print(f"{'T'+str(t):>8}", end="")
        print()
        for question_id in ALL_QUESTIONS:
            records = per_q[question_id]
            if not records:
                continue
            print(f"{question_id:<6} ", end="")
            for t in range(expected_trials):
                trial_records = [r for r in records if r[2] == t]
                if trial_records:
                    accuracy = sum(score for _, _, _, score in trial_records) / len(trial_records)
                    print(f"{accuracy:>7.1%}", end=" ")
                else:
                    print(f"{'-':>8}", end="")
            print()

    # Targeted-only breakdown for Q5/Q6/Q7/Q10/Q11/Q13.
    if group_by_targeted:
        print("\nAccuracy on targeted vs non-targeted:")
        print(f"{'Question':<6} {'Targeted':>10} {'Random':>10}")
        for question_id in ("Q5", "Q6", "Q7", "Q10", "Q11", "Q13"):
            records = per_q[question_id]
            if not records:
                continue
            targeted = []
            non_targeted = []
            for story_id, agent, trial, score in records:
                meta = (counterfactual_meta.get(story_id) if question_id in ("Q5","Q6","Q7","Q10")
                        else knowledge_meta.get(story_id))
                if meta is None:
                    continue
                is_targeted = meta[question_id].get("targeted", False)
                (targeted if is_targeted else non_targeted).append(score)
            t_acc = (sum(targeted)/len(targeted)) if targeted else None
            nt_acc = (sum(non_targeted)/len(non_targeted)) if non_targeted else None
            t_str = f"{t_acc:>8.1%} ({len(targeted)})" if t_acc is not None else f"{'-':>10}"
            nt_str = f"{nt_acc:>8.1%} ({len(non_targeted)})" if nt_acc is not None else f"{'-':>10}"
            print(f"{question_id:<6} {t_str:>14} {nt_str:>14}")

    # Chain-depth breakdown for Q2/Q3/Q6/Q7/Q10.
    if group_by_chain_depth:
        print("\nAccuracy by chain depth:")
        for question_id in ("Q2", "Q3", "Q6", "Q7", "Q10"):
            records = per_q[question_id]
            if not records:
                continue
            buckets = defaultdict(list)
            for story_id, agent, trial, score in records:
                if question_id in ("Q2", "Q3"):
                    meta = higher_meta.get(story_id, {}).get(question_id, {})
                    chain = meta.get("chain") or []
                    depth = len(chain)
                elif question_id == "Q10":
                    meta = counterfactual_meta.get(story_id, {}).get(question_id, {})
                    depth = meta.get("chain_depth", len(meta.get("chain") or []))
                else:
                    meta = counterfactual_meta.get(story_id, {}).get(question_id, {})
                    depth = meta.get("chain_depth", len(meta.get("chain") or []))
                buckets[depth].append(score)
            print(f"  {question_id}: " + ", ".join(
                f"depth={d}: {sum(b)/len(b):.1%} (n={len(b)})"
                for d, b in sorted(buckets.items())
            ))

    # Write scored file.
    out_path = "scored_" + os.path.basename(predictions_path)
    with open(out_path, "w") as f:
        for entry in scored:
            f.write(json.dumps(entry) + "\n")
    print(f"\nPer-prediction scores written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", help="Path to predictions.jsonl")
    parser.add_argument("--trials", type=int, default=None,
                        help="Expected number of trials per (story, question). "
                             "Common values: 3, 4, 5.")
    parser.add_argument("--by-targeted", action="store_true",
                        help="Break down counterfactual question accuracy by targeted vs random.")
    parser.add_argument("--by-chain-depth", action="store_true",
                        help="Break down higher-order question accuracy by chain depth.")
    args = parser.parse_args()

    if not os.path.exists(args.predictions):
        sys.exit(f"Predictions file not found: {args.predictions}")

    score_predictions(args.predictions,
                       expected_trials=args.trials,
                       group_by_targeted=args.by_targeted,
                       group_by_chain_depth=args.by_chain_depth)


if __name__ == "__main__":
    main()
