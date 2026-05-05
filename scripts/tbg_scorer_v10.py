"""
TBG scorer for TGToM (TemporalGraph-ToM).

================================================================================
WHAT THIS SCRIPT DOES
================================================================================
Scores a model's predicted Temporal Belief Graph (TBG) against the canonical
TBG built from each story by `verify_graph_v10.build_graph`. Reports four
metrics:

  1. Final Node Accuracy
       Per-agent accuracy of the final belief.
       FNA = (1/|A|) * sum_i 1(b_hat_i^(T) == b_i^(T))

  2. Temporal Node Accuracy (at update steps only)
       Per-(agent, t) accuracy of beliefs, restricted to (i, t) where the
       gold belief actually changes from t-1 to t.
       TNA = (1/|U|) * sum_{(i,t) in U} 1(b_hat_i^(t) == b_i^(t))
       where U = {(i, t) : b_i^(t) != b_i^(t-1)}.

  3. Edge F1 (strict and lenient)
       Edge tuple: (source_agent, target_agent, line, relation_type).
       Strict:  full 4-tuple equality.
       Lenient: equality on (source_agent, target_agent, relation_type),
                ignoring line.

  4. Normalized Structural Distance
       NSD = |E_hat △ E| / |E_hat ∪ E|
       Range [0, 1]; lower is better. Equivalent to 1 - Jaccard(E_hat, E).

================================================================================
INPUT
================================================================================
Predictions file (JSONL). Each entry corresponds to one (story_id, model,
trial) prediction:
    {
      "story_id": int,
      "model": str (optional),
      "trial": int (optional),
      "final_beliefs": {agent: location, ...},   # OR null/missing
      "edges": [
        {"source_agent": str, "target_agent": str,
         "line": int, "relation_type": str},
        ...
      ]
    }

Both "final_beliefs" and "edges" are optional; missing fields contribute 0
to the corresponding metric without erroring.

================================================================================
USAGE
================================================================================
    python tbg_scorer_v10.py predictions.jsonl
    python tbg_scorer_v10.py predictions.jsonl --by-model
    python tbg_scorer_v10.py --self-test           # run synthetic tests

================================================================================
"""

import argparse
import json
import os
import sys
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from verify_graph_v10 import parse_events, build_graph  # type: ignore

STORIES_PATH = "/mnt/user-data/outputs/v11/data/stories_v10.jsonl"


# ----------------------------------------------------------------------------
# Canonical TBG construction (gold)
# ----------------------------------------------------------------------------

def gold_tbg_for_story(story):
    """Return (final_beliefs, edges, belief_trajectory) for a story.

    final_beliefs:    dict {agent: location}
    edges:            list of edge dicts (see module docstring)
    belief_trajectory: list of dicts, belief[t] for t in 0..len(events)
                      (used by Temporal Node Accuracy)
    """
    events, exit_step = parse_events(
        story["story"], story["object"], story["agents"]
    )
    belief, edges = build_graph(events, exit_step, story["agents"])
    final_beliefs = belief[-1] if belief else {a: None for a in story["agents"]}
    return final_beliefs, edges, belief


# ----------------------------------------------------------------------------
# Metric implementations
# ----------------------------------------------------------------------------

def final_node_accuracy(gold_final, predicted_final):
    """FNA = (correct agents) / (|agents|)."""
    if not gold_final:
        return None
    if predicted_final is None:
        predicted_final = {}
    correct = sum(1 for agent, loc in gold_final.items()
                  if predicted_final.get(agent) == loc)
    return correct / len(gold_final)


def temporal_node_accuracy(gold_belief, predicted_belief):
    """TNA = correctness on (agent, t) pairs where gold belief changed at t.

    `gold_belief` and `predicted_belief` are lists of dicts, indexed by t.
    If predicted_belief is None or shorter than gold_belief, missing
    entries count as zero correct on the relevant update positions.
    """
    if not gold_belief or len(gold_belief) < 2:
        return None
    update_positions = []  # list of (agent, t)
    for t in range(1, len(gold_belief)):
        for agent, loc in gold_belief[t].items():
            if loc != gold_belief[t - 1].get(agent):
                update_positions.append((agent, t))
    if not update_positions:
        return None  # no updates in this story (degenerate)
    if predicted_belief is None:
        predicted_belief = []
    correct = 0
    for agent, t in update_positions:
        if t < len(predicted_belief):
            if predicted_belief[t].get(agent) == gold_belief[t][agent]:
                correct += 1
    return correct / len(update_positions)


def _edge_to_tuple_strict(edge):
    return (edge.get("source_agent"), edge.get("target_agent"),
            edge.get("line"), edge.get("relation_type"))


def _edge_to_tuple_lenient(edge):
    return (edge.get("source_agent"), edge.get("target_agent"),
            edge.get("relation_type"))


def edge_precision_recall_f1(gold_edges, predicted_edges, mode="strict"):
    """Returns (precision, recall, f1) on the edge set under the given mode."""
    if predicted_edges is None:
        predicted_edges = []
    to_tuple = (_edge_to_tuple_strict if mode == "strict"
                else _edge_to_tuple_lenient)
    gold_set = {to_tuple(e) for e in gold_edges}
    pred_set = {to_tuple(e) for e in predicted_edges}
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0
    if not gold_set:
        return 0.0, 1.0, 0.0  # no positives to recall
    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def normalized_structural_distance(gold_edges, predicted_edges, mode="strict"):
    """NSD = |E_hat △ E| / |E_hat ∪ E|. Range [0, 1], lower is better."""
    if predicted_edges is None:
        predicted_edges = []
    to_tuple = (_edge_to_tuple_strict if mode == "strict"
                else _edge_to_tuple_lenient)
    gold_set = {to_tuple(e) for e in gold_edges}
    pred_set = {to_tuple(e) for e in predicted_edges}
    union = gold_set | pred_set
    if not union:
        return 0.0
    sym_diff = gold_set ^ pred_set
    return len(sym_diff) / len(union)


# ----------------------------------------------------------------------------
# Per-story scoring
# ----------------------------------------------------------------------------

def score_one(story, prediction):
    """Return a dict of metrics for one story+prediction pair."""
    gold_final, gold_edges, gold_belief = gold_tbg_for_story(story)
    pred_final = prediction.get("final_beliefs")
    pred_edges = prediction.get("edges") or []
    pred_belief = prediction.get("belief_trajectory")  # optional

    fna = final_node_accuracy(gold_final, pred_final)
    tna = temporal_node_accuracy(gold_belief, pred_belief)
    p_s, r_s, f1_s = edge_precision_recall_f1(gold_edges, pred_edges, "strict")
    p_l, r_l, f1_l = edge_precision_recall_f1(gold_edges, pred_edges, "lenient")
    nsd = normalized_structural_distance(gold_edges, pred_edges, "strict")
    return {
        "story_id": story["id"],
        "fna": fna,
        "tna": tna,
        "edge_precision_strict": p_s,
        "edge_recall_strict": r_s,
        "edge_f1_strict": f1_s,
        "edge_precision_lenient": p_l,
        "edge_recall_lenient": r_l,
        "edge_f1_lenient": f1_l,
        "nsd": nsd,
    }


def aggregate(rows):
    """Mean of each metric across rows, ignoring None values."""
    keys = ["fna", "tna",
            "edge_precision_strict", "edge_recall_strict", "edge_f1_strict",
            "edge_precision_lenient", "edge_recall_lenient", "edge_f1_lenient",
            "nsd"]
    out = {}
    for k in keys:
        values = [r[k] for r in rows if r.get(k) is not None]
        out[k] = sum(values) / len(values) if values else None
    return out


def print_aggregate(label, agg, n):
    print(f"\n=== {label} (n={n}) ===")
    print(f"  Final Node Accuracy:        "
          f"{_fmt(agg.get('fna'))}")
    print(f"  Temporal Node Accuracy:     "
          f"{_fmt(agg.get('tna'))}")
    print(f"  Edge F1 (strict):           "
          f"{_fmt(agg.get('edge_f1_strict'))}  "
          f"(P={_fmt(agg.get('edge_precision_strict'))}, "
          f"R={_fmt(agg.get('edge_recall_strict'))})")
    print(f"  Edge F1 (lenient):          "
          f"{_fmt(agg.get('edge_f1_lenient'))}  "
          f"(P={_fmt(agg.get('edge_precision_lenient'))}, "
          f"R={_fmt(agg.get('edge_recall_lenient'))})")
    print(f"  Normalized Struct Distance: "
          f"{_fmt(agg.get('nsd'))}")


def _fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


# ----------------------------------------------------------------------------
# Main entry point: scoring a predictions file
# ----------------------------------------------------------------------------

def main_score(predictions_path, by_model=False):
    stories = {entry["id"]: entry
               for entry in (json.loads(l) for l in open(STORIES_PATH))}
    predictions = [json.loads(l) for l in open(predictions_path)]

    rows = []
    rows_by_model = defaultdict(list)
    for prediction in predictions:
        story_id = prediction["story_id"]
        story = stories.get(story_id)
        if story is None:
            continue
        row = score_one(story, prediction)
        row["model"] = prediction.get("model", "unknown")
        rows.append(row)
        rows_by_model[row["model"]].append(row)

    print_aggregate("OVERALL", aggregate(rows), len(rows))
    if by_model:
        for model in sorted(rows_by_model):
            model_rows = rows_by_model[model]
            print_aggregate(model, aggregate(model_rows), len(model_rows))


# ----------------------------------------------------------------------------
# Self-test with synthetic predictions
# ----------------------------------------------------------------------------

def _run_self_test():
    """Run the scorer against synthetic predictions on the first 5 stories.
    Verifies the 4 metrics behave as expected."""
    import random

    stories = [json.loads(l) for l in open(STORIES_PATH)][:5]
    print("Running self-test on first 5 stories...\n")

    # Build gold reference for each story.
    gold_refs = []
    for s in stories:
        final, edges, belief = gold_tbg_for_story(s)
        gold_refs.append((s, final, edges, belief))

    def score_synthetic(label, build_prediction):
        rows = []
        for s, gold_final, gold_edges, gold_belief in gold_refs:
            prediction = build_prediction(s, gold_final, gold_edges, gold_belief)
            row = score_one(s, prediction)
            rows.append(row)
        agg = aggregate(rows)
        print_aggregate(label, agg, len(rows))

    # Case 1: Perfect predictions (copy gold).
    score_synthetic(
        "PERFECT (gold copy)",
        lambda s, gf, ge, gb: {
            "story_id": s["id"],
            "final_beliefs": dict(gf),
            "edges": list(ge),
            "belief_trajectory": [dict(b) for b in gb],
        },
    )

    # Case 2: Empty predictions.
    score_synthetic(
        "EMPTY",
        lambda s, gf, ge, gb: {
            "story_id": s["id"],
            "final_beliefs": {},
            "edges": [],
        },
    )

    # Case 3: 50% subset of edges, perfect final beliefs.
    rng = random.Random(123)
    score_synthetic(
        "50% edges kept, perfect beliefs",
        lambda s, gf, ge, gb: {
            "story_id": s["id"],
            "final_beliefs": dict(gf),
            "edges": rng.sample(ge, k=len(ge) // 2) if ge else [],
        },
    )

    # Case 4: All gold edges with line numbers shifted by 1.
    score_synthetic(
        "All edges, lines shifted by +1 (lenient should still match)",
        lambda s, gf, ge, gb: {
            "story_id": s["id"],
            "final_beliefs": dict(gf),
            "edges": [{**e, "line": e["line"] + 1} for e in ge],
        },
    )

    # Case 5: Random fake edges (wrong relation_type for each).
    def corrupt_type(e):
        types = ["trusted", "untrusted", "cooperative", "deceptive"]
        wrong = [t for t in types if t != e["relation_type"]]
        return {**e, "relation_type": rng.choice(wrong)}

    score_synthetic(
        "All edges, relation_type corrupted",
        lambda s, gf, ge, gb: {
            "story_id": s["id"],
            "final_beliefs": dict(gf),
            "edges": [corrupt_type(e) for e in ge],
        },
    )

    print("\nSelf-test complete.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("predictions", nargs="?",
                        help="Path to predictions JSONL")
    parser.add_argument("--by-model", action="store_true",
                        help="Break down accuracy by model")
    parser.add_argument("--self-test", action="store_true",
                        help="Run synthetic test cases")
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        return

    if not args.predictions:
        parser.error("Either --self-test or a predictions file is required")
    if not os.path.exists(args.predictions):
        sys.exit(f"Predictions file not found: {args.predictions}")
    main_score(args.predictions, by_model=args.by_model)


if __name__ == "__main__":
    main()
