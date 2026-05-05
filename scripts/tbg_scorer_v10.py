"""
TBG scorer for TGToM (TemporalGraph-ToM).

================================================================================
SCRIPT DESCRIPTION
================================================================================
Scores a model's predicted Temporal Belief Graph (TBG) against the ground truth
TBG built from each story by `verify_graph_v10.build_graph`. Reports four
metrics over the predicted graph versus the ground truth.

  1. Final Node Accuracy (FNA)
       Accuracy of the final belief state for each agent in A.
       FNA = (1 / |A|) * sum_{i in A} 1{b_hat_i^(T) == b_i^(T)}

  2. Time-Respecting Node Accuracy (TNA)
       Accuracy across the full belief trajectory: for each (agent, t) cell
       with t in {1, ..., T}, the predicted belief is compared to the gold
       belief.
       TNA = (1 / (|A| * T)) * sum_{i in A} sum_{t=1}^{T} 1{b_hat_i^(t) == b_i^(t)}

  3. Edge F1 (Temporal and Static)
       Each edge is a tuple (source_agent, target_agent, line, relation_type),
       with relation_type in {trusted, untrusted, cooperative, deceptive}.
       Temporal F1 requires equality on the full 4-tuple.
       Static F1 requires equality on (source_agent, target_agent,
       relation_type) only, ignoring line. The gap between Temporal and
       Static F1 separates timing errors from topological errors.

  4. Normalized Structural Distance (NSD)
       NSD = |E_hat △ E| / |E_hat ∪ E| = 1 - Jaccard(E_hat, E)
       Range [0, 1], where lower indicates higher agreement. Reported as a
       per-story average. The Exact TBG Reconstruction Rate, defined as
       the fraction of stories with NSD = 0, is reported separately.

================================================================================
INPUT
================================================================================
Predictions file (JSONL). Each entry corresponds to one (story_id, model,
trial) prediction:
    {
      "story_id": int,
      "model": str (optional),
      "trial": int (optional),
      "final_beliefs": {agent: location, ...},
      "edges": [
        {"source_agent": str, "target_agent": str,
         "line": int, "relation_type": str},
        ...
      ],
      "belief_trajectory": [{agent: location, ...}, ...]    (optional)
    }

The fields `final_beliefs`, `edges`, and `belief_trajectory` are optional.
Missing fields contribute zero to the corresponding metric without error.
For TNA, the prediction must include `belief_trajectory` covering t = 0
through T; otherwise TNA cannot be evaluated for that prediction.

================================================================================
USAGE
================================================================================
    python tbg_scorer_v10.py predictions.jsonl
    python tbg_scorer_v10.py predictions.jsonl --by-model
    python tbg_scorer_v10.py --self-test           # run synthetic test cases
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
# Canonical TBG construction (ground truth)
# ----------------------------------------------------------------------------

def gold_tbg_for_story(story):
    """Return the ground truth TBG for one story as a tuple
    (final_beliefs, edges, belief_trajectory).

    final_beliefs:     dict mapping each agent to its final belief.
    edges:             list of edge dicts (see SCRIPT DESCRIPTION).
    belief_trajectory: list of dicts; entry t holds the belief of each
                       agent after applying event t. Index 0 is the empty
                       initial state.
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
    """Fraction of agents whose final belief matches the gold final belief."""
    if not gold_final:
        return None
    if predicted_final is None:
        predicted_final = {}
    correct = sum(1 for agent, location in gold_final.items()
                  if predicted_final.get(agent) == location)
    return correct / len(gold_final)


def time_respecting_node_accuracy(gold_belief, predicted_belief):
    """Accuracy across the full belief trajectory.

    For each agent in A and each timestep t in {1, ..., T}, the predicted
    belief is compared to the gold belief. The denominator is |A| * T (the
    full trajectory grid). Returns None if the trajectory has no timesteps
    after t = 0 or if there are no agents.
    """
    if not gold_belief or len(gold_belief) < 2:
        return None
    agents = list(gold_belief[0].keys())
    if not agents:
        return None
    if predicted_belief is None:
        predicted_belief = []
    correct = 0
    total = 0
    for t in range(1, len(gold_belief)):
        for agent in agents:
            total += 1
            if t < len(predicted_belief):
                if predicted_belief[t].get(agent) == gold_belief[t][agent]:
                    correct += 1
    return correct / total if total > 0 else None


def _edge_to_tuple_temporal(edge):
    """4-tuple representation of an edge for Temporal F1."""
    return (edge.get("source_agent"), edge.get("target_agent"),
            edge.get("line"), edge.get("relation_type"))


def _edge_to_tuple_static(edge):
    """3-tuple representation of an edge for Static F1 (ignores time)."""
    return (edge.get("source_agent"), edge.get("target_agent"),
            edge.get("relation_type"))


def edge_precision_recall_f1(gold_edges, predicted_edges, mode="temporal"):
    """Return (precision, recall, F1) over the edge set in the given mode.

    mode = "temporal" enforces full 4-tuple equality.
    mode = "static" enforces equality on (source_agent, target_agent,
    relation_type) only, ignoring line.
    """
    if predicted_edges is None:
        predicted_edges = []
    to_tuple = (_edge_to_tuple_temporal if mode == "temporal"
                else _edge_to_tuple_static)
    gold_set = {to_tuple(e) for e in gold_edges}
    predicted_set = {to_tuple(e) for e in predicted_edges}
    if not gold_set and not predicted_set:
        return 1.0, 1.0, 1.0
    if not predicted_set:
        return 0.0, 0.0, 0.0
    if not gold_set:
        return 0.0, 1.0, 0.0  # no positives in gold, recall undefined
    true_positives = len(gold_set & predicted_set)
    precision = true_positives / len(predicted_set)
    recall = true_positives / len(gold_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def normalized_structural_distance(gold_edges, predicted_edges, mode="temporal"):
    """Symmetric distance between predicted and gold edge sets, normalized
    by their union. Equivalent to 1 - Jaccard(E_hat, E). Lower is better.
    """
    if predicted_edges is None:
        predicted_edges = []
    to_tuple = (_edge_to_tuple_temporal if mode == "temporal"
                else _edge_to_tuple_static)
    gold_set = {to_tuple(e) for e in gold_edges}
    predicted_set = {to_tuple(e) for e in predicted_edges}
    union = gold_set | predicted_set
    if not union:
        return 0.0
    symmetric_difference = gold_set ^ predicted_set
    return len(symmetric_difference) / len(union)


# ----------------------------------------------------------------------------
# Per-story scoring
# ----------------------------------------------------------------------------

def score_one(story, prediction):
    """Compute all metrics for one (story, prediction) pair."""
    gold_final, gold_edges, gold_belief = gold_tbg_for_story(story)
    predicted_final = prediction.get("final_beliefs")
    predicted_edges = prediction.get("edges") or []
    predicted_belief = prediction.get("belief_trajectory")

    fna = final_node_accuracy(gold_final, predicted_final)
    tna = time_respecting_node_accuracy(gold_belief, predicted_belief)
    p_t, r_t, f1_t = edge_precision_recall_f1(
        gold_edges, predicted_edges, mode="temporal"
    )
    p_s, r_s, f1_s = edge_precision_recall_f1(
        gold_edges, predicted_edges, mode="static"
    )
    nsd = normalized_structural_distance(
        gold_edges, predicted_edges, mode="temporal"
    )
    return {
        "story_id": story["id"],
        "fna": fna,
        "tna": tna,
        "edge_precision_temporal": p_t,
        "edge_recall_temporal": r_t,
        "edge_f1_temporal": f1_t,
        "edge_precision_static": p_s,
        "edge_recall_static": r_s,
        "edge_f1_static": f1_s,
        "nsd": nsd,
    }


def aggregate(rows):
    """Mean of each metric across rows, ignoring None values, plus the
    Exact TBG Reconstruction Rate (fraction of stories with NSD = 0).
    """
    keys = ["fna", "tna",
            "edge_precision_temporal", "edge_recall_temporal", "edge_f1_temporal",
            "edge_precision_static", "edge_recall_static", "edge_f1_static",
            "nsd"]
    out = {}
    for k in keys:
        values = [r[k] for r in rows if r.get(k) is not None]
        out[k] = sum(values) / len(values) if values else None

    nsd_values = [r["nsd"] for r in rows if r.get("nsd") is not None]
    if nsd_values:
        out["exact_tbg_reconstruction_rate"] = (
            sum(1 for v in nsd_values if v == 0.0) / len(nsd_values)
        )
    else:
        out["exact_tbg_reconstruction_rate"] = None
    return out


def print_aggregate(label, agg, n):
    print(f"\n=== {label} (n={n}) ===")
    print(f"  Final Node Accuracy:                   "
          f"{_fmt(agg.get('fna'))}")
    print(f"  Time-Respecting Node Accuracy:         "
          f"{_fmt(agg.get('tna'))}")
    print(f"  Edge F1 (Temporal):                    "
          f"{_fmt(agg.get('edge_f1_temporal'))}  "
          f"(P={_fmt(agg.get('edge_precision_temporal'))}, "
          f"R={_fmt(agg.get('edge_recall_temporal'))})")
    print(f"  Edge F1 (Static):                      "
          f"{_fmt(agg.get('edge_f1_static'))}  "
          f"(P={_fmt(agg.get('edge_precision_static'))}, "
          f"R={_fmt(agg.get('edge_recall_static'))})")
    print(f"  Normalized Structural Distance:        "
          f"{_fmt(agg.get('nsd'))}")
    print(f"  Exact TBG Reconstruction Rate:         "
          f"{_fmt(agg.get('exact_tbg_reconstruction_rate'))}")


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
    Verifies that the four metrics behave as expected.
    """
    import random

    stories = [json.loads(l) for l in open(STORIES_PATH)][:5]
    print("Running self-test on the first 5 stories.\n")

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

    # Case 1: Perfect predictions (gold copy).
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

    # Case 3: 50% subset of edges, perfect final beliefs and trajectory.
    rng = random.Random(123)
    score_synthetic(
        "50% edges retained, perfect beliefs and trajectory",
        lambda s, gf, ge, gb: {
            "story_id": s["id"],
            "final_beliefs": dict(gf),
            "edges": rng.sample(ge, k=len(ge) // 2) if ge else [],
            "belief_trajectory": [dict(b) for b in gb],
        },
    )

    # Case 4: All edges with line numbers shifted by +1.
    score_synthetic(
        "All edges, lines shifted by +1 (Static F1 should still match)",
        lambda s, gf, ge, gb: {
            "story_id": s["id"],
            "final_beliefs": dict(gf),
            "edges": [{**e, "line": e["line"] + 1} for e in ge],
            "belief_trajectory": [dict(b) for b in gb],
        },
    )

    # Case 5: All edges with relation_type corrupted.
    def corrupt_type(e):
        relation_types = ["trusted", "untrusted", "cooperative", "deceptive"]
        wrong = [t for t in relation_types if t != e["relation_type"]]
        return {**e, "relation_type": rng.choice(wrong)}

    score_synthetic(
        "All edges, relation_type corrupted",
        lambda s, gf, ge, gb: {
            "story_id": s["id"],
            "final_beliefs": dict(gf),
            "edges": [corrupt_type(e) for e in ge],
            "belief_trajectory": [dict(b) for b in gb],
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
                        help="Break down metrics by model")
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
