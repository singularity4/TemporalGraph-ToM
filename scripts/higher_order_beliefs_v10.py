"""
Higher-order beliefs generator (Q0–Q3) — derives ground truth from TGToM stories.

================================================================================
SCRIPT DESCRIPTION
================================================================================
Reads stories_v10.jsonl. For each story, re-parses the text to recover the
event sequence, then produces ground truth and questions for the four core question types:

  Q0  true location
  Q1  first-order beliefs (one question per agent)
  Q2  higher-order belief chain (location)
  Q3  higher-order belief chain (intent)

================================================================================
INPUT (read-only):
  /mnt/user-data/outputs/v10/stories_v10.jsonl
    Each line: {id, story, agents, object}.

OUTPUT:
  /mnt/user-data/outputs/v10/higher_order_beliefs_v10.jsonl
    One entry per story. Per-entry fields:
      id, n_agents, agents, object, actual_location, comm_log,
      Q0  : {question, answer}
      Q1  : {questions: [{agent, question, answer}, ...]}
      Q2  : {chain, question, answer}
      Q3  : {chain, target_actor, target_occurrence, question, answer}

================================================================================
DETERMINISM
================================================================================
Each question type uses a per-(question_type, story_id) seed so that the
random choices for one question type do not depend on the choices made
for any other. Re-running with the same input file produces byte-identical
output, and adding/removing/reordering questions does not affect the
random choices of unrelated questions.
"""

import json
import random
import sys

sys.path.insert(0, "/mnt/user-data/outputs/v10")
from verify_v10 import parse_and_recompute  # type: ignore

STORIES_PATH = "/mnt/user-data/outputs/v10/stories_v10.jsonl"
OUT_PATH = "/mnt/user-data/outputs/v10/higher_order_beliefs_v10.jsonl"


def chain_belief_location(events, chain):
    """Latest event whose witnesses are a superset of the chain set.
    Always defined for a non-empty story (initial placement covers all)."""
    chain_set = set(chain)
    last = None
    for event in events:
        if chain_set.issubset(event["witnesses"]):
            last = event["location"]
    return last


def chain_belief_intent(events, chain, target_actor, occurrence):
    """If every chain agent witnessed target_actor's occurrence-th move,
    return that move's (help_target, hide_target). Else None."""
    if target_actor is None:
        return None
    chain_set = set(chain)
    seen = 0
    for event in events:
        if event["kind"] == "move" and event["intent"]["actor"] == target_actor:
            if seen == occurrence:
                if chain_set.issubset(event["witnesses"]):
                    return {
                        "help_target": event["intent"]["help_target"],
                        "hide_target": event["intent"]["hide_target"],
                    }
                return None
            seen += 1
    return None


def build_chain_question_location(obj, chain):
    """Build Q2 question text. Pattern: 'What does A think B thinks ... ?'"""
    parts = [f"What does {chain[0]} think"]
    for agent in chain[1:]:
        parts.append(f" {agent} thinks")
    return "".join(parts) + "?"


def build_chain_question_intent(chain, target_actor):
    """Build Q3 question text. Pattern: 'What does A think ... target's intent was?'"""
    parts = [f"What does {chain[0]} think"]
    for agent in chain[1:]:
        parts.append(f" {agent} thinks")
    return "".join(parts) + f" {target_actor}'s intent was?"


# Per-(question, story) RNG seed offsets. Each question type uses its own
# offset so that adding/removing/reordering questions does not affect the
# random choices of other questions. The seed for question Q in story s
# is computed as Q_SEED_OFFSET + s["id"].
Q2_SEED_OFFSET = 30000
Q3_SEED_OFFSET = 30100


def main():
    stories = [json.loads(l) for l in open(STORIES_PATH)]

    out = []
    for s in stories:
        obj = s["object"]
        agents = s["agents"]
        n_agents = len(agents)
        chain_depth = n_agents - 1

        # Re-derive event timeline from text. parse_and_recompute returns:
        #   loc           : actual final location
        #   first_order_beliefs     : dict of {agent: final_belief}
        #   events        : ordered list of event dicts with 'witnesses'
        actual_location, first_order_beliefs, events = parse_and_recompute(
            s["story"], obj, agents
        )

        # Derive comm_log for transparency
        comm_log = []
        for event in events:
            if event["kind"] == "comm":
                comm_log.append({
                    "speaker": event["speaker"],
                    "listeners": event.get("listeners", []),
                    "updated_listeners": event.get("updated_listeners", []),
                    "claimed_location": event.get("claimed_location"),
                    "is_truth": event.get("is_truth"),
                    "is_public": event.get("is_public"),
                })

        # ----------------- Q0 -----------------
        q0 = {
            "question": f"Where is the {obj}?",
            "answer": actual_location,
        }

        # ----------------- Q1 -----------------
        # One question per agent.
        q1_per_agent = [
            {
                "agent": agent,
                "question": f"Where does {agent} think the {obj} is?",
                "answer": first_order_beliefs[agent],
            }
            for agent in agents
        ]
        q1 = {"questions": q1_per_agent}

        # ----------------- Q2 -----------------
        # Random chain of N-1 agents.
        rng_q2 = random.Random(Q2_SEED_OFFSET + s["id"])
        q2_chain = rng_q2.sample(agents, chain_depth)
        q2 = {
            "chain": q2_chain,
            "question": build_chain_question_location(obj, q2_chain),
            "answer": chain_belief_location(events, q2_chain),
        }

        # ----------------- Q3 -----------------
        # Random target actor (a mover) and random chain of N-1 agents.
        # TARGETED SELECTION: pick (mover, occurrence) such that the move's
        # witness set (excluding the mover) is at least chain_depth in size,
        # then sample the chain from those witnesses. This guarantees a
        # non-null answer. If no qualifying move exists, fall back to using
        # the move with the largest witness pool and a chain of that smaller
        # size (still guarantees non-null at lower depth).
        rng_q3 = random.Random(Q3_SEED_OFFSET + s["id"])
        movers = sorted({
            event["intent"]["actor"]
            for event in events
            if event["kind"] == "move" and event["intent"]["actor"] is not None
        })
        # Build the qualifying set of (move_event, occurrence) pairs.
        q3_qualifying = []
        for mover_candidate in movers:
            mover_moves = [event for event in events
                           if event["kind"] == "move"
                           and event["intent"]["actor"] == mover_candidate]
            for occurrence_idx, mv in enumerate(mover_moves):
                witnesses_minus_mover = sorted(
                    a for a in mv["witnesses"] if a != mover_candidate
                )
                if len(witnesses_minus_mover) >= chain_depth:
                    q3_qualifying.append(
                        (mover_candidate, occurrence_idx, mv, witnesses_minus_mover)
                    )

        if q3_qualifying:
            target_actor, occurrence, target_mv, witness_pool = rng_q3.choice(q3_qualifying)
            q3_chain = rng_q3.sample(witness_pool, chain_depth)
            q3_answer = chain_belief_intent(
                events, q3_chain, target_actor, occurrence
            )
            q3 = {
                "chain": q3_chain,
                "target_actor": target_actor,
                "target_occurrence": occurrence,
                "question": build_chain_question_intent(q3_chain, target_actor),
                "answer": q3_answer,
            }
        elif movers:
            # No move has ≥chain_depth witnesses-excluding-mover. Find the
            # move with the largest witness pool and use a shorter chain
            # (still guarantees non-null but at lower depth).
            best_mover = None
            best_occurrence = None
            best_pool = []
            for mover_candidate in movers:
                mover_moves = [event for event in events
                               if event["kind"] == "move"
                               and event["intent"]["actor"] == mover_candidate]
                for occurrence_idx, mv in enumerate(mover_moves):
                    witnesses_minus_mover = sorted(
                        a for a in mv["witnesses"] if a != mover_candidate
                    )
                    if len(witnesses_minus_mover) > len(best_pool):
                        best_mover = mover_candidate
                        best_occurrence = occurrence_idx
                        best_pool = witnesses_minus_mover
            if best_mover is not None and len(best_pool) >= 2:
                short_depth = len(best_pool)
                target_actor = best_mover
                occurrence = best_occurrence
                q3_chain = rng_q3.sample(best_pool, short_depth)
                q3_answer = chain_belief_intent(
                    events, q3_chain, target_actor, occurrence
                )
                q3 = {
                    "chain": q3_chain,
                    "target_actor": target_actor,
                    "target_occurrence": occurrence,
                    "question": build_chain_question_intent(q3_chain, target_actor),
                    "answer": q3_answer,
                }
            else:
                # Pathological: not enough witnesses even for depth-2 chain.
                # Fall back to random with possibly null answer.
                target_actor = rng_q3.choice(movers)
                q3_chain = rng_q3.sample(agents, chain_depth)
                occurrence = 0
                q3_answer = chain_belief_intent(
                    events, q3_chain, target_actor, occurrence
                )
                q3 = {
                    "chain": q3_chain,
                    "target_actor": target_actor,
                    "target_occurrence": occurrence,
                    "question": build_chain_question_intent(q3_chain, target_actor),
                    "answer": q3_answer,
                }
        else:
            q3 = {
                "chain": None,
                "target_actor": None,
                "target_occurrence": None,
                "question": None,
                "answer": None,
            }

        out.append({
            "id": s["id"],
            "n_agents": n_agents,
            "agents": agents,
            "object": obj,
            "actual_location": actual_location,
            "comm_log": comm_log,
            "Q0": q0,
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
        })

    with open(OUT_PATH, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"Wrote {len(out)} entries -> {OUT_PATH}")
    print()
    print("--- Sample (id=0) ---")
    print(json.dumps(out[0], indent=2))


if __name__ == "__main__":
    main()
