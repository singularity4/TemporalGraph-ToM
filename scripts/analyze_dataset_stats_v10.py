"""
Dataset statistics analysis for TGToM (TemporalGraph-ToM).

================================================================================
SCRIPT DESCRIPTION
================================================================================
Reads stories and the four question JSONL files and reports descriptive
statistics about the benchmark. 

Reports:
  - Story-level: count, agents per story, distinct containers, distinct
    rooms, distinct objects.
  - Event-level: move count, comm count (public vs private), exit count.
  - Intent distribution: hide vs help intents per move.
  - Trust distribution: trusted vs untrusted comms (under the trust rule).
  - Per-question chain-depth distributions.
  - Per-question yes/no rates (for Q9, Q11, Q13).
  - Per-question null-answer rates (where applicable).

================================================================================
INPUT (read-only):
  /mnt/user-data/outputs/v10/stories_v10.jsonl
  /mnt/user-data/outputs/v10/higher_order_beliefs_v10.jsonl
  /mnt/user-data/outputs/v10/counterfactual_beliefs_v10.jsonl
  /mnt/user-data/outputs/v10/causal_beliefs_v10.jsonl
  /mnt/user-data/outputs/v10/common_knowledge_v10.jsonl

================================================================================
USAGE
================================================================================
    python analyze_dataset_stats_v10.py
"""

import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/mnt/user-data/outputs/v10")
from verify_v10 import parse_and_recompute  # type: ignore

STORIES_PATH = "/mnt/user-data/outputs/v10/stories_v10.jsonl"
HIGHER_PATH = "/mnt/user-data/outputs/v10/higher_order_beliefs_v10.jsonl"
COUNTERFACTUAL_PATH = "/mnt/user-data/outputs/v10/counterfactual_beliefs_v10.jsonl"
CAUSAL_PATH = "/mnt/user-data/outputs/v10/causal_beliefs_v10.jsonl"
KNOWLEDGE_PATH = "/mnt/user-data/outputs/v10/common_knowledge_v10.jsonl"


def histogram(values):
    """Return a dict {value: count} sorted by value."""
    counter = Counter(values)
    return dict(sorted(counter.items()))


def report_story_stats(stories):
    print("\n=== STORY-LEVEL ===")
    print(f"Total stories: {len(stories)}")

    agent_counts = [len(s["agents"]) for s in stories]
    print(f"Agents per story: min={min(agent_counts)}, "
          f"max={max(agent_counts)}, mean={sum(agent_counts)/len(agent_counts):.2f}")
    print(f"  Distribution: {histogram(agent_counts)}")

    objects = [s["object"] for s in stories]
    print(f"Distinct objects: {len(set(objects))}")

    rooms = []
    containers = []
    for s in stories:
        first_line = s["story"].split("\n")[0]
        m_room = re.search(r"entered the (\S+)\.$", first_line)
        if m_room:
            rooms.append(m_room.group(1))
        for line in s["story"].split("\n"):
            for m in re.finditer(r"(?:in|to) the (\S+)(?:[.,]| from)", line):
                containers.append(m.group(1))
    print(f"Distinct rooms (sampled from line 1): {len(set(rooms))}")
    print(f"Distinct containers used: {len(set(containers))}")


def report_event_stats(stories):
    print("\n=== EVENT-LEVEL ===")
    total_moves = 0
    total_comms = 0
    total_public_comms = 0
    total_private_comms = 0
    total_exits = 0
    total_hide_intents = 0
    total_help_intents = 0
    total_no_intent = 0
    total_trusted_comms = 0
    total_untrusted_comms = 0

    for s in stories:
        try:
            _loc, _fo, events = parse_and_recompute(s["story"], s["object"], s["agents"])
        except Exception as e:
            print(f"  Failed parsing story id={s['id']}: {e}", file=sys.stderr)
            continue
        for event in events:
            if event["kind"] == "move":
                total_moves += 1
                hide_t = event.get("intent", {}).get("hide_target") if event.get("intent") else None
                help_t = event.get("intent", {}).get("help_target") if event.get("intent") else None
                if hide_t is not None:
                    total_hide_intents += 1
                if help_t is not None:
                    total_help_intents += 1
                if hide_t is None and help_t is None:
                    total_no_intent += 1
            elif event["kind"] == "comm":
                total_comms += 1
                if event.get("is_public"):
                    total_public_comms += 1
                else:
                    total_private_comms += 1
                # trusted_listeners are stored as updated_listeners
                trusted = event.get("updated_listeners") or []
                listeners = event.get("listeners") or []
                total_trusted_comms += len(trusted)
                total_untrusted_comms += max(0, len(listeners) - len(trusted))

        # Count exits from text
        for line in s["story"].split("\n"):
            if re.match(r"^\d+ \S+ exited the \S+\.$", line):
                total_exits += 1

    n = len(stories)
    print(f"Total moves: {total_moves} ({total_moves/n:.2f} per story)")
    print(f"Total comms: {total_comms} ({total_comms/n:.2f} per story)")
    print(f"  Public:  {total_public_comms}")
    print(f"  Private: {total_private_comms}")
    print(f"Total exits: {total_exits} ({total_exits/n:.2f} per story)")
    print(f"\nMove intent distribution:")
    print(f"  Hide intent:  {total_hide_intents} ({total_hide_intents/max(1,total_moves):.1%})")
    print(f"  Help intent:  {total_help_intents} ({total_help_intents/max(1,total_moves):.1%})")
    print(f"  No intent:    {total_no_intent} ({total_no_intent/max(1,total_moves):.1%})")
    print(f"\nComm trust distribution (counted per (comm, listener)):")
    print(f"  Trusted (speaker exited later):    {total_trusted_comms}")
    print(f"  Untrusted (speaker exited earlier): {total_untrusted_comms}")


def report_question_stats():
    print("\n=== QUESTION-LEVEL ===")

    higher = [json.loads(l) for l in open(HIGHER_PATH)]
    cf = [json.loads(l) for l in open(COUNTERFACTUAL_PATH)]
    causal = [json.loads(l) for l in open(CAUSAL_PATH)]
    knowledge = [json.loads(l) for l in open(KNOWLEDGE_PATH)]

    n = len(higher)
    print(f"\n--- Higher-order beliefs (Q0–Q3) ---")
    q1_per_story = [len(e["Q1"]["questions"]) for e in higher]
    print(f"Q1 questions per story: min={min(q1_per_story)}, "
          f"max={max(q1_per_story)}, total={sum(q1_per_story)}")

    q2_depths = [len(e["Q2"]["chain"]) for e in higher]
    print(f"Q2 chain depth: {histogram(q2_depths)}")
    q2_null = sum(1 for e in higher if e["Q2"]["answer"] is None)
    print(f"Q2 null answers: {q2_null}/{n}")

    q3_depths = [len(e["Q3"]["chain"]) for e in higher if e["Q3"].get("chain")]
    if q3_depths:
        print(f"Q3 chain depth: {histogram(q3_depths)}")
    q3_null = sum(1 for e in higher if e["Q3"]["answer"] is None)
    print(f"Q3 null answers: {q3_null}/{n}")

    print(f"\n--- Counterfactual beliefs (Q5–Q7, Q10) ---")
    for q in ("Q5", "Q6", "Q7", "Q10"):
        targeted = sum(1 for e in cf if e[q].get("targeted"))
        nulls = sum(1 for e in cf if e[q].get("answer") is None)
        print(f"  {q}: targeted={targeted}/{n}, null answers={nulls}/{n}")
    q6_depths = [len(e["Q6"]["chain"]) for e in cf if e["Q6"].get("chain")]
    if q6_depths:
        print(f"  Q6 chain depth: {histogram(q6_depths)}")
    q7_depths = [len(e["Q7"]["chain"]) for e in cf if e["Q7"].get("chain")]
    if q7_depths:
        print(f"  Q7 chain depth: {histogram(q7_depths)}")
    q10_depths = [len(e["Q10"]["chain"]) for e in cf if e["Q10"].get("chain")]
    if q10_depths:
        print(f"  Q10 chain depth: {histogram(q10_depths)}")

    print(f"\n--- Causal beliefs (Q8) ---")
    q8_kinds = [e["Q8"]["answer"]["kind"] for e in causal if e["Q8"].get("answer")]
    print(f"  Cause-event kind distribution: {histogram(q8_kinds)}")

    print(f"\n--- Common knowledge (Q9, Q11, Q13) ---")
    for q in ("Q9", "Q11", "Q13"):
        yes = sum(1 for e in knowledge if e[q].get("answer") and e[q]["answer"].get("common_knowledge"))
        nulls = sum(1 for e in knowledge if e[q].get("answer") is None)
        targeted = sum(1 for e in knowledge if e[q].get("targeted"))
        targeted_str = f", targeted={targeted}/{n}" if q != "Q9" else ""
        print(f"  {q}: yes={yes}/{n}, null={nulls}/{n}{targeted_str}")
    q9_sizes = [len(e["Q9"]["agent_set"]) for e in knowledge if e["Q9"].get("agent_set")]
    print(f"  Q9 agent_set size: {histogram(q9_sizes)}")
    q11_sizes = [len(e["Q11"]["agent_set_S"]) for e in knowledge if e["Q11"].get("agent_set_S")]
    print(f"  Q11 agent_set size: {histogram(q11_sizes)}")
    q13_sizes = [len(e["Q13"]["agent_set_S"]) for e in knowledge if e["Q13"].get("agent_set_S")]
    if q13_sizes:
        print(f"  Q13 agent_set size: {histogram(q13_sizes)}")


def main():
    stories = [json.loads(l) for l in open(STORIES_PATH)]
    print(f"=== TGToM dataset statistics ===")
    report_story_stats(stories)
    report_event_stats(stories)
    report_question_stats()
    print()


if __name__ == "__main__":
    main()
