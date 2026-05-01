"""
Causal beliefs generator (Q8) — derives ground truth from TGToM stories.

================================================================================
INPUT (read-only):
  /mnt/user-data/outputs/v10/stories_v10.jsonl
  /mnt/user-data/outputs/v10/higher_order_beliefs_v10.jsonl

OUTPUT:
  /mnt/user-data/outputs/v10/causal_beliefs_v10.jsonl
    one entry per story with Q8: question text, answer, and metadata.

================================================================================
QUESTION FAMILY
================================================================================

Q8 — CAUSE OF BELIEF / BELIEF INERTIA 
  "Which event caused <agent>'s final belief about the <object> to take its
   current value?"
  Answer = the latest event the agent witnessed (move, place, or trusted
  comm under the trust rule). Returned as a structured field with line
  number, kind, and short description.
  Selection: random agent. Always defined (every agent witnesses at least
  the placement at line 2).
  Tests: backward causal attribution — which event in the actual story
  produced the agent's current belief.

================================================================================
GROUND-TRUTH DERIVATION
================================================================================
- Re-derive event timeline from text via verify_v10.parse_and_recompute.
- Q8: walk events in order; for the chosen agent A, the latest event whose
  witnesses include A (for moves/place) OR whose updated_listeners include
  A (for comms) is the cause.

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
import re
import sys

sys.path.insert(0, "/mnt/user-data/outputs/v10")
from verify_v10 import parse_and_recompute  # type: ignore

STORIES_PATH = "/mnt/user-data/outputs/v10/stories_v10.jsonl"
CORE_QUESTIONS_PATH = "/mnt/user-data/outputs/v10/higher_order_beliefs_v10.jsonl"
OUT_PATH = "/mnt/user-data/outputs/v10/causal_beliefs_v10.jsonl"

# Per-(question, story) seed offsets — see DETERMINISM section above.
Q8_SEED_OFFSET = 80000


def get_real_lines(story_text):
    out = []
    for raw in story_text.split("\n"):
        m = re.match(r"^(\d+)\s+(.*)$", raw)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


def find_event_line(real_lines, event, obj):
    """Find the 1-indexed line number of the event in the story text.
    For 'place': the placement line.
    For 'move': the move line by (mover, location).
    For 'comm': the comm line by (speaker, claimed_location).
    Uses occurrence counting to disambiguate when speaker/mover has multiple."""
    if event["kind"] == "place":
        for line_no, text in real_lines:
            if text.startswith(f"The {obj} is in the "):
                return line_no
        return None
    if event["kind"] == "move":
        mover = event["intent"]["actor"]
        target_occ = event["intent"]["occurrence"]
        occ = 0
        for line_no, text in real_lines:
            if text.startswith(f"{mover} moved the "):
                if occ == target_occ:
                    return line_no
                occ += 1
        return None
    if event["kind"] == "comm":
        speaker = event["speaker"]
        # We don't have an occurrence in comm events; count by (speaker, claimed_location)
        # against the order of comms in events (passed via caller indexing)
        return None  # filled in by caller using event index
    return None


def event_short_description(event, obj):
    """Short readable description of an event."""
    if event["kind"] == "place":
        return f"the {obj} placed in the {event['location']}"
    if event["kind"] == "move":
        actor = event["intent"]["actor"]
        return f"{actor} moved the {obj} to the {event['location']}"
    if event["kind"] == "comm":
        speaker = event["speaker"]
        if event.get("is_public", True):
            return f"{speaker} publicly claimed the {obj} is in the {event['location']}"
        listeners = event.get("listeners", [])
        listener = listeners[0] if listeners else "someone"
        return f"{speaker} privately told {listener} the {obj} is in the {event['location']}"
    return "unknown event"


def latest_event_witnessed_by(events, agent):
    """Return (latest_event, latest_index) where latest_event is the latest
    event whose effective witness set includes agent, and latest_index is
    its position in `events`. Returns (None, -1) if no such event exists.

    For moves/place: agent in event['witnesses'].
    For comms: agent in event['updated_listeners'] (i.e., agent trusted the speaker).
    Note: parse_and_recompute stores comm 'witnesses' = updated_listeners already.

    Returning the index explicitly avoids a subtle bug: two comm events
    with identical content (same speaker, same claim, same witnesses)
    compare equal as dicts, so `events.index(latest_event)` would return
    the first matching event rather than the actual latest one.
    """
    last_event = None
    last_index = -1
    for index, event in enumerate(events):
        if agent in event["witnesses"]:
            last_event = event
            last_index = index
    return last_event, last_index


def find_comm_line_by_index(real_lines, comm_event_index_in_comms):
    """Find the line number of the comm_event_index_in_comms-th comm
    (0-indexed) in the story text — comms in story appear in event order."""
    seen = 0
    for line_no, text in real_lines:
        if (" publicly claimed that the " in text or
            " privately told " in text):
            if seen == comm_event_index_in_comms:
                return line_no
            seen += 1
    return None


def main():
    stories = [json.loads(l) for l in open(STORIES_PATH)]
    qa_core = [json.loads(l) for l in open(CORE_QUESTIONS_PATH)]

    # Merge story + qa_core into a unified per-id view for downstream code
    # that expects bundled story metadata plus derived ground truth.
    answers = []
    for s_, qa in zip(stories, qa_core):
        answers.append({
            'id': s_['id'],
            'object': s_['object'],
            'agents': s_['agents'],
            'n_agents': qa['n_agents'],
            'actual_location': qa['actual_location'],
            'first_order_beliefs': {q['agent']: q['answer'] for q in qa['Q1']['questions']},
            'higher_order_location': {'chain': qa['Q2']['chain'], 'answer': qa['Q2']['answer']},
            'higher_order_intent': {'chain': qa['Q3']['chain'], 'target_actor': qa['Q3']['target_actor'], 'target_occurrence': qa['Q3']['target_occurrence'], 'answer': qa['Q3']['answer']},
            'comm_log': qa['comm_log'],
        })

    out = []
    for s, a in zip(stories, answers):
        obj = a["object"]
        agents = a["agents"]
        n_agents = a["n_agents"]
        story_text = s["story"]
        loc, first_order_beliefs, events = parse_and_recompute(story_text, obj, agents)
        real_lines = get_real_lines(story_text)

        # Pre-compute the line number for each event (in the order they appear
        # in the events list).
        comm_idx = 0
        event_line_nos = []
        for event in events:
            if event["kind"] == "comm":
                line_no = find_comm_line_by_index(real_lines, comm_idx)
                comm_idx += 1
            else:
                line_no = find_event_line(real_lines, event, obj)
            event_line_nos.append(line_no)

        # ----------------------- Q8 -----------------------
        # Belief inertia: which event caused [agent]'s final belief?
        rng_q8 = random.Random(Q8_SEED_OFFSET + s["id"])
        q8_agent = rng_q8.choice(agents)
        cause_ev, cause_idx = latest_event_witnessed_by(events, q8_agent)
        if cause_ev is None:
            q8_entry = {
                "question_type": "Q8",
                "subtype": "belief_inertia",
                "agent": q8_agent,
                "question": (
                    f"Which event caused {q8_agent}'s final belief about the "
                    f"{obj} to take its current value?"
                ),
                "answer": None,
            }
        else:
            cause_line = event_line_nos[cause_idx]
            q8_entry = {
                "question_type": "Q8",
                "subtype": "belief_inertia",
                "agent": q8_agent,
                "question": (
                    f"Which event caused {q8_agent}'s final belief about the "
                    f"{obj} to take its current value? Answer with the line "
                    f"number from the story."
                ),
                "answer": {
                    "line": cause_line,
                    "kind": cause_ev["kind"],
                    "description": event_short_description(cause_ev, obj),
                    "resulting_belief": cause_ev["location"],
                },
            }

        out.append({
            "id": s["id"],
            "Q8": q8_entry,
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
