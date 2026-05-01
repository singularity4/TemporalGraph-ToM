"""
Forward verifier for TGToM ground truth — line-by-line trust/witness sweep.

Re-parses each story and recomputes Q0..Q3 ground truth, asserting it
matches higher_order_beliefs_v10.jsonl. Acts as a check on the question
generator: parse_and_recompute walks the story text using the witness rule
and trust rule and produces a canonical event timeline; the recomputed
Q0..Q3 values must equal those stored.

Used together with verify_graph_v10.py (the TBG-style independent verifier)
to cross-check ground truth from two different reasoning paths.

Belief propagation rule:

Trust rule:
  Listener L believes speaker S iff S exited the room later than L.

Witness rule:
  Hide-target does not witness their own hiding move.
  Help-target witnesses normally.

Distractors are ignored.
"""
import json
import re

STORIES_PATH = "/mnt/user-data/outputs/v10/stories_v10.jsonl"
CORE_QUESTIONS_PATH = "/mnt/user-data/outputs/v10/higher_order_beliefs_v10.jsonl"


def parse_and_recompute(story_text, obj, agents):
    in_room = set()
    first_order_beliefs = {}
    location = None
    events = []
    move_count_per_agent = {a: 0 for a in agents}
    room = None

    # exit_step: line index of each agent's "X exited the <room>." line.
    # Agents that never exit get sentinel 10**9 (still in room).
    exit_step = {}
    line_idx = 0  # increments for every line, including distractors

    lines = story_text.split("\n")

    for raw in lines:
        line_idx += 1
        m = re.match(r"^\d+\s+(.*)$", raw)
        s = m.group(1)

        # Placement
        m_place = re.match(rf"^The {re.escape(obj)} is in the (\S+)\.$", s)
        if m_place:
            location = m_place.group(1)
            for a in in_room:
                first_order_beliefs[a] = location
            events.append({
                "kind": "place",
                "location": location,
                "witnesses": frozenset(in_room),
                "intent": None,
            })
            continue

        # Move
        m_move = re.match(rf"^(\S+) moved the {re.escape(obj)} to the (.+)\.$", s)
        if m_move:
            mover = m_move.group(1)
            tail = m_move.group(2)
            help_target = None
            hide_target = None
            help_marker = " to help "
            hide_marker = " to hide it from "
            help_pos = tail.find(help_marker)
            hide_pos = tail.find(hide_marker)
            markers = []
            if help_pos != -1: markers.append(("help", help_pos))
            if hide_pos != -1: markers.append(("hide", hide_pos))
            markers.sort(key=lambda x: x[1])
            if not markers:
                destination = tail
            else:
                destination = tail[:markers[0][1]]
                for kind, pos in markers:
                    if kind == "help":
                        rest = tail[pos + len(help_marker):]
                        # Phrasing is "to help X find it" — followed by
                        # " and to hide ..." (multi-intent) or end-of-tail
                        # (help-only). The trailing period is already stripped
                        # by the move regex.
                        marker = " find it"
                        end_continuation = rest.find(marker + " ")
                        if end_continuation != -1:
                            end = end_continuation
                        elif rest.endswith(marker):
                            end = len(rest) - len(marker)
                        else:
                            end = -1
                        help_target = rest[:end] if end != -1 else rest
                    else:
                        rest = tail[pos + len(hide_marker):]
                        end = rest.find(" and ")
                        hide_target = rest if end == -1 else rest[:end]
            location = destination
            # Witness rule: hide_target excluded from witnesses.
            witnesses_set = set(in_room)
            if hide_target is not None and hide_target in witnesses_set:
                witnesses_set.discard(hide_target)
            witnesses = frozenset(witnesses_set)
            event = {
                "kind": "move",
                "location": destination,
                "witnesses": witnesses,
                "intent": {
                    "actor": mover,
                    "help_target": help_target,
                    "hide_target": hide_target,
                    "destination": destination,
                    "occurrence": move_count_per_agent[mover],
                },
            }
            events.append(event)
            move_count_per_agent[mover] += 1
            # Belief update follows the witness rule: only witnesses update.
            # The hide_target is in the room but did not witness, so they
            # do not update their belief.
            for witness in witnesses_set:
                first_order_beliefs[witness] = destination
            continue

        # Stay
        m_stay = re.match(r"^(\S+) made no movements and stayed in the \S+ for 1 minute\.$", s)
        if m_stay:
            continue

        # Exit
        m_exit = re.match(r"^(\S+) exited the \S+\.$", s)
        if m_exit:
            agent = m_exit.group(1)
            in_room.discard(agent)
            exit_step[agent] = line_idx
            continue

        # Public communication
        m_pub = re.match(rf"^(\S+) publicly claimed that the {re.escape(obj)} is in the (\S+)\.$", s)
        if m_pub:
            speaker = m_pub.group(1)
            claimed_location = m_pub.group(2)
            speaker_step = exit_step.get(speaker, 10**9)
            updated_listeners = []
            for L in agents:
                if L == speaker:
                    continue
                listener_step = exit_step.get(L, 10**9)
                if speaker_step > listener_step:
                    first_order_beliefs[L] = claimed_location
                    updated_listeners.append(L)
            events.append({
                "kind": "comm",
                "speaker": speaker,
                "listeners": [a for a in agents if a != speaker],
                "updated_listeners": updated_listeners,
                "claimed_location": claimed_location,
                "is_public": True,
                "witnesses": frozenset(updated_listeners),
                "location": claimed_location,
                "intent": None,
            })
            continue

        # Private communication
        m_priv = re.match(rf"^(\S+) privately told (\S+) that the {re.escape(obj)} is in the (\S+)\.$", s)
        if m_priv:
            speaker = m_priv.group(1)
            listener = m_priv.group(2)
            claimed_location = m_priv.group(3)
            speaker_step = exit_step.get(speaker, 10**9)
            listener_step = exit_step.get(listener, 10**9)
            updated_listeners = []
            if speaker_step > listener_step:
                first_order_beliefs[listener] = claimed_location
                updated_listeners.append(listener)
            events.append({
                "kind": "comm",
                "speaker": speaker,
                "listeners": [listener],
                "updated_listeners": updated_listeners,
                "claimed_location": claimed_location,
                "is_public": False,
                "witnesses": frozenset(updated_listeners),
                "location": claimed_location,
                "intent": None,
            })
            continue

        # Entry / reconvene
        m_enter = re.match(r"^(.+) entered the (\S+)\.$", s)
        if m_enter:
            names_part = m_enter.group(1)
            entered_room = m_enter.group(2)
            names = [n.strip() for n in re.split(r",| and ", names_part) if n.strip()]
            if room is None:
                room = entered_room
                for n in names:
                    in_room.add(n)
            continue

        # Otherwise: distractor, ignored.

    return location, first_order_beliefs, events


def main():
    """Verify higher_order_beliefs_v10.jsonl by re-deriving Q0..Q3 from stories alone."""
    stories = [json.loads(l) for l in open(STORIES_PATH)]
    qa_core = [json.loads(l) for l in open(CORE_QUESTIONS_PATH)]
    n_mismatches = 0
    for s, qa in zip(stories, qa_core):
        actual_location, beliefs, events = parse_and_recompute(
            s["story"], s["object"], s["agents"]
        )
        problems = []

        # Q0
        if actual_location != qa["Q0"]["answer"]:
            problems.append(
                f"Q0: stored={qa['Q0']['answer']} recomp={actual_location}"
            )

        # Q1: per-agent
        recomputed_q1 = {agent: beliefs[agent] for agent in s["agents"]}
        stored_q1 = {q["agent"]: q["answer"] for q in qa["Q1"]["questions"]}
        if recomputed_q1 != stored_q1:
            problems.append(f"Q1: stored={stored_q1} recomp={recomputed_q1}")

        # Q2: higher-order location chain
        chain_set = set(qa["Q2"]["chain"])
        last_location = None
        for event in events:
            if chain_set.issubset(event["witnesses"]):
                last_location = event["location"]
        if last_location != qa["Q2"]["answer"]:
            problems.append(
                f"Q2: stored={qa['Q2']['answer']} recomp={last_location}"
            )

        # Q3: higher-order intent chain
        q3 = qa["Q3"]
        target_actor = q3.get("target_actor")
        occurrence = q3.get("target_occurrence")
        chain_set = set(q3["chain"]) if q3.get("chain") else set()
        recomputed_q3 = None
        if target_actor is not None:
            seen = 0
            for event in events:
                if event["kind"] == "move" and event["intent"]["actor"] == target_actor:
                    if seen == occurrence:
                        if chain_set.issubset(event["witnesses"]):
                            recomputed_q3 = {
                                "help_target": event["intent"]["help_target"],
                                "hide_target": event["intent"]["hide_target"],
                            }
                        break
                    seen += 1
        if recomputed_q3 != q3["answer"]:
            problems.append(
                f"Q3: stored={q3['answer']} recomp={recomputed_q3}"
            )

        if problems:
            n_mismatches += 1
            if n_mismatches <= 5:
                print(f"\n--- Mismatch id={s['id']} ---")
                for p in problems:
                    print(" ", p)
    print(f"\nTotal {len(stories)}, mismatches {n_mismatches}")


if __name__ == "__main__":
    main()
