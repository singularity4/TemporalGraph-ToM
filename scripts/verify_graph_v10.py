"""
TBG verifier — independent graph-based verifier for ground truth.

Builds an explicit Temporal Belief Graph (TBG) from each story:

  Nodes:  (agent, time_step) pairs, each holding that agent's belief about
          the object's location at that timestep.
  Links:  - Default link: (agent, t) -> (agent, t+1) with unchanged belief.
          - Witness update: if event at step t is a placement or move with
            agent in witnesses, (agent, t+1).belief = event.location.
          - Belief (trust) update: if event at step t is a communication where the
            speaker exited later than the listener, (listener, t+1).belief =
            claimed_location.

Final beliefs are read off as (agent, T_final).belief.

This implements the TBG reasoning scaffold described in the abstract: nodes
are agent beliefs after each event; communication links and intent links determine
how beliefs propagate; observations are direct node-state updates (no links).

This is a structurally different computation from verify_v10.py, which does
a single forward pass over the text. They should agree on the ground truth.

Belief propagation trust rule (same as verify_v10.py):
  Listener trusts speaker iff speaker exited the room later than the listener.
"""

import json
import re

STORIES_PATH = "/mnt/user-data/outputs/v11/data/stories_v10.jsonl"
CORE_QUESTIONS_PATH = "/mnt/user-data/outputs/v11/data/higher_order_beliefs_v10.jsonl"


# --------------------------------------------------------------------------
# Story parsing — extract events without computing any beliefs.
# --------------------------------------------------------------------------

def parse_events(story_text, obj, agents):
    """
    Walk the story text and return:
      events:    ordered list of {kind, location, ..., line_index}
                 kind ∈ {place, move, comm}
      exit_step: dict {agent: line_index} of when each agent exited
                 (sentinel 10**9 for agents who never explicitly exit)

    Belief computation happens in build_graph().
    """
    events = []
    exit_step = {a: 10**9 for a in agents}
    in_room = set()
    room_set = False
    line_index = 0
    move_count_per_agent = {a: 0 for a in agents}

    for raw in story_text.split("\n"):
        line_index += 1
        m = re.match(r"^\d+\s+(.*)$", raw)
        if not m:
            continue
        text = m.group(1)

        # Initial entry (only the first one defines in_room; the second
        # is the reconvene line and does not affect belief tracking).
        m_enter = re.match(r"^(.+) entered the (\S+)\.$", text)
        if m_enter and not room_set:
            names = [n.strip() for n in re.split(r",| and ", m_enter.group(1)) if n.strip()]
            for n in names:
                in_room.add(n)
            room_set = True
            continue
        if m_enter and room_set:
            # reconvene; ignore for graph purposes
            continue

        # Placement
        m_place = re.match(rf"^The {re.escape(obj)} is in the (\S+)\.$", text)
        if m_place:
            location = m_place.group(1)
            events.append({
                "kind": "place",
                "line_index": line_index,
                "location": location,
                "witnesses": frozenset(in_room),
            })
            continue

        # Move (with optional motive clause)
        m_move = re.match(rf"^(\S+) moved the {re.escape(obj)} to the (.+)\.$", text)
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
            if help_pos != -1:
                markers.append(("help", help_pos))
            if hide_pos != -1:
                markers.append(("hide", hide_pos))
            markers.sort(key=lambda x: x[1])
            if not markers:
                destination = tail
            else:
                destination = tail[:markers[0][1]]
                for kind, pos in markers:
                    if kind == "help":
                        rest = tail[pos + len(help_marker):]
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

            # Witness rule: every agent in the room except the hide-target.
            witnesses_set = set(in_room)
            if hide_target is not None:
                witnesses_set.discard(hide_target)

            events.append({
                "kind": "move",
                "line_index": line_index,
                "mover": mover,
                "location": destination,
                "help_target": help_target,
                "hide_target": hide_target,
                "occurrence": move_count_per_agent[mover],
                "witnesses": frozenset(witnesses_set),
            })
            move_count_per_agent[mover] += 1
            continue

        # Stay (no belief change)
        if re.match(r"^(\S+) made no movements and stayed in the \S+ for 1 minute\.$", text):
            continue

        # Exit
        m_exit = re.match(r"^(\S+) exited the \S+\.$", text)
        if m_exit:
            agent = m_exit.group(1)
            in_room.discard(agent)
            exit_step[agent] = line_index
            continue

        # Public comm
        m_pub = re.match(rf"^(\S+) publicly claimed that the {re.escape(obj)} is in the (\S+)\.$", text)
        if m_pub:
            speaker = m_pub.group(1)
            claimed_location = m_pub.group(2)
            events.append({
                "kind": "comm",
                "line_index": line_index,
                "speaker": speaker,
                "listeners": [a for a in agents if a != speaker],
                "claimed_location": claimed_location,
                "is_public": True,
            })
            continue

        # Private comm
        m_priv = re.match(rf"^(\S+) privately told (\S+) that the {re.escape(obj)} is in the (\S+)\.$", text)
        if m_priv:
            speaker = m_priv.group(1)
            listener = m_priv.group(2)
            claimed_location = m_priv.group(3)
            events.append({
                "kind": "comm",
                "line_index": line_index,
                "speaker": speaker,
                "listeners": [listener],
                "claimed_location": claimed_location,
                "is_public": False,
            })
            continue

        # Otherwise: distractor, ignored.

    return events, exit_step


# --------------------------------------------------------------------------
# Graph construction and belief propagation
# --------------------------------------------------------------------------

def build_graph(events, exit_step, agents):
    """
    Build the temporal belief graph. Returns belief[t][agent] for t in
    {0, 1, ..., len(events)}, where belief[t] is the state AFTER applying
    the t-th event (belief[0] is the empty initial state).

    Graph propagation rules:
      - Default link: belief[t+1][agent] starts as belief[t][agent].
      - For a place or move event with witnesses W and location L:
          for w in W: belief[t+1][w] = L
      - For a comm event with speaker S and listeners L:
          for ell in L:
            if exit_step[S] > exit_step[ell]:  (trust rule)
              belief[t+1][ell] = claimed_location
    """
    T = len(events)
    belief = [{a: None for a in agents}]  # belief[0]: empty initial state

    for t, event in enumerate(events):
        new_state = dict(belief[t])  # carry forward by default

        if event["kind"] in ("place", "move"):
            for witness in event["witnesses"]:
                new_state[witness] = event["location"]
        elif event["kind"] == "comm":
            speaker = event["speaker"]
            speaker_step = exit_step.get(speaker, 10**9)
            for listener in event["listeners"]:
                listener_step = exit_step.get(listener, 10**9)
                if speaker_step > listener_step:
                    new_state[listener] = event["claimed_location"]

        belief.append(new_state)

    return belief


def witnesses_for_event(event, exit_step):
    """
    Return the effective 'witnesses' set for an event (= agents whose
    belief is updated by it). For place/move this is event['witnesses'].
    For comm this is the subset of listeners who applied the trust rule.
    Used for higher-order chain questions (Q2, etc.).
    """
    if event["kind"] in ("place", "move"):
        return event["witnesses"]
    speaker = event["speaker"]
    speaker_step = exit_step.get(speaker, 10**9)
    updated = []
    for listener in event["listeners"]:
        listener_step = exit_step.get(listener, 10**9)
        if speaker_step > listener_step:
            updated.append(listener)
    return frozenset(updated)


# --------------------------------------------------------------------------
# Cross-check against higher_order_beliefs_v10.jsonl
# --------------------------------------------------------------------------

def reparse_events_with_modified_exits(events, exit_step_overrides, agents):
    """
    Re-derive event witness sets when one or more agents' exit_step is
    overridden. Used for Q10 (exit-swap) and Q11 (exit-change).

    The witness set of a place/move event is determined by who is in the
    room at that line. With modified exit_steps, an agent A is in the
    room at line L iff their (possibly overridden) exit_step > L.

    Returns: (modified_events, new_exit_step) where modified_events have
    updated `witnesses` for place/move and the same comm fields (the
    trust rule for comms uses `new_exit_step` directly via
    witnesses_for_event).
    """
    new_exit_step = dict(exit_step_overrides)  # all agents must have a value
    modified = []
    for event in events:
        if event["kind"] in ("place", "move"):
            line = event["line_index"]
            present = {a for a in agents if new_exit_step[a] > line}
            new_witnesses = set(present)
            if event["kind"] == "move":
                hide_target = event.get("hide_target")
                if hide_target is not None:
                    new_witnesses.discard(hide_target)
            new_event = dict(event)
            new_event["witnesses"] = frozenset(new_witnesses)
            modified.append(new_event)
        else:  # comm — content unchanged; effective witnesses depend on
               # new_exit_step and are computed by witnesses_for_event.
            modified.append(event)
    return modified, new_exit_step


def reparse_with_motive_flip(events, target_event, new_help, new_hide, in_room_at_event):
    """
    Re-derive the witness set of a single move event after swapping its
    motive (used for Q7). Other events are unchanged.

    in_room_at_event: set of agents in the room at the target event's line.
    """
    modified = []
    for event in events:
        if event is target_event:
            new_event = dict(event)
            new_event["help_target"] = new_help
            new_event["hide_target"] = new_hide
            new_witnesses = set(in_room_at_event)
            if new_hide is not None:
                new_witnesses.discard(new_hide)
            new_event["witnesses"] = frozenset(new_witnesses)
            modified.append(new_event)
        else:
            modified.append(event)
    return modified


def in_room_at_line(events, agents, exit_step, line):
    """Set of agents present in the room at a given line index."""
    return {a for a in agents if exit_step[a] > line}


def find_event_at_line(events, line):
    for event in events:
        if event["line_index"] == line:
            return event
    return None


# --------------------------------------------------------------------------
# Question-by-question graph checks
# --------------------------------------------------------------------------

def check_q0_q3(core_entry, events, exit_step, belief, agents):
    """Q0: actual location. Q1: final beliefs. Q2: chain location.
    Q3: chain intent. Returns list of problem strings (empty if all pass)."""
    final_belief = belief[-1]
    problems = []

    # Q0
    actual_location = None
    for event in events:
        if event["kind"] in ("place", "move"):
            actual_location = event["location"]
    if actual_location != core_entry["Q0"]["answer"]:
        problems.append(f"Q0: stored={core_entry['Q0']['answer']} graph={actual_location}")

    # Q1
    stored_q1 = {q["agent"]: q["answer"] for q in core_entry["Q1"]["questions"]}
    if final_belief != stored_q1:
        problems.append(f"Q1: stored={stored_q1} graph={final_belief}")

    # Q2
    chain_set = set(core_entry["Q2"]["chain"])
    last_location = None
    for event in events:
        if chain_set.issubset(witnesses_for_event(event, exit_step)):
            last_location = event["location"] if event["kind"] != "comm" else event["claimed_location"]
    if last_location != core_entry["Q2"]["answer"]:
        problems.append(f"Q2: stored={core_entry['Q2']['answer']} graph={last_location}")

    # Q3
    q3 = core_entry["Q3"]
    target_actor = q3.get("target_actor")
    occurrence = q3.get("target_occurrence")
    chain_set_q3 = set(q3["chain"]) if q3.get("chain") else set()
    graph_q3 = None
    if target_actor is not None:
        seen = 0
        for event in events:
            if event["kind"] == "move" and event["mover"] == target_actor:
                if seen == occurrence:
                    if chain_set_q3.issubset(event["witnesses"]):
                        graph_q3 = {
                            "help_target": event["help_target"],
                            "hide_target": event["hide_target"],
                        }
                    break
                seen += 1
    if graph_q3 != q3["answer"]:
        problems.append(f"Q3: stored={q3['answer']} graph={graph_q3}")

    return problems


def check_q5_q7(cf_entry, events, exit_step, belief, agents):
    """Q5/Q6/Q7: counterfactual variants of belief and intent."""
    final_belief = belief[-1]
    problems = []

    # Q5: drop a move at perturbed_line. Re-build graph without that event.
    q5 = cf_entry["Q5"]
    drop_line = q5["perturbed_line"]
    events_q5 = [e for e in events if e["line_index"] != drop_line]
    belief_q5 = build_graph(events_q5, exit_step, agents)
    asked_q5 = q5["asked_about"]
    graph_q5 = belief_q5[-1].get(asked_q5)
    if graph_q5 != q5["answer"]:
        problems.append(f"Q5: stored={q5['answer']} graph={graph_q5}")

    # Q6: swap a comm's claimed_location. Trust ordering unchanged, only content.
    q6 = cf_entry["Q6"]
    swap_line_q6 = q6["perturbed_line"]
    new_claim_q6 = q6["counterfactual_claim"]
    events_q6 = []
    for event in events:
        if event["line_index"] == swap_line_q6 and event["kind"] == "comm":
            new_event = dict(event)
            new_event["claimed_location"] = new_claim_q6
            events_q6.append(new_event)
        else:
            events_q6.append(event)
    chain_set_q6 = set(q6["chain"])
    last_loc_q6 = None
    for event in events_q6:
        if chain_set_q6.issubset(witnesses_for_event(event, exit_step)):
            last_loc_q6 = event["location"] if event["kind"] != "comm" else event["claimed_location"]
    if last_loc_q6 != q6["answer"]:
        problems.append(f"Q6: stored={q6['answer']} graph={last_loc_q6}")

    # Q7: motive flip. Re-derive the perturbed move's witness set.
    q7 = cf_entry["Q7"]
    flip_line = q7["perturbed_line"]
    target_event = find_event_at_line(events, flip_line)
    new_motive = q7["counterfactual_motive"]
    new_help = new_motive["help_target"]
    new_hide = new_motive["hide_target"]
    in_room_at_flip = in_room_at_line(events, agents, exit_step, flip_line)
    events_q7 = reparse_with_motive_flip(events, target_event, new_help, new_hide, in_room_at_flip)
    # Q7 asks about chain intent, not chain location.
    chain_set_q7 = set(q7["chain"])
    graph_q7 = None
    target_q7_event = next((e for e in events_q7 if e["line_index"] == flip_line), None)
    if target_q7_event is not None:
        if chain_set_q7.issubset(target_q7_event["witnesses"]):
            graph_q7 = {
                "help_target": target_q7_event["help_target"],
                "hide_target": target_q7_event["hide_target"],
            }
    if graph_q7 != q7["answer"]:
        problems.append(f"Q7: stored={q7['answer']} graph={graph_q7}")

    return problems


def check_q8(causal_entry, events, exit_step, belief, agents):
    """Q8: belief inertia (cause of agent's final belief)."""
    problems = []

    # Q8: walk events; find the LAST event whose effective witnesses
    # include the asked agent. That event is the agent's most recent
    # evidence — regardless of whether it changed their belief.
    q8 = causal_entry["Q8"]
    if q8.get("answer") is not None:
        asked = q8["agent"]
        cause_event = None
        for event in events:
            if asked in witnesses_for_event(event, exit_step):
                cause_event = event
        graph_q8_line = cause_event["line_index"] if cause_event is not None else None
        if graph_q8_line != q8["answer"]["line"]:
            problems.append(f"Q8: stored_line={q8['answer']['line']} graph_line={graph_q8_line}")

    return problems


def check_q9(knowledge_entry, events, exit_step, belief, agents):
    """Q9: common knowledge over an agent set (no perturbation)."""
    problems = []

    # Q9: CK(S) iff all agents in S share the same final belief, AND
    # there's an event whose effective witnesses ⊇ S yielding that location.
    q9 = knowledge_entry["Q9"]
    if q9.get("answer") is not None:
        agent_set = set(q9["agent_set"])
        final_belief = belief[-1]
        beliefs_in_set = {final_belief.get(a) for a in agent_set}
        if len(beliefs_in_set) == 1 and None not in beliefs_in_set:
            shared = next(iter(beliefs_in_set))
            anchor_loc = None
            for event in events:
                if agent_set.issubset(witnesses_for_event(event, exit_step)):
                    anchor_loc = event["location"] if event["kind"] != "comm" else event["claimed_location"]
            if anchor_loc == shared:
                graph_q9 = {"common_knowledge": True, "shared_location": shared}
            else:
                graph_q9 = {"common_knowledge": False, "shared_location": None}
        else:
            graph_q9 = {"common_knowledge": False, "shared_location": None}
        if graph_q9 != q9["answer"]:
            problems.append(f"Q9: stored={q9['answer']} graph={graph_q9}")

    return problems


def check_q10(q10_entry, events, exit_step, agents):
    """Q10: counterfactual chain belief under exit-time swap."""
    problems = []
    q10 = q10_entry["Q10"]
    swap_a = q10["swap_agent_A"]
    swap_b = q10["swap_agent_B"]
    new_exit = dict(exit_step)
    new_exit[swap_a], new_exit[swap_b] = exit_step[swap_b], exit_step[swap_a]
    modified_events, _ = reparse_events_with_modified_exits(events, new_exit, agents)

    chain_set = set(q10["chain"])
    last_location = None
    for event in modified_events:
        if chain_set.issubset(witnesses_for_event(event, new_exit)):
            last_location = event["location"] if event["kind"] != "comm" else event["claimed_location"]
    if last_location != q10["answer"]:
        problems.append(f"Q10: stored={q10['answer']} graph={last_location}")
    return problems


def check_q11_q13(knowledge_entry, events, exit_step, agents):
    """Q11: counterfactual exit + CK. Q13: counterfactual comm + CK."""
    problems = []

    # Q11: change one agent's exit_step, ask CK over a set.
    q11 = knowledge_entry["Q11"]
    perturbed_agent = q11["perturbed_agent"]
    new_exit_value = q11["perturbed_agent_counterfactual_exit"]
    new_exit_step = dict(exit_step)
    new_exit_step[perturbed_agent] = new_exit_value
    modified_events, _ = reparse_events_with_modified_exits(events, new_exit_step, agents)

    belief_q11 = build_graph(modified_events, new_exit_step, agents)
    final_q11 = belief_q11[-1]
    agent_set_q11 = set(q11["agent_set_S"])
    beliefs_q11 = {final_q11.get(a) for a in agent_set_q11}
    graph_q11 = {"common_knowledge": False, "shared_location": None}
    if len(beliefs_q11) == 1 and None not in beliefs_q11:
        shared = next(iter(beliefs_q11))
        anchor_loc = None
        for event in modified_events:
            if agent_set_q11.issubset(witnesses_for_event(event, new_exit_step)):
                anchor_loc = event["location"] if event["kind"] != "comm" else event["claimed_location"]
        if anchor_loc == shared:
            graph_q11 = {"common_knowledge": True, "shared_location": shared}
    if graph_q11 != q11["answer"]:
        problems.append(f"Q11: stored={q11['answer']} graph={graph_q11}")

    # Q13: swap a comm's claim, ask CK.
    q13 = knowledge_entry["Q13"]
    if q13.get("answer") is not None:
        swap_line = q13["perturbed_line"]
        new_claim = q13["counterfactual_claim"]
        modified_events_q13 = []
        for event in events:
            if event["line_index"] == swap_line and event["kind"] == "comm":
                new_event = dict(event)
                new_event["claimed_location"] = new_claim
                modified_events_q13.append(new_event)
            else:
                modified_events_q13.append(event)
        belief_q13 = build_graph(modified_events_q13, exit_step, agents)
        final_q13 = belief_q13[-1]
        agent_set_q13 = set(q13["agent_set_S"])
        beliefs_q13 = {final_q13.get(a) for a in agent_set_q13}
        graph_q13 = {"common_knowledge": False, "shared_location": None}
        if len(beliefs_q13) == 1 and None not in beliefs_q13:
            shared = next(iter(beliefs_q13))
            anchor_loc = None
            for event in modified_events_q13:
                if agent_set_q13.issubset(witnesses_for_event(event, exit_step)):
                    anchor_loc = event["location"] if event["kind"] != "comm" else event["claimed_location"]
            if anchor_loc == shared:
                graph_q13 = {"common_knowledge": True, "shared_location": shared}
        if graph_q13 != q13["answer"]:
            problems.append(f"Q13: stored={q13['answer']} graph={graph_q13}")

    return problems


# --------------------------------------------------------------------------
# Cross-check against all per-question files
# --------------------------------------------------------------------------

def main():
    stories = [json.loads(l) for l in open(STORIES_PATH)]
    higher_qa = [json.loads(l) for l in open(CORE_QUESTIONS_PATH)]
    counterfactual_qa = [json.loads(l) for l in open("/mnt/user-data/outputs/v11/data/counterfactual_beliefs_v10.jsonl")]
    causal_qa = [json.loads(l) for l in open("/mnt/user-data/outputs/v11/data/causal_beliefs_v10.jsonl")]
    knowledge_qa = [json.loads(l) for l in open("/mnt/user-data/outputs/v11/data/common_knowledge_v10.jsonl")]

    n_stories_with_mismatch = 0
    counts_per_question = {}

    for story, higher_entry, cf_entry, causal_entry, knowledge_entry in zip(
        stories, higher_qa, counterfactual_qa, causal_qa, knowledge_qa
    ):
        events, exit_step = parse_events(
            story["story"], story["object"], story["agents"]
        )
        belief = build_graph(events, exit_step, story["agents"])

        problems = []
        problems += check_q0_q3(higher_entry, events, exit_step, belief, story["agents"])
        problems += check_q5_q7(cf_entry, events, exit_step, belief, story["agents"])
        problems += check_q8(causal_entry, events, exit_step, belief, story["agents"])
        problems += check_q9(knowledge_entry, events, exit_step, belief, story["agents"])
        problems += check_q10(cf_entry, events, exit_step, story["agents"])
        problems += check_q11_q13(knowledge_entry, events, exit_step, story["agents"])

        if problems:
            n_stories_with_mismatch += 1
            for problem in problems:
                qid = problem.split(":")[0]
                counts_per_question[qid] = counts_per_question.get(qid, 0) + 1
            if n_stories_with_mismatch <= 5:
                print(f"\n--- Mismatch id={story['id']} ---")
                for problem in problems:
                    print(" ", problem)

    print(f"\nGraph verifier — {len(stories)} stories, {n_stories_with_mismatch} with mismatches")
    if counts_per_question:
        print("Mismatches per question:")
        for q in sorted(counts_per_question):
            print(f"  {q}: {counts_per_question[q]}")


if __name__ == "__main__":
    main()
