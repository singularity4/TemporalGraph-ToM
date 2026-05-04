"""
Common knowledge question generator (Q9, Q11, Q13) — derives ground truth from TGToM stories.

  Q9  — common knowledge over an agent set (no perturbation)
        "Is the location of the <object> common knowledge among {Y, Z, W}?"

  Q11 — counterfactual-exit CK
        "Suppose X had exited at line N. Would the location be common
         knowledge among {Y, Z, W}?"

  Q13 — counterfactual-comm CK
        "Suppose <speaker> had said the <object> is in the <claim> at line N.
         Would the location of the <object> then be common knowledge among
         {Y, Z, W}?"

Q9 has no perturbation; Q11 perturbs an exit time; Q13 perturbs a
communication's claim. All three share the same answer shape
{common_knowledge: bool, shared_location: str|null} and the same
common-knowledge definition (below).

================================================================================
INPUT (read-only):
  /mnt/user-data/outputs/v11/data/stories_v10.jsonl
  /mnt/user-data/outputs/v11/data/higher_order_beliefs_v10.jsonl

OUTPUT:
  /mnt/user-data/outputs/v11/data/common_knowledge_v10.jsonl
    one entry per story with Q9, Q11 and Q13.

================================================================================
COMMON KNOWLEDGE DEFINITION
================================================================================
For a set S, the location L is common knowledge among S iff BOTH:
  (a) every agent in S has L as their final belief, AND
  (b) the latest event whose witness set ⊇ S has location L.

================================================================================
Q11 — COUNTERFACTUAL EXIT + CK
================================================================================
"Suppose <X> had exited at line <N>. Would the location of the <object> be
 common knowledge among <Y, Z, W>?"

The perturbation: X's exit_step is replaced with a new line number N. We
re-simulate the world: at every event, agents present are determined by
the new exit_step values. For comms, the trust rule uses the new exit
ordering (listener trusts speaker iff speaker.new_exit > listener.new_exit).

Selection of N: pick a line that's earlier OR later than X's actual exit
by enough to cross at least one move event. This ensures the perturbation
actually changes what X witnessed. Bounded random search.

Targeted: prefer (X, N, S) where original-CK (over S) ≠ counterfactual-CK
(over S). Bounded search; fall back to random.

|S| = 3..min(N_agents - 1, 4) — set excludes X.

================================================================================
Q13 — COUNTERFACTUAL COMM CLAIM + CK
================================================================================
"Suppose <speaker> had said the <object> is in the <claim> at line N.
 Would the location of the <object> then be common knowledge among <Y, Z, W>?"

The perturbation: the comm at line N has its claimed_location swapped to <claim>.
Trust relations are unchanged (depend only on exit-time order). Listeners'
final beliefs are re-derived under the modified comm.

Selection: random comm + random alt_loc (≠ original) + random set S of
size 3..min(N_agents - 1, 4).
Targeted: prefer (comm, alt_loc, S) where original-CK ≠ counterfactual-CK.

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

STORIES_PATH = "/mnt/user-data/outputs/v11/data/stories_v10.jsonl"
CORE_QUESTIONS_PATH = "/mnt/user-data/outputs/v11/data/higher_order_beliefs_v10.jsonl"
OUT_PATH = "/mnt/user-data/outputs/v11/data/common_knowledge_v10.jsonl"

# Per-(question, story) seed offsets — see DETERMINISM section above.
Q9_SEED_OFFSET = 90000
Q11_SEED_OFFSET = 110000
Q13_SEED_OFFSET = 130000


def get_real_lines(story_text):
    out = []
    for raw in story_text.split("\n"):
        m = re.match(r"^(\d+)\s+(.*)$", raw)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


def get_event_lines(real_lines, obj):
    """Return list of (line_no, event_kind) for actual events in story order."""
    out = []
    for line_no, text in real_lines:
        if text.startswith(f"The {obj} is in the "):
            out.append((line_no, "place"))
        elif " moved the " in text and obj in text:
            out.append((line_no, "move"))
        elif " publicly claimed that " in text or " privately told " in text:
            out.append((line_no, "comm"))
    return out


def find_comm_line_by_index(real_lines, comm_event_index):
    seen = 0
    for line_no, text in real_lines:
        if (" publicly claimed that the " in text
                or " privately told " in text):
            if seen == comm_event_index:
                return line_no
            seen += 1
    return None


def first_order_belief_per_agent(events, agents):
    first_order_beliefs = {a: None for a in agents}
    for event in events:
        for a in event["witnesses"]:
            first_order_beliefs[a] = event["location"]
    return first_order_beliefs


def latest_event_for_set(events, agent_set):
    s = set(agent_set)
    last = None
    for event in events:
        if s.issubset(event["witnesses"]):
            last = event
    return last


def is_common_knowledge(events, agents, agent_set):
    """Strict CK: every agent in set has the shared anchor's location as
    their final belief."""
    first_order_beliefs = first_order_belief_per_agent(events, agents)
    latest = latest_event_for_set(events, agent_set)
    if latest is None:
        return False, None
    shared_loc = latest["location"]
    if all(first_order_beliefs.get(a) == shared_loc for a in agent_set):
        return True, shared_loc
    return False, None


def simulate_under_exit_change(events, agents, exit_step, agent_X, new_exit_X,
                                ev_lines):
    """Re-simulate events with X's exit_step replaced by new_exit_X.
    Returns the modified event list with witnesses recomputed."""
    new_exit = dict(exit_step)
    new_exit[agent_X] = new_exit_X

    modified = []
    assert len(events) == len(ev_lines)
    for event, (line_no, _kind) in zip(events, ev_lines):
        # Agent A is in room at this event iff new_exit[A] > line_no.
        present = {a for a in agents if new_exit[a] > line_no}

        if event["kind"] in ("place", "move"):
            new_ev = dict(event)
            if event["kind"] == "move":
                # Apply witness rule: hide_target does not witness.
                move_witnesses = set(present)
                hide_target = event.get("intent", {}).get("hide_target") if event.get("intent") else None
                if hide_target is not None:
                    move_witnesses.discard(hide_target)
                new_ev["witnesses"] = frozenset(move_witnesses)
            else:
                new_ev["witnesses"] = frozenset(present)
            modified.append(new_ev)
        else:  # comm
            speaker = event["speaker"]
            new_updated = []
            for L in event.get("listeners", []):
                if L == speaker:
                    continue
                if new_exit[speaker] > new_exit[L]:
                    new_updated.append(L)
            new_ev = dict(event)
            new_ev["updated_listeners"] = new_updated
            new_ev["witnesses"] = frozenset(new_updated)
            modified.append(new_ev)

    return modified


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
        ev_lines = get_event_lines(real_lines, obj)
        comm_events = [event for event in events if event["kind"] == "comm"]
        move_events = [event for event in events if event["kind"] == "move"]

        # Sort for canonical iteration order — see lie_pool note in gen_v10.py.
        used_containers_set = set()
        for event in events:
            if event["kind"] in ("place", "move"):
                used_containers_set.add(event["location"])
        for event in comm_events:
            used_containers_set.add(event["claimed_location"])
        used_containers = sorted(used_containers_set)

        # Recover exit_step
        exit_step = {}
        for line_no, text in real_lines:
            m = re.match(r"^(\S+) exited the \S+\.$", text)
            if m:
                exit_step[m.group(1)] = line_no
        for agent in agents:
            if agent not in exit_step:
                exit_step[agent] = 10**9

        # ----------------------- Q9 -----------------------
        # Common ground: is the location of the object common knowledge among
        # a randomly chosen subset of agents? No perturbation.
        rng_q9 = random.Random(Q9_SEED_OFFSET + s["id"])
        max_q9_set_size = min(n_agents, 5)
        q9_set_size = rng_q9.randint(3, max_q9_set_size)
        q9_agent_set = rng_q9.sample(agents, q9_set_size)

        q9_ck, q9_loc = is_common_knowledge(events, agents, q9_agent_set)
        q9_question = (
            f"Is the location of the {obj} common knowledge among "
            f"{', '.join(q9_agent_set[:-1])} and {q9_agent_set[-1]}? "
            f"Answer yes (with the location) or no."
        )
        q9_entry = {
            "question_type": "Q9",
            "subtype": "common_ground",
            "agent_set": q9_agent_set,
            "question": q9_question,
            "answer": {
                "common_knowledge": q9_ck,
                "shared_location": q9_loc if q9_ck else None,
            },
        }

        # ----------------------- Q11 -----------------------
        # Counterfactual exit + CK.
        rng_q11 = random.Random(Q11_SEED_OFFSET + s["id"])
        max_s_size = min(n_agents - 1, 4)
        s_size = rng_q11.randint(3, max(3, max_s_size))

        # Candidate exit_step values: any move event line, any comm line.
        # We want a counterfactual exit that crosses an event boundary.
        candidate_lines = sorted({line_no for line_no, _ in ev_lines})

        q11_chosen = None
        for _ in range(120):
            X = rng_q11.choice(agents)
            pool = [agent for agent in agents if agent != X]
            if len(pool) < s_size:
                continue
            S = rng_q11.sample(pool, s_size)

            # Pick a counterfactual exit line that differs from X's actual
            # exit and would cross at least one event.
            x_actual = exit_step[X]
            choices = [
                line_no + 1 for line_no in candidate_lines
                if (line_no + 1) != x_actual
                and (line_no + 1) >= 1
            ]
            if not choices:
                continue
            new_exit_X = rng_q11.choice(choices)
            if new_exit_X == x_actual:
                continue

            # Original CK over S
            orig_ck, orig_loc = is_common_knowledge(events, agents, S)
            # Counterfactual CK
            modified = simulate_under_exit_change(events, agents, exit_step,
                                                   X, new_exit_X, ev_lines)
            cf_ck, cf_loc = is_common_knowledge(modified, agents, S)

            if orig_ck != cf_ck:
                q11_chosen = (X, x_actual, new_exit_X, S, cf_ck, cf_loc, True)
                break

        if q11_chosen is None:
            # fallback: random
            X = rng_q11.choice(agents)
            pool = [agent for agent in agents if agent != X]
            S = rng_q11.sample(pool, min(s_size, len(pool)))
            x_actual = exit_step[X]
            choices = [
                line_no + 1 for line_no in candidate_lines
                if (line_no + 1) != x_actual and (line_no + 1) >= 1
            ]
            new_exit_X = rng_q11.choice(choices) if choices else x_actual
            modified = simulate_under_exit_change(events, agents, exit_step,
                                                   X, new_exit_X, ev_lines)
            cf_ck, cf_loc = is_common_knowledge(modified, agents, S)
            q11_chosen = (X, x_actual, new_exit_X, S, cf_ck, cf_loc, False)

        X, x_actual, new_exit_X, S, q11_ck, q11_loc, q11_targeted = q11_chosen
        s_phrase = (", ".join(S[:-1]) + " and " + S[-1]
                    if len(S) > 1 else S[0])
        q11_question = (
            f"Suppose {X} had exited at line {new_exit_X}. Would the "
            f"location of the {obj} then be common knowledge among "
            f"{s_phrase}? Answer yes (with the location) or no."
        )
        q11_entry = {
            "question_type": "Q11",
            "subtype": "ck_counterfactual_exit",
            "perturbed_agent": X,
            "perturbed_agent_original_exit": x_actual,
            "perturbed_agent_counterfactual_exit": new_exit_X,
            "agent_set_S": S,
            "targeted": q11_targeted,
            "question": q11_question,
            "answer": {
                "common_knowledge": q11_ck,
                "shared_location": q11_loc if q11_ck else None,
            },
        }

        # ----------------------- Q13 -----------------------
        # Counterfactual comm + CK.
        if not comm_events:
            q13_entry = {
                "question_type": "Q13",
                "subtype": "ck_counterfactual_comm",
                "question": None,
                "answer": None,
            }
        else:
            rng_q13 = random.Random(Q13_SEED_OFFSET + s["id"])
            max_s_size_q13 = min(n_agents - 1, 4)
            s13_size = rng_q13.randint(3, max(3, max_s_size_q13))
            q13_chosen = None
            for _ in range(80):
                # Pick target by index, not by `rng_q13.choice(comm_events)`,
                # because two comm events with identical content compare
                # equal as dicts; downstream `comm_events.index(...)` would
                # then return the wrong (earliest) match.
                target_idx = rng_q13.randint(0, len(comm_events) - 1)
                target_comm = comm_events[target_idx]
                speaker = target_comm["speaker"]
                original_claim = target_comm["claimed_location"]
                alt_pool = [c for c in used_containers if c != original_claim]  # already sorted
                if not alt_pool:
                    continue
                alt_loc = rng_q13.choice(alt_pool)
                if len(agents) < s13_size:
                    continue
                agent_set = rng_q13.sample(agents, s13_size)
                orig_ck, _ = is_common_knowledge(events, agents, agent_set)

                events_modified = []
                for event in events:
                    if event is target_comm:
                        ev_new = dict(event)
                        ev_new["location"] = alt_loc
                        ev_new["claimed_location"] = alt_loc
                        events_modified.append(ev_new)
                    else:
                        events_modified.append(event)
                cf_ck, cf_loc = is_common_knowledge(events_modified, agents, agent_set)

                if orig_ck != cf_ck:
                    line_no = find_comm_line_by_index(real_lines, target_idx)
                    q13_chosen = (target_comm, speaker, original_claim, alt_loc,
                                  agent_set, cf_ck, cf_loc, line_no, True)
                    break

            if q13_chosen is None:
                target_idx = rng_q13.randint(0, len(comm_events) - 1)
                target_comm = comm_events[target_idx]
                speaker = target_comm["speaker"]
                original_claim = target_comm["claimed_location"]
                alt_pool = [c for c in used_containers if c != original_claim]  # already sorted
                if not alt_pool:
                    q13_entry = {
                        "question_type": "Q13",
                        "subtype": "ck_counterfactual_comm",
                        "question": None,
                        "answer": None,
                    }
                    out.append({
                        "id": s["id"],
                        "Q9": q9_entry,
                        "Q11": q11_entry,
                        "Q13": q13_entry,
                    })
                    continue
                alt_loc = rng_q13.choice(alt_pool)
                agent_set = rng_q13.sample(agents, min(s13_size, len(agents)))
                events_modified = []
                for event in events:
                    if event is target_comm:
                        ev_new = dict(event)
                        ev_new["location"] = alt_loc
                        ev_new["claimed_location"] = alt_loc
                        events_modified.append(ev_new)
                    else:
                        events_modified.append(event)
                cf_ck, cf_loc = is_common_knowledge(events_modified, agents, agent_set)
                line_no = find_comm_line_by_index(real_lines, target_idx)
                q13_chosen = (target_comm, speaker, original_claim, alt_loc,
                              agent_set, cf_ck, cf_loc, line_no, False)

            (target_comm, speaker, original_claim, alt_loc, agent_set, cf_ck, cf_loc,
             line_no, q13_targeted) = q13_chosen
            s13_phrase = (", ".join(agent_set[:-1]) + " and " + agent_set[-1]
                          if len(agent_set) > 1 else agent_set[0])
            q13_question = (
                f"Suppose {speaker} had said the {obj} is in the {alt_loc} "
                f"at line {line_no}. Would the location of the {obj} then "
                f"be common knowledge among {s13_phrase}? Answer yes "
                f"(with the location) or no."
            )
            q13_entry = {
                "question_type": "Q13",
                "subtype": "ck_counterfactual_comm",
                "perturbed_line": line_no,
                "speaker": speaker,
                "original_claim": original_claim,
                "counterfactual_claim": alt_loc,
                "agent_set_S": agent_set,
                "targeted": q13_targeted,
                "question": q13_question,
                "answer": {
                    "common_knowledge": cf_ck,
                    "shared_location": cf_loc if cf_ck else None,
                },
            }

        out.append({
            "id": s["id"],
            "Q9": q9_entry,
            "Q11": q11_entry,
            "Q13": q13_entry,
        })

    with open(OUT_PATH, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"Wrote {len(out)} entries -> {OUT_PATH}")
    print()
    print("--- Sample (id=1) ---")
    print(json.dumps(out[1], indent=2))

    # Stats
    q11_targeted = sum(1 for o in out if o["Q11"].get("targeted"))
    q11_yes = sum(1 for o in out if o["Q11"]["answer"]
                  and o["Q11"]["answer"]["common_knowledge"])
    q13_targeted = sum(1 for o in out
                       if o["Q13"]["answer"] is not None
                       and o["Q13"].get("targeted"))
    q13_yes = sum(1 for o in out if o["Q13"]["answer"]
                  and o["Q13"]["answer"]["common_knowledge"])
    n = len(out)
    print(f"\nQ11: yes={q11_yes}/{n}, targeted={q11_targeted}/{n}")
    print(f"Q13: yes={q13_yes}/{n}, targeted={q13_targeted}/{n}")


if __name__ == "__main__":
    main()
