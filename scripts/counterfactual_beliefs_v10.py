"""
Counterfactual beliefs generator (Q5–Q7, Q10) — derives ground truth from TGToM stories.

================================================================================
SCRIPT DESCRIPTION
================================================================================
Reads stories_v10.jsonl + higher_order_beliefs_v10.jsonl. For each story,
re-derives the event sequence (via verify_v10.parse_and_recompute) and
produces ground truth and question text for four counterfactual question
types:

  Q5   first-order counterfactual belief — drop a move
  Q6   higher-order counterfactual belief — swap a comm's claimed location
  Q7   higher-order counterfactual intent — flip a move's motive
  Q10  higher-order counterfactual belief — swap two agents' exit times

All four use TARGETED SELECTION: configurations are searched so that the
perturbation actually changes the answer (avoiding degenerate cases). When
no qualifying configuration exists the script falls back to random and
marks `targeted: false`. See per-question comments below.

Question text never reveals the original perturbed value — the model must
read the indicated story line to recover it.

================================================================================
INPUT (read-only):
  /mnt/user-data/outputs/v10/stories_v10.jsonl
  /mnt/user-data/outputs/v10/higher_order_beliefs_v10.jsonl

OUTPUT:
  /mnt/user-data/outputs/v10/counterfactual_beliefs_v10.jsonl
    one entry per story; fields: id, Q5..Q7 each with question text,
    answer, and full perturbation metadata (perturbed_line,
    original_*, counterfactual_*, chain, chain_depth, targeted, ...).

================================================================================
QUESTION FAMILIES
================================================================================

Q5 — FIRST-ORDER COUNTERFACTUAL BELIEF  (drop a move)

  "Suppose <mover>'s move at line <N> had not happened. Where would
   <other_agent> now believe the <object> is?"

  Answer: re-simulate without that move event; read <other_agent>'s
  final belief.

  TARGETED selection: enumerate all (move, agent) pairs and keep
  only those where dropping the move changes the agent's final belief.
  Sample uniformly from this qualifying set. If empty, fall back
  to random selection and mark `targeted: false`.

  Note: question text intentionally does NOT reveal the original
  destination — bot must read line <N>.

Q6 — HIGHER-ORDER COUNTERFACTUAL BELIEF  (change a comm)

  "Suppose <speaker> had said the <object> is in the <alt_loc> at line
   <N>. What does <chain_phrase>?"
  Answer: re-simulate with the comm's claimed_location swapped to <alt_loc>;
  apply higher-order chain rule (latest event whose witnesses ⊇ chain
  set) on the modified events.

  Comm-event semantics: changing a comm's claimed_location does NOT change
  which agents trust the comm (trust depends only on speaker/listener
  exit-time order). The comm event's `witnesses` (= updated_listeners)
  stays the same; only `location` is swapped.

  TARGETED selection: bounded random search (up to 100 attempts) for a
  (comm, alt_loc, chain) such that the chain's higher-order belief with
  the swap differs from the chain's belief on unmodified events. Q6 has
  a low qualifying rate (~17/100 stories) because most chains are
  already determined by a later move event whose witnesses ⊇ chain set,
  so changing a comm's claim has no effect on the higher-order answer.
  When no qualifying config is found, fall back to random and mark
  `targeted: false`.

  Note: question text does NOT reveal the original claim — bot must
  read line <N>.

Q7 — HIGHER-ORDER COUNTERFACTUAL INTENT  (motive flip)
  "Suppose <mover>'s move at line <N> had been to <flipped_motive>.
   What does <chain_phrase> <mover>'s intent was?"

  The motive flip is a help↔hide direction swap with targets preserved:
    - help X (no hide) → hide from X (no help)
    - hide from Y (no help) → help Y (no hide)
    - help X and hide Y → help Y and hide X

  Answer: if every chain agent witnesses the move under the
  COUNTERFACTUAL motive, the SWAPPED intent pair. Else null.

  TARGETED selection: pick a move whose COUNTERFACTUAL witness set
  (i.e., witnesses after the motive flip — original_witnesses ∪
  {original_hide} ∖ {new_hide}, excluding mover) has ≥ chain_depth
  members, then sample the chain from that set. This guarantees a
  non-null answer because the chain witnesses the counterfactual
  move. If no move qualifies at chain_depth, fall back to using
  the move with the largest counterfactual witness pool and a
  shorter chain (still non-null at lower depth).

  Note: question text does NOT reveal the original motive — bot must
  read line <N>.

================================================================================
GROUND-TRUTH DERIVATION
================================================================================
- Re-derive events from story text via verify_v10.parse_and_recompute.
  This gives a canonical event timeline (place + moves + comms) with
  per-event witness sets, so we never trust the generator's stored
  metadata.
- Q5: build modified events by removing the target move; re-run
  per-event witness updates to get final beliefs.
- Q6: build modified events by replacing the target comm's
  location/claimed_location with alt_loc; re-run higher-order chain rule.
  Comm witnesses stay the same — see semantics note above.
- Q7: build modified events by swapping help_target↔hide_target on
  the target move; locate the target move by (mover, occurrence) and
  read off its swapped intent under the chain rule.

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
OUT_PATH = "/mnt/user-data/outputs/v10/counterfactual_beliefs_v10.jsonl"

# Per-(question, story) seed offsets — see DETERMINISM section above.
Q5_SEED_OFFSET = 50000
Q6_SEED_OFFSET = 60000
Q7_SEED_OFFSET = 70000
Q10_SEED_OFFSET = 100000


def get_real_lines(story_text):
    out = []
    for raw in story_text.split("\n"):
        m = re.match(r"^(\d+)\s+(.*)$", raw)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


def find_move_line(real_lines, mover, target_occurrence):
    """Find the line number of mover's `target_occurrence`-th move (0-indexed)."""
    occ = 0
    for line_no, text in real_lines:
        if text.startswith(f"{mover} moved the "):
            if occ == target_occurrence:
                return line_no
            occ += 1
    return None


def find_comm_line(real_lines, speaker, target_occurrence):
    """Find the line number of speaker's `target_occurrence`-th comm (0-indexed)."""
    occ = 0
    for line_no, text in real_lines:
        if (text.startswith(f"{speaker} publicly claimed that the ") or
            text.startswith(f"{speaker} privately told ")):
            if occ == target_occurrence:
                return line_no
            occ += 1
    return None


def resimulate_first_order(events_modified, agents):
    """Walk the modified events and return per-agent final belief.
    Each event's witnesses set tells us who updates."""
    first_order_beliefs = {a: None for a in agents}
    for event in events_modified:
        for a in event["witnesses"]:
            first_order_beliefs[a] = event["location"]
    return first_order_beliefs


def chain_belief(events_modified, chain):
    """Latest event whose witnesses ⊇ chain set."""
    chain_set = set(chain)
    last = None
    for event in events_modified:
        if chain_set.issubset(event["witnesses"]):
            last = event["location"]
    return last


def chain_belief_intent(events_modified, chain, target_actor, target_occurrence):
    """Among modified events, find the move by target_actor whose
    occurrence==target_occurrence, then return its intent IF the chain
    is a subset of its witnesses."""
    chain_set = set(chain)
    seen = 0
    for event in events_modified:
        if event["kind"] == "move" and event["intent"]["actor"] == target_actor:
            if seen == target_occurrence:
                if chain_set.issubset(event["witnesses"]):
                    return {
                        "help_target": event["intent"]["help_target"],
                        "hide_target": event["intent"]["hide_target"],
                    }
                return None
            seen += 1
    return None


# ----- Small helpers shared across Q6, Q7, Q10 -----

def build_chain_phrase(chain):
    """Build the natural-language chain phrase used in Q6/Q7/Q10 questions:
    'A think B thinks C thinks ...' (first verb 'think', then 'thinks')."""
    if not chain:
        return ""
    parts = [f"{chain[0]} think"]
    for agent in chain[1:]:
        parts.append(f"{agent} thinks")
    return " ".join(parts)


def find_speaker_comm_occurrence(comm_events, target_comm):
    """Return the occurrence index (0-based) of target_comm among its
    speaker's comms in comm_events. Used for line-number lookup."""
    speaker = target_comm["speaker"]
    occurrence = 0
    for event in comm_events:
        if event["speaker"] == speaker:
            if event is target_comm:
                return occurrence
            occurrence += 1
    return None


def replace_comm_claim(events, target_comm, new_claim):
    """Return a new events list where target_comm has its location and
    claimed_location replaced by new_claim (other events unchanged)."""
    out = []
    for event in events:
        if event is target_comm:
            ev_new = dict(event)
            ev_new["location"] = new_claim
            ev_new["claimed_location"] = new_claim
            out.append(ev_new)
        else:
            out.append(event)
    return out


def counterfactual_motive_witnesses(move_event, mover, original_hide, new_hide):
    """Witness set under a motive flip on `move_event`: the old witness
    set with original_hide added back (no longer hide-target) and
    new_hide removed (now hide-target). Mover is excluded so chains
    drawn from this set are valid for chain-witnessing checks."""
    cf_witnesses = set(move_event["witnesses"])
    if original_hide is not None:
        cf_witnesses.add(original_hide)
    if new_hide is not None:
        cf_witnesses.discard(new_hide)
    cf_witnesses.discard(mover)
    return cf_witnesses


# ----- Q10 helpers (exit-swap simulation) -----

def get_event_lines(real_lines, obj):
    """Return list of (line_no, event_kind) for actual events in the story.
    Used to map events to their line numbers for the exit-swap simulation.
    """
    out = []
    for line_no, text in real_lines:
        if text.startswith(f"The {obj} is in the "):
            out.append((line_no, "place"))
        elif " moved the " in text and obj in text:
            out.append((line_no, "move"))
        elif " publicly claimed that " in text or " privately told " in text:
            out.append((line_no, "comm"))
    return out


def simulate_under_exit_swap(events, agents, exit_step, swap_a, swap_b, ev_lines):
    """Re-simulate under a swap of swap_a's and swap_b's exit_step.

    Returns (modified_events, new_exit_step). Each modified event has its
    `witnesses` recomputed under the new exit ordering. For comms,
    witnesses (= updated listeners) is recomputed using the new trust rule.
    """
    new_exit_step = dict(exit_step)
    new_exit_step[swap_a], new_exit_step[swap_b] = exit_step[swap_b], exit_step[swap_a]

    modified = []
    assert len(events) == len(ev_lines), (
        f"event count mismatch: {len(events)} vs {len(ev_lines)}"
    )

    for event, (line_no, _kind) in zip(events, ev_lines):
        present = {a for a in agents if new_exit_step[a] > line_no}

        if event["kind"] == "place":
            new_ev = dict(event)
            new_ev["witnesses"] = frozenset(present)
            modified.append(new_ev)
        elif event["kind"] == "move":
            # Apply witness rule: hide_target does not witness.
            move_witnesses = set(present)
            hide_target = event.get("intent", {}).get("hide_target") if event.get("intent") else None
            if hide_target is not None:
                move_witnesses.discard(hide_target)
            new_ev = dict(event)
            new_ev["witnesses"] = frozenset(move_witnesses)
            modified.append(new_ev)
        elif event["kind"] == "comm":
            # Apply new trust rule with swapped exit times.
            speaker = event["speaker"]
            new_updated = []
            for listener in event.get("listeners", []):
                if listener == speaker:
                    continue
                if new_exit_step[speaker] > new_exit_step[listener]:
                    new_updated.append(listener)
            new_ev = dict(event)
            new_ev["updated_listeners"] = new_updated
            new_ev["witnesses"] = frozenset(new_updated)
            modified.append(new_ev)

    return modified, new_exit_step


def main():
    stories = [json.loads(l) for l in open(STORIES_PATH)]
    qa_core = [json.loads(l) for l in open(CORE_QUESTIONS_PATH)]

    # Merge each (story, qa_core) into a single dict for downstream code
    # that expects bundled story metadata (object, agents) plus core-question
    # ground truth.
    answers = []
    for s, qa in zip(stories, qa_core):
        answers.append({
            "id": s["id"],
            "object": s["object"],
            "agents": s["agents"],
            "n_agents": qa["n_agents"],
            "actual_location": qa["actual_location"],
            "first_order_beliefs": {
                q["agent"]: q["answer"] for q in qa["Q1"]["questions"]
            },
            "higher_order_location": {
                "chain": qa["Q2"]["chain"],
                "answer": qa["Q2"]["answer"],
            },
            "higher_order_intent": {
                "chain": qa["Q3"]["chain"],
                "target_actor": qa["Q3"]["target_actor"],
                "target_occurrence": qa["Q3"]["target_occurrence"],
                "answer": qa["Q3"]["answer"],
            },
            "comm_log": qa["comm_log"],
        })

    out = []
    for s, a in zip(stories, answers):
        obj = a["object"]
        agents = a["agents"]
        n_agents = a["n_agents"]
        chain_depth = n_agents - 1
        story_text = s["story"]

        # Re-derive event timeline from text
        loc, first_order_beliefs, events = parse_and_recompute(story_text, obj, agents)
        real_lines = get_real_lines(story_text)

        move_events = [event for event in events if event["kind"] == "move"]
        comm_events = [event for event in events if event["kind"] == "comm"]

        # ------------------------- Q5 -------------------------
        # TARGETED SELECTION : pick (move, asked_agent) such that
        # dropping the move actually changes the asked_agent's final belief.
        # Strategy: enumerate all (move, agent) pairs; keep only those where
        # original belief != belief after dropping the move; sample uniformly
        # from the qualifying set. Fall back to random if none qualify.
        rng_q5 = random.Random(Q5_SEED_OFFSET + s["id"])
        original_fo = resimulate_first_order(events, agents)
        q5_candidates = []  # list of (move_event, agent)
        for mv in move_events:
            mover_of_mv = mv["intent"]["actor"]
            events_without_mv = [event for event in events if event is not mv]
            fo_without = resimulate_first_order(events_without_mv, agents)
            for agent in agents:
                if agent == mover_of_mv:
                    continue
                if fo_without.get(agent) != original_fo.get(agent):
                    q5_candidates.append((mv, agent))

        if q5_candidates:
            target_move, q5_agent = rng_q5.choice(q5_candidates)
            q5_targeted = True
        elif move_events:
            target_move = rng_q5.choice(move_events)
            mover_of_target = target_move["intent"]["actor"]
            q5_agent = rng_q5.choice([a for a in agents if a != mover_of_target])
            q5_targeted = False
        else:
            target_move = None

        if target_move is not None:
            mover = target_move["intent"]["actor"]
            destination = target_move["location"]
            occurrence = target_move["intent"]["occurrence"]
            line_no = find_move_line(real_lines, mover, occurrence)
            events_modified = [event for event in events if event is not target_move]
            fo_modified = resimulate_first_order(events_modified, agents)
            q5_answer = fo_modified.get(q5_agent)
            q5_question = (
                f"Suppose {mover}'s move at line {line_no} had not happened. "
                f"Where would {q5_agent} now believe the {obj} is?"
            )
            q5_entry = {
                "question_type": "Q5",
                "subtype": "drop_move_first_order",
                "perturbed_line": line_no,
                "perturbed_mover": mover,
                "perturbed_dest": destination,
                "perturbed_occurrence": occurrence,
                "asked_about": q5_agent,
                "targeted": q5_targeted,
                "question": q5_question,
                "answer": q5_answer,
            }
        else:
            q5_entry = {
                "question_type": "Q5",
                "subtype": "drop_move_first_order",
                "question": None,
                "answer": None,
            }

        # ------------------------- Q6 -------------------------
        # Higher-order counterfactual belief: change a comm's claimed_location;
        # ask N-1 chain.
        # TARGETED SELECTION : pick (comm, alt_loc, chain) such that the
        # chain's higher-order belief actually changes when alt_loc is
        # substituted. Without this constraint, ~34% of Q6s were degenerate.
        # Strategy: enumerate viable (comm, alt_loc) pairs by checking that
        # the comm has at least one updated_listener; then sample a chain
        # uniformly from N-1-subsets and check that swapping alt_loc actually
        # changes chain_belief. Use the first qualifying combination after
        # bounded random search.
        # Sort for canonical iteration order — see lie_pool note in gen_v10.py.
        used_containers_set = set()
        for event in events:
            if event["kind"] in ("place", "move"):
                used_containers_set.add(event["location"])
        for event in comm_events:
            used_containers_set.add(event["claimed_location"])
        used_containers = sorted(used_containers_set)

        q6_entry = None
        if comm_events:
            rng_q6 = random.Random(Q6_SEED_OFFSET + s["id"])
            # Compute original chain answers for caching: try multiple random
            # combinations and pick the first one where the perturbation
            # actually changes the answer. Bound at 100 attempts.
            attempts = 0
            max_attempts = 100
            while attempts < max_attempts:
                attempts += 1
                target_comm_try = rng_q6.choice(comm_events)
                # Fast filter: skip if no listener was updated
                if not target_comm_try.get("updated_listeners") and not target_comm_try.get("witnesses"):
                    # comm has empty witness set — perturbation can never propagate
                    continue
                speaker = target_comm_try["speaker"]
                original_claim = target_comm_try["claimed_location"]
                alt_pool = sorted(c for c in used_containers if c != original_claim)
                if not alt_pool:
                    continue
                alt_loc = rng_q6.choice(alt_pool)
                chain_try = rng_q6.sample(agents, chain_depth)

                # Compute original chain answer (with original comm)
                orig_chain_answer = chain_belief(events, chain_try)

                # Compute counterfactual chain answer
                events_modified = replace_comm_claim(events, target_comm_try, alt_loc)
                cf_chain_answer = chain_belief(events_modified, chain_try)

                if cf_chain_answer != orig_chain_answer:
                    # qualifying: the perturbation matters for this chain
                    target_comm = target_comm_try
                    q6_chain = chain_try
                    q6_answer = cf_chain_answer
                    comm_occ = find_speaker_comm_occurrence(comm_events, target_comm)
                    line_no = find_comm_line(real_lines, speaker, comm_occ)

                    chain_phrase = build_chain_phrase(q6_chain)
                    q6_question = (
                        f"Suppose {speaker} had said the {obj} is in the "
                        f"{alt_loc} at line {line_no}. "
                        f"What does {chain_phrase}?"
                    )
                    q6_entry = {
                        "question_type": "Q6",
                        "subtype": "comm_swap_higher_order",
                        "perturbed_line": line_no,
                        "speaker": speaker,
                        "original_claim": original_claim,
                        "counterfactual_claim": alt_loc,
                        "chain": q6_chain,
                        "chain_depth": chain_depth,
                        "targeted": True,
                        "question": q6_question,
                        "answer": q6_answer,
                    }
                    break

            if q6_entry is None:
                # Fallback: random pick (some stories may have no qualifying combo)
                target_comm = rng_q6.choice(comm_events)
                speaker = target_comm["speaker"]
                original_claim = target_comm["claimed_location"]
                alt_pool = sorted(c for c in used_containers if c != original_claim)
                if alt_pool:
                    alt_loc = rng_q6.choice(alt_pool)
                    comm_occ = find_speaker_comm_occurrence(comm_events, target_comm)
                    line_no = find_comm_line(real_lines, speaker, comm_occ)
                    events_modified = replace_comm_claim(events, target_comm, alt_loc)
                    q6_chain = rng_q6.sample(agents, chain_depth)
                    q6_answer = chain_belief(events_modified, q6_chain)
                    chain_phrase = build_chain_phrase(q6_chain)
                    q6_question = (
                        f"Suppose {speaker} had said the {obj} is in the "
                        f"{alt_loc} at line {line_no}. "
                        f"What does {chain_phrase}?"
                    )
                    q6_entry = {
                        "question_type": "Q6",
                        "subtype": "comm_swap_higher_order",
                        "perturbed_line": line_no,
                        "speaker": speaker,
                        "original_claim": original_claim,
                        "counterfactual_claim": alt_loc,
                        "chain": q6_chain,
                        "chain_depth": chain_depth,
                        "targeted": False,
                        "question": q6_question,
                        "answer": q6_answer,
                    }
                else:
                    q6_entry = {
                        "question_type": "Q6",
                        "subtype": "comm_swap_higher_order",
                        "question": None,
                        "answer": None,
                    }
        else:
            q6_entry = {
                "question_type": "Q6",
                "subtype": "comm_swap_higher_order",
                "question": None,
                "answer": None,
            }

        # ------------------------- Q7 -------------------------
        # Higher-order counterfactual intent: flip help↔hide on a move.
        # TARGETED SELECTION : pick (move, chain) such that the answer
        # is non-null. With random selection ~54% were null (chain didn't all
        # witness the move). For non-null under the counterfactual motive
        # flip, the chain must be a subset of the COUNTERFACTUAL witnesses
        # (witnesses after recomputing under the flipped motive — i.e., with
        # the new hide_target removed and the original hide_target restored).
        # Strategy: enumerate moves; for each, compute the counterfactual
        # witness set excluding the mover. If this set has ≥chain_depth,
        # the chain can be sampled from it. Pick uniformly from qualifying
        # moves.
        q7_qualifying = []
        for mv in move_events:
            mv_mover = mv["intent"]["actor"]
            mv_orig_help = mv["intent"]["help_target"]
            mv_orig_hide = mv["intent"]["hide_target"]
            mv_new_hide = mv_orig_help  # after motive flip
            cf_witnesses = counterfactual_motive_witnesses(
                mv, mv_mover, mv_orig_hide, mv_new_hide
            )
            if len(cf_witnesses) >= chain_depth:
                q7_qualifying.append(mv)

        if q7_qualifying:
            rng_q7 = random.Random(Q7_SEED_OFFSET + s["id"])
            target_move_q7 = rng_q7.choice(q7_qualifying)
            mover = target_move_q7["intent"]["actor"]
            destination = target_move_q7["location"]
            occurrence = target_move_q7["intent"]["occurrence"]
            line_no = find_move_line(real_lines, mover, occurrence)
            original_help = target_move_q7["intent"]["help_target"]
            original_hide = target_move_q7["intent"]["hide_target"]
            new_help = original_hide
            new_hide = original_help

            new_motive_parts = []
            if new_help is not None:
                new_motive_parts.append(f"help {new_help} find it later")
            if new_hide is not None:
                new_motive_parts.append(f"hide it from {new_hide}")
            new_motive_str = " and ".join(new_motive_parts) if new_motive_parts else "no stated motive"

            # Sample chain from witnesses-under-counterfactual (excluding mover
            # AND the new hide_target, since the hide_target won't witness the
            # move under the flipped motive). This guarantees the chain
            # actually witnesses the counterfactual move.
            counterfactual_witnesses = counterfactual_motive_witnesses(
                target_move_q7, mover, original_hide, new_hide
            )
            witness_pool = sorted(counterfactual_witnesses)
            if len(witness_pool) < chain_depth:
                # Cannot sample a chain of the required depth — fall through
                # to a smaller chain (still witnesses the counterfactual move).
                if not witness_pool:
                    q7_chain = []
                else:
                    q7_chain = rng_q7.sample(witness_pool, len(witness_pool))
            else:
                q7_chain = rng_q7.sample(witness_pool, chain_depth)

            events_modified = []
            for event in events:
                if event is target_move_q7:
                    ev_new = dict(event)
                    ev_new["intent"] = dict(event["intent"])
                    ev_new["intent"]["help_target"] = new_help
                    ev_new["intent"]["hide_target"] = new_hide
                    # Witness rule: hide_target does not witness. Since the
                    # hide_target changed (original_hide → new_hide), the witness
                    # set must be recomputed. Old witnesses were
                    # in_room \ {original_hide}; new witnesses are
                    # in_room \ {new_hide} = (old ∪ {original_hide}) \ {new_hide}.
                    new_witnesses = set(event["witnesses"])
                    if original_hide is not None:
                        new_witnesses.add(original_hide)
                    if new_hide is not None:
                        new_witnesses.discard(new_hide)
                    ev_new["witnesses"] = frozenset(new_witnesses)
                    events_modified.append(ev_new)
                else:
                    events_modified.append(event)

            q7_answer = chain_belief_intent(events_modified, q7_chain, mover, occurrence)
            chain_phrase = build_chain_phrase(q7_chain)
            q7_question = (
                f"Suppose {mover}'s move at line {line_no} had been to "
                f"{new_motive_str}. "
                f"What does {chain_phrase} {mover}'s intent was?"
            )
            q7_entry = {
                "question_type": "Q7",
                "subtype": "motive_flip_higher_order",
                "perturbed_line": line_no,
                "perturbed_mover": mover,
                "perturbed_occurrence": occurrence,
                "original_motive": {"help_target": original_help, "hide_target": original_hide},
                "counterfactual_motive": {"help_target": new_help, "hide_target": new_hide},
                "chain": q7_chain,
                "chain_depth": chain_depth,
                "targeted": True,
                "question": q7_question,
                "answer": q7_answer,
            }
        elif move_events:
            # No move has ≥chain_depth counterfactual-witnesses. Find the move
            # with the largest counterfactual witness pool and use a shorter
            # chain (still guarantees non-null but at lower depth).
            rng_q7 = random.Random(Q7_SEED_OFFSET + s["id"])
            best_move = None
            best_pool = []
            for mv in move_events:
                mv_mover = mv["intent"]["actor"]
                mv_orig_help = mv["intent"]["help_target"]
                mv_orig_hide = mv["intent"]["hide_target"]
                mv_new_hide = mv_orig_help
                cf_witnesses = set(mv["witnesses"])
                if mv_orig_hide is not None:
                    cf_witnesses.add(mv_orig_hide)
                if mv_new_hide is not None:
                    cf_witnesses.discard(mv_new_hide)
                cf_witnesses.discard(mv_mover)
                if len(cf_witnesses) > len(best_pool):
                    best_move = mv
                    best_pool = sorted(cf_witnesses)

            if best_move is not None and len(best_pool) >= 2:
                # Use shorter chain matching the available pool size.
                target_move_q7 = best_move
                mover = target_move_q7["intent"]["actor"]
                destination = target_move_q7["location"]
                occurrence = target_move_q7["intent"]["occurrence"]
                line_no = find_move_line(real_lines, mover, occurrence)
                original_help = target_move_q7["intent"]["help_target"]
                original_hide = target_move_q7["intent"]["hide_target"]
                new_help = original_hide
                new_hide = original_help

                new_motive_parts = []
                if new_help is not None:
                    new_motive_parts.append(f"help {new_help} find it later")
                if new_hide is not None:
                    new_motive_parts.append(f"hide it from {new_hide}")
                new_motive_str = " and ".join(new_motive_parts) if new_motive_parts else "no stated motive"

                short_depth = len(best_pool)
                q7_chain = rng_q7.sample(best_pool, short_depth)
            else:
                # Pathological: not enough witnesses. Fall back to random
                # with possibly null answer.
                target_move_q7 = rng_q7.choice(move_events)
                mover = target_move_q7["intent"]["actor"]
                destination = target_move_q7["location"]
                occurrence = target_move_q7["intent"]["occurrence"]
                line_no = find_move_line(real_lines, mover, occurrence)
                original_help = target_move_q7["intent"]["help_target"]
                original_hide = target_move_q7["intent"]["hide_target"]
                new_help = original_hide
                new_hide = original_help

                new_motive_parts = []
                if new_help is not None:
                    new_motive_parts.append(f"help {new_help} find it later")
                if new_hide is not None:
                    new_motive_parts.append(f"hide it from {new_hide}")
                new_motive_str = " and ".join(new_motive_parts) if new_motive_parts else "no stated motive"

                chain_pool = [agent for agent in agents if agent != mover]
                depth = min(chain_depth, len(chain_pool))
                q7_chain = rng_q7.sample(chain_pool, depth)

            events_modified = []
            for event in events:
                if event is target_move_q7:
                    ev_new = dict(event)
                    ev_new["intent"] = dict(event["intent"])
                    ev_new["intent"]["help_target"] = new_help
                    ev_new["intent"]["hide_target"] = new_hide
                    # Witness rule: hide_target does not witness. Since the
                    # hide_target changed (original_hide → new_hide), the witness
                    # set must be recomputed. Old witnesses were
                    # in_room \ {original_hide}; new witnesses are
                    # in_room \ {new_hide} = (old ∪ {original_hide}) \ {new_hide}.
                    new_witnesses = set(event["witnesses"])
                    if original_hide is not None:
                        new_witnesses.add(original_hide)
                    if new_hide is not None:
                        new_witnesses.discard(new_hide)
                    ev_new["witnesses"] = frozenset(new_witnesses)
                    events_modified.append(ev_new)
                else:
                    events_modified.append(event)

            q7_answer = chain_belief_intent(events_modified, q7_chain, mover, occurrence)
            chain_phrase = build_chain_phrase(q7_chain)
            q7_question = (
                f"Suppose {mover}'s move at line {line_no} had been to "
                f"{new_motive_str}. "
                f"What does {chain_phrase} {mover}'s intent was?"
            )
            q7_entry = {
                "question_type": "Q7",
                "subtype": "motive_flip_higher_order",
                "perturbed_line": line_no,
                "perturbed_mover": mover,
                "perturbed_occurrence": occurrence,
                "original_motive": {"help_target": original_help, "hide_target": original_hide},
                "counterfactual_motive": {"help_target": new_help, "hide_target": new_hide},
                "chain": q7_chain,
                "chain_depth": len(q7_chain),
                "targeted": False,
                "question": q7_question,
                "answer": q7_answer,
            }
        else:
            q7_entry = {
                "question_type": "Q7",
                "subtype": "motive_flip_higher_order",
                "question": None,
                "answer": None,
            }

        # ------------------------- Q10 ------------------------
        # Higher-order counterfactual belief — swap two agents' exit times.
        # TARGETED SELECTION: try (A, B, chain) where the swap actually
        # changes the chain's higher-order belief; fall back to random.
        ev_lines = get_event_lines(real_lines, obj)

        # Recover exit_step from the story text.
        exit_step = {}
        for line_no, text in real_lines:
            m_exit = re.match(r"^(\S+) exited the \S+\.$", text)
            if m_exit:
                exit_step[m_exit.group(1)] = line_no
        for agent in agents:
            if agent not in exit_step:
                exit_step[agent] = 10**9  # never explicitly exited

        # For Q10 swaps, both agents need explicit exit lines so the
        # perturbation is a real swap of two known points.
        exit_agents = [agent for agent in agents if exit_step[agent] != 10**9]
        if len(exit_agents) < 2:
            q10_entry = {
                "question_type": "Q10",
                "subtype": "exit_swap_higher_order",
                "question": None,
                "answer": None,
                "targeted": False,
            }
        else:
            rng_q10 = random.Random(Q10_SEED_OFFSET + s["id"])
            q10_attempts = 0
            q10_max_attempts = 100
            q10_chosen = None

            while q10_attempts < q10_max_attempts:
                q10_attempts += 1
                A, B = rng_q10.sample(exit_agents, 2)
                chain_try = rng_q10.sample(agents, chain_depth)
                orig_q10_answer = chain_belief(events, chain_try)
                modified_q10, new_exit = simulate_under_exit_swap(
                    events, agents, exit_step, A, B, ev_lines
                )
                cf_q10_answer = chain_belief(modified_q10, chain_try)
                if cf_q10_answer != orig_q10_answer:
                    q10_chosen = (A, B, chain_try, cf_q10_answer)
                    break

            q10_targeted = q10_chosen is not None
            if q10_chosen is None:
                # Fallback: random pick.
                A, B = rng_q10.sample(exit_agents, 2)
                chain_try = rng_q10.sample(agents, chain_depth)
                modified_q10, _ = simulate_under_exit_swap(
                    events, agents, exit_step, A, B, ev_lines
                )
                cf_q10_answer = chain_belief(modified_q10, chain_try)
                q10_chosen = (A, B, chain_try, cf_q10_answer)

            A, B, q10_chain, q10_answer = q10_chosen
            line_A_old = exit_step[A]
            line_B_old = exit_step[B]
            line_A_new = line_B_old
            line_B_new = line_A_old

            q10_chain_phrase = build_chain_phrase(q10_chain)

            q10_question = (
                f"Suppose {A} had exited at line {line_A_new} and {B} had exited "
                f"at line {line_B_new} (their exit times swapped). "
                f"What does {q10_chain_phrase}?"
            )

            q10_entry = {
                "question_type": "Q10",
                "subtype": "exit_swap_higher_order",
                "swap_agent_A": A,
                "swap_agent_A_original_line": line_A_old,
                "swap_agent_A_new_line": line_A_new,
                "swap_agent_B": B,
                "swap_agent_B_original_line": line_B_old,
                "swap_agent_B_new_line": line_B_new,
                "chain": q10_chain,
                "chain_depth": chain_depth,
                "targeted": q10_targeted,
                "question": q10_question,
                "answer": q10_answer,
            }

        out.append({
            "id": s["id"],
            "Q5": q5_entry,
            "Q6": q6_entry,
            "Q7": q7_entry,
            "Q10": q10_entry,
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
