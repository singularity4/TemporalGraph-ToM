"""
Story generator for the TemporalGraph-ToM benchmark.

================================================================================
SCRIPT DESCRIPTION
================================================================================
Generates multi-agent stories with controlled higher-order belief attribution.
Output: stories_v10.jsonl with one JSON object per line: {id, story, agents, object}.
 
Ground truth is derived from the story text by other scripts.

Story structure (in-room phase, communication phase, reconvene), the witnessing rule, the belief propagation trust rule, sampled variables, and design parameters
are documented in docs/design.md.

See docs/changelog.md for version history.

================================================================================
INVARIANTS
================================================================================
- Distractors never reference the object. 
- Communications are real events that reference the object.
- Communication speakers and listeners are distinct.
- False content is drawn from containers used in the story, ≠ the true
  current location at communication time.
- Determinism: same SEED produces byte-identical output across runs.
- The forward verifier (verify_v10.py) re-derives all ground truth from text 
  alone, providing an independent check on the generator.
- The graph verifier implements Temporal Belief Graph (TBG) (graph_v10.py) 
  which re-derives the ground truth, providing a 2nd independent check on the generator.
"""

import json
import os
import random

# ============================================================================
# DESIGN PARAMETERS for controlling difficulty / story variation
# ============================================================================
# These constants centralize the parameters described in the docstring.
# Adjusting them changes the difficulty profile of the generated dataset.

SEED = 0  # master seed; same SEED → byte-identical output

# Story scale
N_RANGE = (6, 8)                 # number of agents per story (uniform)
N_DISTRACTORS_RANGE = (4, 8)     # noise sentences per story (uniform)
N_COMMUNICATIONS_RANGE = (1, 3)  # communications per story (uniform)

# Agent dynamics
P_MOVER = 0.6                    # per-agent prob. of being a mover
MOVE_COUNT_WEIGHTS = (3, 1)      # weights for [1 move, 2 moves] per mover

# Motive composition is uniform when ≥2 candidate targets available.
# The hide-target witness rule excludes the hide-target from witnessing. 

MOTIVE_COMPOSITIONS = ("help_only", "hide_only", "both")

# Communication phase
P_PUBLIC_COMM = 0.5              # public vs private claim
P_TRUTHFUL_COMM = 0.5            # truthful vs lie

# Distractors
P_SOFT_DISTRACTOR = 0.5          # soft (no container) vs hard (with container)


# ============================================================================
# ENTITY POOLS — namespaces sampled per story
# ============================================================================
AGENTS_POOL = [
    "Mila", "Ava", "Emily", "Evelyn", "Jacob", "Liam", "Sophia",
    "Noah", "Olivia", "Ethan", "Isabella", "Mason", "Charlotte",
    "Lucas", "Amelia", "Logan", "Harper", "Aiden", "Ella", "Benjamin",
    "Mia", "Oliver", "Aria", "Elijah", "Scarlett", "James", "Chloe",
    "Henry", "Lily", "Daniel",
]

ROOMS_POOL = [
    "front_yard", "kitchen", "garden", "living_room", "playroom",
    "basement", "attic", "garage", "porch", "study", "bedroom",
    "hallway", "den", "library", "sunroom", "office", "cellar",
    "patio", "loft", "workshop", "pantry", "nursery",
]

OBJECTS_POOL = [
    "watermelon", "apple", "book", "lamp", "vase", "kettle",
    "scarf", "wallet", "umbrella", "notebook", "candle", "mug",
    "pear", "lemon", "peach", "carrot", "tomato", "radish",
    "potato", "onion", "spoon", "bowl",
]

CONTAINERS_POOL = [
    "blue_cupboard", "red_box", "green_bottle", "yellow_basket",
    "blue_bathtub", "red_drawer", "green_crate", "yellow_pantry",
    "purple_chest", "orange_bin", "white_suitcase", "black_trunk",
    "silver_bucket", "gold_jar", "red_envelope", "blue_pantry",
    "green_drawer", "red_basket", "blue_suitcase", "red_treasure_chest",
    "green_envelope", "blue_container", "red_container",
]

SECOND_ROOMS_POOL = [
    "waiting_room", "dining_room", "lobby",
]

# ----- Distractor templates -----
# These are pure noise lines. They never change object location or witness
# state; the verifier and answer-key generation ignore distractors.
# Soft distractors: mention only an agent. No containers (locations), no objects.

SOFT_DISTRACTOR_TEMPLATES = [
    "{agent} hummed a tune.",
    "{agent} stretched their arms.",
    "{agent} thought about the weather.",
    "{agent} yawned quietly.",
    "{agent} smiled to themselves.",
    "{agent} adjusted their sleeves.",
    "{agent} glanced at the clock.",
    "{agent} took a deep breath.",
    "{agent} tapped their foot.",
    "{agent} cleared their throat.",
]

# Hard distractors: mention a container that's actually used in the story,
# but don't change events. The container reference is a trap for
# models that pattern-match "find the most recently mentioned container".

HARD_DISTRACTOR_TEMPLATES = [
    "{agent} noticed the {container} was a bit dusty.",
    "{agent} commented on how clean the {container} looked.",
    "{agent} thought the {container} was a nice color.",
    "{agent} remarked that the {container} had been there a while.",
    "{agent} pointed at the {container} casually.",
    "{agent} mentioned liking the {container}.",
    "{agent} wondered who picked the {container}.",
    "{agent} admired the {container} for a moment.",
]


def join_names(names):
    """Build a comma-separated list with 'and' before the last name."""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def generate_story(rng: random.Random, story_id: int):
    """
    Generate one story. Returns {id, story, agents, object}.

    Generation proceeds in stages:
      1–4. World setup (agents, rooms, object, containers)
      5–7. Action plan per agent (mover flags, move counts, action sequence)
      8.   Action interleaving (randomized event_timeline)
      9.   Story emission (in-room phase)
      10.  Communication phase (post-exit claims)
      11.  Distractor injection (noise sentences)
      12.  Line numbering and final assembly
    """
    # ------------------------------------------------------------------
    # STAGE 1–4: World setup
    # ------------------------------------------------------------------

    # 1. Number of agents. Chain depth = N - 1, so N ∈ {6, 7, 8} gives
    #    higher-order chains of depth 5 / 6 / 7 (uniform mix).
    n_agents = rng.randint(*N_RANGE)

    # 2. Sample agents (sampling order = entry order).
    agents = rng.sample(AGENTS_POOL, n_agents)
    entry_order = agents

    # 3. Sample physical setting and object.
    room = rng.choice(ROOMS_POOL)
    second_room = rng.choice(SECOND_ROOMS_POOL)
    obj = rng.choice(OBJECTS_POOL)

    # 4. Sample containers: 1 initial + up to 2N destinations.
    #    Each move uses a fresh container (no immediate repeats), so we
    #    need at least 1 + max_total_moves containers. Worst case is
    #    2 moves × N agents = 2N moves.
    n_containers = 1 + n_agents * 2
    container_pool = rng.sample(CONTAINERS_POOL, n_containers)
    initial_container = container_pool[0]
    destination_pool = container_pool[1:]

    # ------------------------------------------------------------------
    # STAGE 5–7: Action plan per agent
    # ------------------------------------------------------------------

    # 5. Mover flag per agent. Bernoulli(P_MOVER), resampled until
    #    ≥2 movers exist. Stories with <2 movers cannot host moves with
    #    a non-actor witness, which would degenerate the witness-asymmetry
    #    structure the benchmark depends on.
    while True:
        mover_flags = [rng.random() < P_MOVER for _ in agents]
        if sum(mover_flags) >= 2:
            break

    # 6. Move count per mover: 1 or 2, weighted by MOVE_COUNT_WEIGHTS.
    #    More moves per mover → more belief flips per agent.
    move_counts = {}
    for a, is_mover in zip(agents, mover_flags):
        move_counts[a] = (
            rng.choices([1, 2], weights=list(MOVE_COUNT_WEIGHTS))[0]
            if is_mover else 0
        )

    # 7. Build each agent's action sequence. Order within an
    #    agent's actions is fixed by construction; cross-agent
    #    interleaving happens in stage 8.
    #      Non-mover: STAY then EXIT.
    #      Mover:     MOVE × move_count, then EXIT.
    agent_actions = {}
    for a in agents:
        if move_counts[a] == 0:
            agent_actions[a] = ["STAY", "EXIT"]
        else:
            agent_actions[a] = ["MOVE"] * move_counts[a] + ["EXIT"]

    # ------------------------------------------------------------------
    # STAGE 8: Action interleaving (randomized event_timeline)
    # ------------------------------------------------------------------
    # Produces a uniformly-random valid interleaving of per-agent action
    # sequences subject to:
    #   (a) per-agent action order is preserved,
    #   (b) MOVE requires ≥1 other agent in the room (audience constraint).
    # At each step, sample uniformly from agents whose next action is
    # currently legal:
    #   STAY/EXIT: legal iff the agent is still in the room.
    #   MOVE:      legal iff the agent is in the room AND ≥1 other agent is
    #              still in the room (audience constraint).
    # If no agent has a legal action and the remaining agents have only
    # pending MOVEs, drop those MOVEs (they cannot fire).
    event_timeline = []
    pointers = {a: 0 for a in agents}
    agents_in_room = set(agents)

    def next_action(a):
        return agent_actions[a][pointers[a]] if pointers[a] < len(agent_actions[a]) else None

    while True:
        active_agents = []
        for a in agents:
            action = next_action(a)
            if action is None:
                continue
            if action == "MOVE":
                if a in agents_in_room and len(agents_in_room - {a}) >= 1:
                    active_agents.append(a)
            elif action in ("STAY", "EXIT"):
                if a in agents_in_room:
                    active_agents.append(a)
        if not active_agents:
            inactive_agents = [a for a in agents if pointers[a] < len(agent_actions[a])]
            if not inactive_agents:
                break
            # Drop pending MOVEs that can't fire (no audience)
            for a in inactive_agents:
                while (pointers[a] < len(agent_actions[a])
                       and agent_actions[a][pointers[a]] == "MOVE"):
                    pointers[a] += 1
            continue

        a = rng.choice(active_agents)
        action = agent_actions[a][pointers[a]]
        pointers[a] += 1
        event_timeline.append((a, action))
        if action == "EXIT":
            agents_in_room.discard(a)

    # ------------------------------------------------------------------
    # STAGE 9: Story emission (in-room phase)
    # ------------------------------------------------------------------
    # Walk the event_timeline and emit one line per action.
    #   location: current location of the object.
    #   agents_in_room:  agents still present (used for motive-target candidates).
    #   events:   minimal ledger of {kind, location} for moves and the
    #             initial placement, used only for the lie pool and the
    #             hard-distractor pool. Ground truth is derived later from
    #             the final story.
    location = initial_container
    agents_in_room = set(agents)
    events = [{"kind": "place", "location": initial_container}]

    # Main story lines are accumulated without numbering; distractors are
    # injected later and the lines are then numbered.

    story_lines = []
    story_lines.append(f"{join_names(entry_order)} entered the {room}.")
    story_lines.append(f"The {obj} is in the {initial_container}.")

    destination_index = 0

    for actor, action in event_timeline:
        if action == "STAY":
            story_lines.append(
                f"{actor} made no movements and stayed in the {room} for 1 minute."
            )
        elif action == "EXIT":
            story_lines.append(f"{actor} exited the {room}.")
            agents_in_room.discard(actor)
        elif action == "MOVE":
            # Pick destination, distinct from current location
            while destination_index < len(destination_pool) and destination_pool[destination_index] == location:
                destination_index += 1
            assert destination_index < len(destination_pool), "ran out of containers"
            destination = destination_pool[destination_index]
            destination_index += 1

            # Motive composition: see MOTIVE_COMPOSITIONS at module level
            # for the rationale of the 33/33/33 ratio.
            #
            # The actor cannot be their own help/hide target. Sort
            # candidate agents for canonical order: agents_in_room is a set, and
            # iterating a set is non-deterministic across Python
            # invocations (PYTHONHASHSEED).
            candidates = sorted(p for p in agents_in_room if p != actor)
            assert candidates, "no candidates for motive target"
            if len(candidates) >= 2:
                composition = rng.choice(list(MOTIVE_COMPOSITIONS))
            else:
                # 'both' requires two distinct candidates; restrict to single-intent.
                composition = rng.choice(["help_only", "hide_only"])

            help_target = None
            hide_target = None
            if composition == "help_only":
                help_target = rng.choice(candidates)
            elif composition == "hide_only":
                hide_target = rng.choice(candidates)
            else:
                pair = rng.sample(candidates, 2)
                help_target, hide_target = pair[0], pair[1]

            motive_parts = []
            if help_target is not None:
                motive_parts.append(f"to help {help_target} find it")
            if hide_target is not None:
                motive_parts.append(f"to hide it from {hide_target}")
            motive_str = " and ".join(motive_parts)

            story_lines.append(
                f"{actor} moved the {obj} to the {destination} {motive_str}."
            )

            # Update state
            location = destination
            # Minimal event record — just enough for the lie pool and
            # distractor pool to know which containers have been used.
            events.append({"kind": "move", "location": destination})

    # ------------------------------------------------------------------
    # STAGE 10: Communication phase (post-exit claims)
    # ------------------------------------------------------------------
    # 1..3 communication lines after all in-room exits, before the
    # reconvene line. Each is independently:
    #   - public/private (controls who hears it)
    #   - true/false     (controls what is claimed)
    # The belief propagation trust-rule details (who actually believes whom) are NOT applied
    # here.
    # See docs/why_llms_fail_trust_rule.md.

    n_communications = rng.randint(*N_COMMUNICATIONS_RANGE)
    used_containers_for_lies = set()
    for ev in events:
        used_containers_for_lies.add(ev["location"])
    used_containers_for_lies.add(initial_container)

    for _ in range(n_communications):
        speaker = rng.choice(agents)
        is_public = rng.random() < P_PUBLIC_COMM
        is_truth = rng.random() < P_TRUTHFUL_COMM

        if is_truth:
            claimed_location = location  # the actual current location
        else:
            # Sort to get canonical order. Iterating a set in Python gives a
            # hash-randomized order across invocations (PYTHONHASHSEED) and
            # would break determinism even with a fixed rng seed.
            lie_pool = sorted(c for c in used_containers_for_lies if c != location)
            if not lie_pool:
                # All used containers equal the current location. 

                claimed_location = location  # fallback to truth
            else:
                claimed_location = rng.choice(lie_pool)

        if is_public:
            line = f"{speaker} publicly claimed that the {obj} is in the {claimed_location}."
        else:
            listener = rng.choice([a for a in agents if a != speaker])
            line = f"{speaker} privately told {listener} that the {obj} is in the {claimed_location}."

        story_lines.append(line)
        # No ground-truth tracking here; higher_order_beliefs_v10.py re-derives the
        # comm phase (trust rule, updated listeners, etc.) from text.

    # Final reconvene line
    story_lines.append(f"{join_names(entry_order)} entered the {second_room}.")

    # ------------------------------------------------------------------
    # STAGE 11: Distractor injection
    # ------------------------------------------------------------------
    # Distractors are sentences that never reference the object and never
    # change world state. They probe robustness against irrelevant content.
    # Two types:
    #   - SOFT: mentions only an agent, no container.
    #   - HARD: mentions a container that has been used in the story
    #           (a lexical overlap with relevant content).
    # Distractors are inserted into the in-room phase only; never at the
    # entry line, the placement line, or the final reconvene.
    n_distractors = rng.randint(*N_DISTRACTORS_RANGE)

    # Containers used in the story = initial + all destinations actually used.
    # Sort for canonical order (set iteration is non-deterministic).
    used_containers_set = {initial_container}
    for ev in events:
        if ev["kind"] == "move":
            used_containers_set.add(ev["location"])
    used_containers = sorted(used_containers_set)

    distractor_lines = []
    for _ in range(n_distractors):
        is_soft = rng.random() < P_SOFT_DISTRACTOR
        agent = rng.choice(agents)
        if is_soft:
            template = rng.choice(SOFT_DISTRACTOR_TEMPLATES)
            distractor_lines.append(template.format(agent=agent))
        else:
            template = rng.choice(HARD_DISTRACTOR_TEMPLATES)
            container = rng.choice(used_containers)
            distractor_lines.append(template.format(agent=agent, container=container))

    # Insert distractors at random positions, excluding line 0 (entry),
    # line 1 (placement), and the final reconvene line.
    final_lines = list(story_lines)
    for d in distractor_lines:
        # Recompute valid range each iteration since the list grows.
        max_pos = len(final_lines)  # final reconvene is at max_pos - 1
        valid = list(range(2, max_pos))  # exclude entry, placement, reconvene
        pos = rng.choice(valid)
        final_lines.insert(pos, d)

    # ------------------------------------------------------------------
    # STAGE 12: Line numbering and final assembly
    # ------------------------------------------------------------------
    numbered = [f"{i+1} {ln}" for i, ln in enumerate(final_lines)]
    story_text = "\n".join(numbered)

    # Story file includes `agents` and `object` because these are story
    # metadata (not answers) — they identify the entities involved before
    # any question is asked. Q0–Q3 ground truth is computed by higher_order_beliefs_v10.py.
    return {
        "id": story_id,
        "story": story_text,
        "agents": agents,
        "object": obj,
    }


def main(n=100,
         stories_out="/mnt/user-data/outputs/v10/stories_v10.jsonl"):
    """
    Generate n stories. Writes only stories — no ground truth.

    Q0–Q3 ground truth is produced separately by higher_order_beliefs_v10.py from the
    story text alone. Other question types live in their own scripts.
    """
    rng = random.Random(SEED)
    stories = []
    for i in range(n):
        ex = generate_story(rng, i)
        stories.append({
            "id": ex["id"],
            "story": ex["story"],
            "agents": ex["agents"],
            "object": ex["object"],
        })

    os.makedirs(os.path.dirname(stories_out), exist_ok=True)

    with open(stories_out, "w") as f:
        for s in stories:
            f.write(json.dumps(s) + "\n")

    print(f"Wrote {n} stories -> {stories_out}")
    print()
    print("--- Example 0 ---")
    print(stories[0]["story"])


if __name__ == "__main__":
    main()
