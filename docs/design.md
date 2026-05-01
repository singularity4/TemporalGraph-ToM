# Story design — gen_v10.py

## Benchmark design

Each multi-agent story tests higher order theory-of-mind reasoning, higher order ounterfactual reasoning, casual reasoning, and common knowledge.

Story consists of three phases. 

### 1. First phase

N agents enter a room together. An object is placed in a container. Agents
perform interleaved `MOVE` / `STAY` / `EXIT` actions. A `MOVE` relocates the
object and is annotated with the actor's motive (help X, hide Y, or both).

Witnessing rule: every agent in the room at the time of the move witnesses
it, except the hide-target (if any). The hide-target agent does not witness —
hiding is causally enacted.

Exits within the in-room phase are irreversible.

### 2. Second (communication) phase

After the last exit, some agents make claims about the object's
location. Each claim is independently public/private and true/false.

Listeners apply the belief propagation trust rule: a listener trusts a speaker iff
the speaker exited the room later than the listener (the speaker plausibly
has more recent info). 

Trusted claims update the listener's belief; untrusted claims do not.

### 3. Reconvene

All agents enter a second room together. This ends the story.

### Distractors

Distractors (noise sentences per story) are randomly distributed into the
first phase to test model robustness against irrelevant content.

## Design parameters

Parameters for controlling difficulty and variation are defined as constants at
the top of `gen_v10.py`.

| Parameter | Default | Meaning |
|---|---|---|
| `N_RANGE` | `(6, 8)` | Number of agents per story (uniform) |
| `P_MOVER` | `0.6` | Per-agent probability of being a mover |
| `MOVE_COUNT_WEIGHTS` | `(3, 1)` | Weights for [1 move, 2 moves] per mover |
| `N_DISTRACTORS_RANGE` | `(4, 8)` | Distractors per story (uniform) |
| `P_SOFT_DISTRACTOR` | `0.5` | Soft (no container) vs hard (uses a container) |
| `N_COMMUNICATIONS_RANGE` | `(1, 3)` | Communications per story (uniform) |
| `P_PUBLIC_COMM` | `0.5` | Public vs private claim |
| `P_TRUTHFUL_COMM` | `0.5` | Truthful vs lie |
| `MOTIVE_COMPOSITIONS` | uniform | `{help_only, hide_only, both}`; 33/33/33 when ≥2 candidate targets present |

Higher N gives chains of greater depth (chain depth = N − 1). Higher
`P_MOVER` and move counts produce more belief-updating events, which
increases the witness asymmetry between agents.

The 33/33/33 motive ratio over `{help_only, hide_only, both}` yields
~36/100 stories where the higher-order chain answer collapses to the
initial placement. Raising the `both` weight to 90% (a deprecated
v5 setting) would push the collapse rate higher, since witness sets
shrink whenever a hide-target is present.

## Sampled variables (randomly drawn per story)

**Story-level:** N, agents (entry order), room, second_room, object,
initial_container, destination_containers (sequence).

**Agent-level:** `is_mover` (per agent), `move_count` (per mover).

**Per-move:** `motive_composition` (help_only / hide_only / both),
`help_target` (if applicable), `hide_target` (if applicable).

**Per-communication:** `speaker`, `is_public`, `is_truthful`, `listener`
(private only), `claimed_location` (= true location if truthful, else a
random used container ≠ true location).

**Per-distractor:** type (soft/hard), agent, container (hard only),
insertion position.

**Timeline:** all agent actions (`MOVE` / `STAY` / `EXIT`) are interleaved
at random subject to two constraints: (a) per-agent action order is
preserved, and (b) `MOVE` requires at least one other agent in the room
(no unobserved moves).
