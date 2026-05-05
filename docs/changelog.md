# Changelog

The evolution of the relevant benchmark code edits (internally tagged v10).

## v8

New: verify_graph_v8.py — a TBG verifier that builds an explicit
temporal belief graph (per-agent state at each time step, propagated
forward by witnessing and trust rules) and reads off ground truth from
the final state. Independent of `parse_and_recompute`. Both verifiers
must agree for ground truth to be considered correct.


## v9
Question file reorganization. The five v8 question scripts are reorganized
into four files grouped by reasoning type:

- `higher_order_beliefs_v9.py` — Q0, Q1, Q2, Q3 (was `core_questions_v8.py`)
- `counterfactual_beliefs_v9.py` — Q5, Q6, Q7, Q10 (Q10 moved in from
  `q10_v8.py`)
- `causal_beliefs_v9.py` — Q8 (was `extra_questions_v8.py`; Q9 moved out)
- `common_knowledge_v9.py` — Q9, Q11, Q13 (Q9 moved in from
  `extra_questions_v8.py`; was `knowledge_questions_v8.py`)

Story-generation logic is unchanged (stories are byte-identical to v8). 

Both verifiers (`verify_v9.py` and `verify_graph_v9.py`) updated to read
from the new file structure and pass 100/100 across all 12 questions.

## v0.1 (internal tag v10)

Public release-ready version.

Naming and framing:
- Benchmark name standardized: **TemporalGraph-ToM (TGToM)**
- Reasoning scaffold name standardized: **Temporal Belief Graph (TBG)**
- Internal names replaced everywhere in code, docstrings, comments,
  README, and docs.

Coverage bug fixes (caught by `analyze_dataset_stats_v10.py`):

- **Q3** (higher-order intent): in v9, chains were sampled from all
  agents, but the chain must be a subset of the target move's witness
  set to yield a non-null intent answer. 84/100 v9 stories had null Q3
  answers because of this. Fix: targeted selection picks (move,
  occurrence) pairs whose witness pool (excluding the mover) has at
  least chain_depth members; chain is sampled from that pool. When no
  such move exists, fall back to a shorter chain drawn from the move
  with the largest available witness pool. v0.1: 0/100 nulls.
- **Q7** (motive-flip counterfactual): in v9, the chain pool was the
  *original* witness set of the move. Under a motive flip, the original
  help-target becomes the new hide-target, who no longer witnesses.
  98/100 v9 stories had null Q7 answers because of this. Fix: chain
  pool computed from the *counterfactual* witness set (original ∪
  {original_hide} ∖ {new_hide}, also excluding the mover). When no move
  has chain_depth counterfactual-witnesses, fall back to a shorter
  chain. v0.1: 0/100 nulls.
- Both fixes preserve targeted-vs-fallback labeling and pass both
  verifiers 100/100.

Documentation:
- README rewritten with 2–3 lines per file/script (description plus
  input and output arrows). Reader can scan the file map and see the
  data dependency graph.

New script:
- `analyze_dataset_stats_v10.py` reports descriptive statistics over
  stories and ground truth: agent counts, container counts, event
  distribution, intent distribution, trust distribution, per-question
  targeted rates, chain-depth distributions, yes-rates for counterfactual questions,
  and null-answer rates.

Code organization:
- `counterfactual_beliefs_v10.py` extracted four small helpers used
  across Q5/Q6/Q7/Q10: `build_chain_phrase`,
  `find_speaker_comm_occurrence`, `replace_comm_claim`,
  `counterfactual_motive_witnesses`.
- Q5 deduplicated: targeted and fallback share entry-building.
- All output JSONLs byte-identical before and after the refactor.

## v0.2 (current)

Scaling and TBG evaluation.

Scaled benchmark from 100 to 1k stories:
- `gen_v10.py` default `n` changed from 100 to 1k.
- First 100 stories byte-identical to v0.1 (per-(question, story)
  seeding means each answer for story i depends only on i).
- Both verifiers pass 1000/1000.
- 1 null Q3 answer and 1 null Q7 answer at story 742 — a structurally
  degenerate case (only 1 move with 2 witnesses, can't form a belief chain
  of any required depth).

`verify_graph_v10.py` extended to track edges:
- `build_graph` now returns `(belief, edges)`. The new `edges` is a list
  of directed links between agents produced by events, each as
  `{source_agent, target_agent, line, relation_type}`, with
  `relation_type` in {trusted, untrusted, cooperative, deceptive}.
- Communication events produce trusted or untrusted edges for each listener 
  agent using the trust rule. Move events produce cooperative or deceptive edges
  (help and hide targets). Observation updates are self-updates
  (node-state changes), not links, following the TBG formalism in the
  paper abstract.
- The four internal callers in `verify_graph_v10.main` were updated to
  unpack the tuple as `belief, _ = build_graph(...)`. No other changes
  to the verifier.

New script: `tbg_scorer_v10.py`
- Scores model-predicted temporal belief graphs against the ground truth
  TBG for each story.
- Reports four graph metrics introduced in the paper abstract:
  Final Node Accuracy (FNA), Time-Respecting Node Accuracy (TNA;
  full belief trajectory), Edge F1 (Temporal and Static), and Normalized 
  Structural Distance (NSD). The Exact TBG Reconstruction Rate (fraction 
  of stories with NSD = 0) is also reported as an aggregate.
- Includes a `--self-test` flag that runs five synthetic test TBG cases
  (perfect predictions, empty predictions, edge subset, line shifts,
  relation-type corruption) to verify graph metrics.

Stylistic:
- `common_knowledge_v10.py` no longer hardcodes "/100" in its print
  output; uses `len(out)`.
- Q6 docstring updated from "~17/100 stories" to "~14% of stories".
- `design.md` updated from "~36/100" to "~36%".
- README updated from "Samples 100 stories" to "Samples 1k stories"
  and includes the `tbg_scorer_v10.py` entry.

