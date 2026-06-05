#!/bin/bash
# ============================================================================
# run.sh — one-command TGToM evaluation pipeline.
# ============================================================================
#
# SCRIPT DESCRIPTION
# ---------------------
# Runs the full evaluation pipeline end-to-end:
#   1. eval_orchestrator.py   produces predictions.jsonl + predictions_tbg.jsonl
#   2. score_v10.py           scores answer accuracy across 12 questions
#   3. tbg_scorer_v10.py      scores TBG graph reconstruction (4 metrics)
#
# Outputs are saved to ./results/.
#
# REQUIREMENTS
# ------------
#   pip install openai
#   export OPENROUTER_API_KEY=sk-or-v1-...
#
# USAGE
# -----
#   ./run.sh                  # full eval (100 stories, 3 trials, 3 models)
#   ./run.sh --pilot          # quick pilot (5 stories, 1 trial)
# ============================================================================

set -e  # fail fast on any error

# ----------------------------------------------------------------------------
# API key. Replace the placeholder below with your OpenRouter key before running.
# Get one at https://openrouter.ai/settings/keys
# Do NOT commit your real key to GitHub.
# ----------------------------------------------------------------------------
export OPENROUTER_API_KEY="your-openrouter-key-here"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
PASSTHROUGH_ARGS="$@"

mkdir -p "$RESULTS_DIR"

if [ -z "$OPENROUTER_API_KEY" ] || [ "$OPENROUTER_API_KEY" = "your-openrouter-key-here" ]; then
    echo "ERROR: OPENROUTER_API_KEY is not set."
    echo "Edit run.sh and replace 'your-openrouter-key-here' with your real key."
    echo "Get one at https://openrouter.ai/settings/keys"
    exit 1
fi

echo "=== Stage 1: Generating predictions ==="
python "$SCRIPT_DIR/eval_orchestrator.py" \
    --output "$RESULTS_DIR/predictions.jsonl" \
    $PASSTHROUGH_ARGS

echo
echo "=== Stage 2: Scoring answer accuracy ==="
python "$SCRIPT_DIR/score_v10.py" \
    "$RESULTS_DIR/predictions.jsonl" \
    | tee "$RESULTS_DIR/answer_metrics.txt"

echo
echo "=== Stage 3: Scoring TBG graph reconstruction ==="
python "$SCRIPT_DIR/tbg_scorer_v10.py" \
    "$RESULTS_DIR/predictions_tbg.jsonl" \
    | tee "$RESULTS_DIR/graph_metrics.txt"

echo
echo "=== Done ==="
echo "Results saved to:"
echo "  $RESULTS_DIR/predictions.jsonl       (raw responses + parsed)"
echo "  $RESULTS_DIR/predictions_tbg.jsonl   (TBG scoring input)"
echo "  $RESULTS_DIR/answer_metrics.txt      (12-question accuracy)"
echo "  $RESULTS_DIR/graph_metrics.txt       (4 graph metrics)"
