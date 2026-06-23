#!/usr/bin/env bash
# run_smoke.sh — end-to-end smoke test for the looped-transformer study.
#
# Wires the WHOLE stack together and fails loudly if any stage breaks:
#   STAGE A  gen_math_problems.py     synthetic problems (exact gold)
#   STAGE B  gen_math_transcripts.py  verified CoT transcripts (mock teacher, offline)
#   TRAIN 1  train_addition.py        from-scratch looped model on the depth task
#   TRAIN 2  train_sft.py             next-token SFT on the transcripts
#
# Everything is tiny and offline (no API key, no GPU). CPU runtime ~3-5 min total.
# Each Python step exits non-zero on failure; `set -e` aborts the whole run.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$HERE/../experiments"
cd "$HERE"

echo "############################################################"
echo "# looped-transformer-study — END-TO-END SMOKE TEST"
echo "############################################################"

# --- preflight ---
python3 -c "import sympy" 2>/dev/null || { echo "[FAIL] sympy missing — pip install -r ../requirements.txt"; exit 1; }
python3 -c "import torch" 2>/dev/null || { echo "[FAIL] torch missing — pip install -r ../requirements.txt"; exit 1; }

echo
echo "=== STAGE A: generate synthetic problems (exact gold) ==="
python3 "$EXP/gen_math_problems.py" --source programmatic --kinds arith \
        --n 200 --out "$HERE/problems.jsonl"

echo
echo "=== STAGE B: generate verified transcripts (mock teacher) ==="
python3 "$EXP/gen_math_transcripts.py" generate --problems "$HERE/problems.jsonl" \
        --teacher mock --samples 8 --keep 2 --out "$HERE/transcripts.jsonl"

echo
echo "=== TRAIN 1: from-scratch looped model on addition ==="
python3 "$HERE/train_addition.py"

echo
echo "=== TRAIN 2: transcript SFT ==="
python3 "$HERE/train_sft.py" --data "$HERE/transcripts.jsonl"

echo
echo "############################################################"
echo "# SMOKE TEST PASSED ✅  (data pipeline + both trainings)"
echo "############################################################"
