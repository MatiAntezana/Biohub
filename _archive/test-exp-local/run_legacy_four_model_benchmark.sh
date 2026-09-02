#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVALUATOR="$PROJECT_ROOT/test-exp-local/evaluator.py"
PROTOCOL="$PROJECT_ROOT/test-exp-local/protocols/legacy_exp017_exp019_pilot.json"
DATA_DIR="$PROJECT_ROOT/data/train"
OUTPUT_ROOT="$PROJECT_ROOT/experiments/test-exp-local-four-models"
CHECKPOINTS="$PROJECT_ROOT/checkpoints"

mkdir -p "$OUTPUT_ROOT"

run_experiment() {
  local experiment_id="$1"
  local architecture="$2"
  local checkpoint="$3"
  local threshold="$4"
  local tta="$5"
  local base_channels="$6"
  local output_dir="$OUTPUT_ROOT/$experiment_id"

  mkdir -p "$output_dir"
  echo "START $experiment_id $(date --iso-8601=seconds)" | tee "$output_dir/run.status"
  "$PYTHON_BIN" -u "$EVALUATOR" \
    --experiment-id "$experiment_id" \
    --architecture "$architecture" \
    --base-channels "$base_channels" \
    --checkpoint "$CHECKPOINTS/$checkpoint" \
    --project-root "$PROJECT_ROOT" \
    --data-dir "$DATA_DIR" \
    --protocol "$PROTOCOL" \
    --output-dir "$output_dir" \
    --threshold "$threshold" \
    --tta "$tta" >"$output_dir/run.log" 2>&1
  echo "SUCCESS $experiment_id $(date --iso-8601=seconds)" | tee "$output_dir/run.status"
}

run_experiment EXP-015 anisotropic_unet3d exp015_anisotropic_90_10.pt 0.035 flip_x 32
run_experiment EXP-017 wide_unet3d exp017_wide_unet3d_90_10.pt 0.035 flip_x 64
run_experiment EXP-019 wide_resunet3d exp019_wide_resunet3d_90_10.pt 0.050 flip_x 64
run_experiment EXP-022 stardist3d exp022_stardist3d_90_10.pt 0.035 identity 32

"$PYTHON_BIN" "$PROJECT_ROOT/test-exp-local/build_board.py" \
  "$OUTPUT_ROOT/EXP-015/evaluation.json" \
  "$OUTPUT_ROOT/EXP-017/evaluation.json" \
  "$OUTPUT_ROOT/EXP-019/evaluation.json" \
  "$OUTPUT_ROOT/EXP-022/evaluation.json" \
  --output-dir "$OUTPUT_ROOT/board" \
  --bootstrap-iterations 10000 >"$OUTPUT_ROOT/board.log" 2>&1

echo "SUCCESS ALL $(date --iso-8601=seconds)" | tee "$OUTPUT_ROOT/run.done"
