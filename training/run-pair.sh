#!/usr/bin/env bash
# Run from the appropriate ROLL worktree with its CUDA/Ray dependencies active.
set -euo pipefail
evidence=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(pwd)
kind=${1:?Usage: run-pair.sh sft|vision NEW_OUTPUT_DIR}
run=$(realpath -m "${2:?Select a new output directory}")
case "$kind" in
  sft) entry=examples/start_sft_pipeline.py ;;
  vision) entry=examples/start_rlvr_vl_pipeline.py ;;
  *) exit 2 ;;
esac
test -f "$entry"
test ! -e "$run"
: "${CUDA_VISIBLE_DEVICES:?Select one GPU by numeric CUDA device index}"
: "${RL_INSIGHT_SERVER_URL:?Set the RL-Insight server API URL}"
[[ $CUDA_VISIBLE_DEVICES =~ ^[0-9]+$ ]]
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAY_USAGE_STATS_ENABLED=0
export NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1 NCCL_NET=Socket
export MULTI_TENANT=1
export MASTER_PORT="${MASTER_PORT:-23679}" DASHBOARD_PORT="${DASHBOARD_PORT:-23680}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,$(hostname -I | awk '{print $1}')"
export no_proxy="$NO_PROXY"
args=()
if [[ -n "${ROLL_MODEL_CONFIG_SOURCE:-}" ]]; then args+=(--source "$ROLL_MODEL_CONFIG_SOURCE"); fi
python "$evidence/prepare.py" "$kind" "$run" "${args[@]}"
config_path=$(realpath --relative-to="$repo/examples" "$run")

# A separate Ray runtime and Python process for each phase.
run_phase() (
  phase=$1
  export RAY_TMPDIR="$(mktemp -d /tmp/roll529-ray.XXXXXX)"
  export ROLL_LOG_DIR="$run/driver-$phase" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_CACHE="$run/datasets"
  timeout --kill-after=10 620 ray start --head --port="$MASTER_PORT" --num-cpus=12 --num-gpus=1 \
    --object-store-memory=536870912 --dashboard-port="$DASHBOARD_PORT" \
    --ray-client-server-port="$((DASHBOARD_PORT+1))" --include-dashboard=false --block > "$run/ray-$phase.log" 2>&1 &
  ray_pid=$!
  trap 'kill -TERM "$ray_pid" 2>/dev/null || true; wait "$ray_pid" || true' EXIT
  export RAY_ADDRESS="127.0.0.1:$MASTER_PORT"
  for ((i=0; i<30; i++)); do
    if ray status --address="$RAY_ADDRESS" > "$run/status-$phase.log" 2>&1; then break; fi
    kill -0 "$ray_pid"
    sleep 1
  done
  timeout --signal=INT --kill-after=20 520 python -u "$entry" \
    --config_path "$config_path" --config_name "$phase" > "$run/$phase.log" 2>&1
  grep -F 'RL-Insight experiment:' "$run/$phase.log"
  grep -F 'pipeline complete!' "$run/$phase.log"
)

git rev-parse HEAD
run_phase first
first=$(sed -n 's/.*add timestamp to output_dir //p' "$run/first.log" | head -1)/checkpoint-2
test -s "$first/pipeline/worker_state_pipeline.json"
python - "$run" "$first" "$kind" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

run = Path(sys.argv[1])
config = OmegaConf.load(run / "first.yaml")
config.resume_from_checkpoint = sys.argv[2]
config.max_steps = 5  # ROLL's final loop index is max_steps - 1.
if sys.argv[3] == "sft":
    config.sft_train.data_args.file_name = [str(run / "resumed.jsonl")]
OmegaConf.save(config, run / "resumed.yaml")
PY
run_phase resumed
grep -F "Loaded DCP checkpoint from" "$run/resumed.log"
last=$(sed -n 's/.*add timestamp to output_dir //p' "$run/resumed.log" | head -1)/checkpoint-4
test -s "$last/pipeline/worker_state_pipeline.json"
echo "$last"
