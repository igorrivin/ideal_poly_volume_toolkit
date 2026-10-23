#!/usr/bin/env bash
set -euo pipefail

dimension="$1"
cpu_pair="$2"
log_path="$3"
repository="/home/igor/devel/ideal_poly_volume_toolkit"
python_bin="/home/igor/miniforge3/bin/python3"

cd "$repository"
exec >>"$log_path" 2>&1

printf 'started_at=%s dimension=%s cpus=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$dimension" "$cpu_pair"
exec taskset -c "$cpu_pair" nice -n 10 env \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR="/tmp/ideal-poly-rl-mpl-d${dimension}" \
    "$python_bin" scripts/train_polytope_controller.py \
    --d "$dimension" \
    --steps 500000 \
    --seed 42 \
    --horizon 40 \
    --threads 2
