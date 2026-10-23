#!/usr/bin/env bash
set -euo pipefail

repository="/home/igor/devel/ideal_poly_volume_toolkit"
python_bin="/home/igor/miniforge3/bin/python3"
run_id="20260815T211114Z"
log_dir="$repository/results/logs/rl_training/$run_id"
model_template="$repository/results/rl_models/ppo_d{d}_controller-v2-seed42.zip"
validation_output="$repository/results/rl_validation/controller-v2-seed42-protocol2-$run_id.json"

cd "$repository"
exec >>"$log_dir/supervisor.log" 2>&1
printf 'supervisor_started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

while true; do
    all_complete=1
    for dimension in 3 4 5 6; do
        model="results/rl_models/ppo_d${dimension}_controller-v2-seed42.zip"
        config="results/rl_models/ppo_d${dimension}_controller-v2-seed42_config.json"
        session="ideal-poly-rl-d${dimension}-$run_id"
        if [[ -s "$model" && -s "$config" ]]; then
            continue
        fi
        all_complete=0
        if ! tmux has-session -t "$session" 2>/dev/null; then
            printf 'training_failed_at=%s dimension=%s reason=session-ended-without-artifact-pair\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$dimension"
            exit 1
        fi
    done
    if [[ "$all_complete" -eq 1 ]]; then
        break
    fi
    sleep 60
done

printf 'training_complete_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for dimension in 3 4 5 6; do
    smoke_path="/tmp/ideal-poly-rl-smoke-d${dimension}-$run_id.json"
    "$python_bin" scripts/validate_rl_controller.py \
        --smoke \
        --dims "$dimension" \
        --model-template "$model_template" \
        --out "$smoke_path" \
        >"$log_dir/smoke_d${dimension}.log" 2>&1
done

printf 'model_smokes_complete_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

exec taskset -c 0-7 nice -n 10 env \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR="/tmp/ideal-poly-rl-validation-mpl-$run_id" \
    "$python_bin" scripts/validate_rl_controller.py \
    --model-template "$model_template" \
    --out "$validation_output" \
    --seeds 10
