#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <build|smoke|train|train-smoke|run|run-smoke|eval|eval-smoke|train-dev|run-dev|eval-dev|clean>"
    exit 1
fi

COMMAND="$1"

# ============================================
# CONFIGURATION
# ============================================

TRAIN_DATA_REL="data/training_set"
RUN_DATA_REL="data/supplementary_set"
SMOKE_DATA_REL="data/training_smoke"

IMAGE_NAME="cinc2026"

MODEL_FULL_REL="model"
MODEL_SMOKE_REL="model_smoke"

OUT_FULL_REL="outputs"
OUT_SMOKE_REL="outputs_smoke"
DEMOGRAPHICS_FILE="demographics.csv"

# ============================================
# HELPERS
# ============================================

get_absolute_path() {
    local rel_path="$1"
    (cd "$rel_path" && pwd)
}

ensure_directory() {
    local dir_path="$1"
    mkdir -p "$dir_path"
}

to_docker_path() {
    local host_path="$1"

    if command -v cygpath >/dev/null 2>&1; then
        cygpath -m "$host_path"
    else
        echo "$host_path"
    fi
}

docker_cli() {
    MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" docker "$@"
}

evaluate_predictions() {
    local data_dir="$1"
    local output_dir="$2"
    local label="$3"
    local data_dir_docker output_dir_docker

    data_dir_docker="$(to_docker_path "$data_dir")"
    output_dir_docker="$(to_docker_path "$output_dir")"

    echo "Evaluating ${label} predictions..."
    docker_cli run --rm \
        -v "${data_dir_docker}:/challenge/eval_data:ro" \
        -v "${output_dir_docker}:/challenge/eval_outputs:ro" \
        "$IMAGE_NAME" \
        python evaluate_model.py \
            -d "/challenge/eval_data/${DEMOGRAPHICS_FILE}" \
            -o "/challenge/eval_outputs/${DEMOGRAPHICS_FILE}"
}

evaluate_predictions_dev() {
    local code_path="$1"
    local data_path="$2"
    local output_path="$3"
    local label="$4"
    local code_path_docker data_path_docker

    code_path_docker="$(to_docker_path "$code_path")"
    data_path_docker="$(to_docker_path "$data_path")"

    echo "Evaluating ${label} predictions..."
    docker_cli run --rm \
        -v "${code_path_docker}:/challenge" \
        -v "${data_path_docker}:/challenge/eval_data:ro" \
        "$IMAGE_NAME" \
        python evaluate_model.py \
            -d "/challenge/eval_data/${DEMOGRAPHICS_FILE}" \
            -o "$output_path/${DEMOGRAPHICS_FILE}"
}

dataset_has_labels() {
    local data_dir="$1"
    local demographics_path="$data_dir/$DEMOGRAPHICS_FILE"

    [[ -f "$demographics_path" ]] && head -n 1 "$demographics_path" | grep -q "Cognitive_Impairment"
}

build_image() {
    docker_cli build -t "$IMAGE_NAME" .
}

create_smoke() {
    echo "Creating smoke dataset..."
    bash scripts/create_smoke.sh
}

train_full() {
    local full_data model_full
    local full_data_docker model_full_docker

    full_data="$(get_absolute_path "$TRAIN_DATA_REL")"
    model_full="$(get_absolute_path ".")/${MODEL_FULL_REL}"
    full_data_docker="$(to_docker_path "$full_data")"
    model_full_docker="$(to_docker_path "$model_full")"

    ensure_directory "$model_full"

    docker_cli run --rm \
        -v "${full_data_docker}:/challenge/training_data:ro" \
        -v "${model_full_docker}:/challenge/model" \
        "$IMAGE_NAME" \
        python train_model.py -d training_data -m model -v
}

train_smoke() {
    local smoke_data model_smoke
    local smoke_data_docker model_smoke_docker

    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    model_smoke="$(get_absolute_path ".")/${MODEL_SMOKE_REL}"
    smoke_data_docker="$(to_docker_path "$smoke_data")"
    model_smoke_docker="$(to_docker_path "$model_smoke")"

    ensure_directory "$model_smoke"

    docker_cli run --rm \
        -v "${smoke_data_docker}:/challenge/training_data:ro" \
        -v "${model_smoke_docker}:/challenge/model" \
        "$IMAGE_NAME" \
        python train_model.py -d training_data -m model -v
}

run_full() {
    local run_data model_full out_full
    local run_data_docker model_full_docker out_full_docker

    run_data="$(get_absolute_path "$RUN_DATA_REL")"
    model_full="$(get_absolute_path "$MODEL_FULL_REL")"
    out_full="$(get_absolute_path ".")/${OUT_FULL_REL}"
    run_data_docker="$(to_docker_path "$run_data")"
    model_full_docker="$(to_docker_path "$model_full")"
    out_full_docker="$(to_docker_path "$out_full")"

    ensure_directory "$out_full"

    docker_cli run --rm \
        -v "${run_data_docker}:/challenge/holdout_data:ro" \
        -v "${model_full_docker}:/challenge/model:ro" \
        -v "${out_full_docker}:/challenge/holdout_outputs" \
        "$IMAGE_NAME" \
        python run_model.py -d holdout_data -m model -o holdout_outputs -v

    if dataset_has_labels "$run_data"; then
        evaluate_predictions "$run_data" "$out_full" "run-dataset"
    else
        echo "Skipping evaluation for run dataset (labels not present in ${RUN_DATA_REL}/${DEMOGRAPHICS_FILE})."
    fi
}

run_smoke() {
    local smoke_data model_smoke out_smoke
    local smoke_data_docker model_smoke_docker out_smoke_docker

    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    model_smoke="$(get_absolute_path "$MODEL_SMOKE_REL")"
    out_smoke="$(get_absolute_path ".")/${OUT_SMOKE_REL}"
    smoke_data_docker="$(to_docker_path "$smoke_data")"
    model_smoke_docker="$(to_docker_path "$model_smoke")"
    out_smoke_docker="$(to_docker_path "$out_smoke")"

    ensure_directory "$out_smoke"

    docker_cli run --rm \
        -v "${smoke_data_docker}:/challenge/holdout_data:ro" \
        -v "${model_smoke_docker}:/challenge/model:ro" \
        -v "${out_smoke_docker}:/challenge/holdout_outputs" \
        "$IMAGE_NAME" \
        python run_model.py -d holdout_data -m model -o holdout_outputs -v

    evaluate_predictions "$smoke_data" "$out_smoke" "smoke"
}

eval_full() {
    local run_data out_full

    run_data="$(get_absolute_path "$RUN_DATA_REL")"
    out_full="$(get_absolute_path "$OUT_FULL_REL")"

    if dataset_has_labels "$run_data"; then
        evaluate_predictions "$run_data" "$out_full" "run-dataset"
    else
        echo "Skipping evaluation for run dataset (labels not present in ${RUN_DATA_REL}/${DEMOGRAPHICS_FILE})."
    fi
}

eval_smoke() {
    local smoke_data out_smoke

    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    out_smoke="$(get_absolute_path "$OUT_SMOKE_REL")"

    evaluate_predictions "$smoke_data" "$out_smoke" "smoke"
}

# =====================
# DEVELOPMENT MODE (NO REBUILD)
# =====================

train_dev() {
    local code_path smoke_data model_smoke
    local code_path_docker smoke_data_docker

    code_path="$(get_absolute_path ".")"
    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    model_smoke="${code_path}/${MODEL_SMOKE_REL}"
    code_path_docker="$(to_docker_path "$code_path")"
    smoke_data_docker="$(to_docker_path "$smoke_data")"

    ensure_directory "$model_smoke"

    docker_cli run --rm \
        -v "${code_path_docker}:/challenge" \
        -v "${smoke_data_docker}:/challenge/data_smoke:ro" \
        "$IMAGE_NAME" \
        python train_model.py -d /challenge/data_smoke -m /challenge/model_smoke -v
}

run_dev() {
    local code_path smoke_data out_smoke
    local code_path_docker smoke_data_docker

    code_path="$(get_absolute_path ".")"
    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"
    out_smoke="${code_path}/${OUT_SMOKE_REL}"
    code_path_docker="$(to_docker_path "$code_path")"
    smoke_data_docker="$(to_docker_path "$smoke_data")"

    ensure_directory "$out_smoke"

    docker_cli run --rm \
        -v "${code_path_docker}:/challenge" \
        -v "${smoke_data_docker}:/challenge/data_smoke:ro" \
        "$IMAGE_NAME" \
        python run_model.py -d /challenge/data_smoke -m /challenge/model_smoke -o /challenge/outputs_smoke -v

    evaluate_predictions_dev "$code_path" "$smoke_data" "/challenge/outputs_smoke" "development smoke"
}

eval_dev() {
    local code_path smoke_data

    code_path="$(get_absolute_path ".")"
    smoke_data="$(get_absolute_path "$SMOKE_DATA_REL")"

    evaluate_predictions_dev "$code_path" "$smoke_data" "/challenge/outputs_smoke" "development smoke"
}

clean_all() {
    rm -rf "$MODEL_FULL_REL" "$MODEL_SMOKE_REL" "$OUT_FULL_REL" "$OUT_SMOKE_REL"
    echo "Models and outputs removed."
}

case "$COMMAND" in
    build)       build_image ;;
    smoke)       create_smoke ;;
    train)       train_full ;;
    train-smoke) train_smoke ;;
    run)         run_full ;;
    run-smoke)   run_smoke ;;
    eval)        eval_full ;;
    eval-smoke)  eval_smoke ;;
    train-dev)   train_dev ;;
    run-dev)     run_dev ;;
    eval-dev)    eval_dev ;;
    clean)       clean_all ;;
    *)
        echo "Invalid command: $COMMAND"
        echo "Valid commands: build, smoke, train, train-smoke, run, run-smoke, eval, eval-smoke, train-dev, run-dev, eval-dev, clean"
        exit 1
        ;;
esac
