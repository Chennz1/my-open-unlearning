#!/usr/bin/env bash
set -euo pipefail

# IA/RSP-TVC merge + TOFU evaluation example.
# Override any variable from the shell, e.g.:
#   GAMMAS="0 0.2" RIDGES="0.001 0.01" bash scripts/run_ia_tvc_tofu_forget10.sh

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export MASTER_PORT

MODEL=${MODEL:-Llama-2-7b-chat-hf}
FORGET_SPLIT=${FORGET_SPLIT:-forget10}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout10}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain90}
DEVICE=${DEVICE:-0}

TOKENIZER_PATH=${TOKENIZER_PATH:-/cnz/data/ms-home/Llama-2-7b-chat-hf}
ORIG_MODEL_PATH=${ORIG_MODEL_PATH:-/cnz/data/ms-home/Llama-2-7b-chat-hf}
FT_MODEL_PATH=${FT_MODEL_PATH:-/cnz/data/project/my-open-unlearning/saves/finetune/tofu_Llama-2-7b-chat-hf_full}
FORGET_PATH=${FORGET_PATH:-/cnz/data/project/my-open-unlearning/saves/finetune/tofu_Llama-2-7b-chat-hf_forget10_epoch5}
RETAIN_PATHS_CSV=${RETAIN_PATHS_CSV:-/cnz/data/project/my-open-unlearning/saves/finetune/tofu_Llama-2-7b-chat-hf_retain90_mimic_epoch5}

OUTPUT_BASE=${OUTPUT_BASE:-/cnz/data/project/my-open-unlearning/saves/unlearn/ia_tvc_forget10_7b_mlp}
FILTER_TYPE=${FILTER_TYPE:-mlp}
FORGET_METRIC=${FORGET_METRIC:-rank1}
ALPHA=${ALPHA:-1.0}
GAMMAS=(${GAMMAS:-0.2})
RIDGES=(${RIDGES:-0.001})

IFS=',' read -r -a RETAIN_PATHS <<< "$RETAIN_PATHS_CSV"
mkdir -p "$OUTPUT_BASE"

LOG_FILE="$OUTPUT_BASE/ia_tvc_run.log"
{
  echo "IA-TVC run started at $(date)"
  echo "MODEL=$MODEL"
  echo "FORGET_SPLIT=$FORGET_SPLIT"
  echo "RETAIN_PATHS=${RETAIN_PATHS[*]}"
  echo "FILTER_TYPE=$FILTER_TYPE"
  echo "FORGET_METRIC=$FORGET_METRIC"
  echo "GAMMAS=${GAMMAS[*]}"
  echo "RIDGES=${RIDGES[*]}"
  echo "MASTER_PORT=$MASTER_PORT"
} | tee "$LOG_FILE"

for gamma in "${GAMMAS[@]}"; do
  for ridge in "${RIDGES[@]}"; do
    run_name="ia_tvc_${MODEL}_${FORGET_SPLIT}_${FILTER_TYPE}_g${gamma}_r${ridge}"
    save_path="$OUTPUT_BASE/$run_name"

    echo ""
    echo "========================================"
    echo "Running $run_name"
    echo "Saving to $save_path"
    echo "========================================"

    CUDA_VISIBLE_DEVICES="$DEVICE" python scripts/ia_tvc_merge.py \
      --tokenizer_path "$TOKENIZER_PATH" \
      --orig_model_path "$ORIG_MODEL_PATH" \
      --ft_model_path "$FT_MODEL_PATH" \
      --forget_path "$FORGET_PATH" \
      --retain_paths "${RETAIN_PATHS[@]}" \
      --save_path "$save_path" \
      --alpha "$ALPHA" \
      --gamma "$gamma" \
      --ridge "$ridge" \
      --forget_metric "$FORGET_METRIC" \
      --filter_type "$FILTER_TYPE" \
      2>&1 | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES="$DEVICE" python src/eval.py \
      experiment=eval/tofu/default.yaml \
      forget_split="$FORGET_SPLIT" \
      holdout_split="$HOLDOUT_SPLIT" \
      model="$MODEL" \
      task_name="$run_name" \
      model.model_args.pretrained_model_name_or_path="$save_path" \
      paths.output_dir="$save_path/evals" \
      retain_logs_path="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json" \
      2>&1 | tee -a "$LOG_FILE"
  done
done

echo "IA-TVC run completed at $(date)" | tee -a "$LOG_FILE"
