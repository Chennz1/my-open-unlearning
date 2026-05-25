#!/usr/bin/env bash
set -euo pipefail

# Stage 2 for IA-TVC: merge with closed-form retain-subspace weights and
# evaluate each merged model by calling scripts/tofu_diy_eval.sh.

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export MASTER_PORT
echo "Master Port: $MASTER_PORT"

MODEL=${MODEL:-Llama-2-7b-chat-hf}
FORGET_SPLIT=${FORGET_SPLIT:-forget10}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout10}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain90}
EPOCH=${EPOCH:-10}
DEVICE=${DEVICE:-0}

TOKENIZER_PATH=${TOKENIZER_PATH:-/cnz/data/ms-home/Llama-2-7b-chat-hf}
ORIG_MODEL_PATH=${ORIG_MODEL_PATH:-/cnz/data/ms-home/Llama-2-7b-chat-hf}
FT_MODEL_PATH=${FT_MODEL_PATH:-/cnz/data/project/my-open-unlearning/saves/finetune/tofu_Llama-2-7b-chat-hf_full}
FORGET_PATH=${FORGET_PATH:-/cnz/data/project/my-open-unlearning/saves/finetune/tofu_${MODEL}_${FORGET_SPLIT}_epoch${EPOCH}}

if [[ -z "${RETAIN_PATHS_CSV:-}" ]]; then
  retain_paths=("/cnz/data/project/my-open-unlearning/saves/finetune/tofu_${MODEL}_${RETAIN_SPLIT}_mimic_epoch${EPOCH}")
else
  IFS=',' read -r -a retain_paths <<< "$RETAIN_PATHS_CSV"
fi

OUTPUT_BASE=${OUTPUT_BASE:-/cnz/data/project/my-open-unlearning/saves/unlearn/ia_tvc_${FORGET_SPLIT}_${MODEL}_mlp_K${#retain_paths[@]}}
FILTER_TYPE=${FILTER_TYPE:-mlp}
FORGET_METRIC=${FORGET_METRIC:-rank1}
ALPHAS=(${ALPHAS:-1.0})
GAMMAS=(${GAMMAS:-0 0.2 0.5})
RIDGES=(${RIDGES:-0.001 0.01})
KEEP_MODELS=${KEEP_MODELS:-0}

mkdir -p "$OUTPUT_BASE"
log_file="$OUTPUT_BASE/search_log.txt"
{
  echo "IA-TVC Hyperparameter Search Started at $(date)"
  echo "MODEL=$MODEL"
  echo "FORGET_SPLIT=$FORGET_SPLIT"
  echo "RETAIN_SPLIT=$RETAIN_SPLIT"
  echo "FORGET_PATH=$FORGET_PATH"
  echo "RETAIN_PATHS=${retain_paths[*]}"
  echo "ALPHAS=${ALPHAS[*]}"
  echo "GAMMAS=${GAMMAS[*]}"
  echo "RIDGES=${RIDGES[*]}"
  echo "FILTER_TYPE=$FILTER_TYPE"
  echo "FORGET_METRIC=$FORGET_METRIC"
  echo "KEEP_MODELS=$KEEP_MODELS"
  echo "========================================"
} | tee "$log_file"

total_combinations=$(( ${#ALPHAS[@]} * ${#GAMMAS[@]} * ${#RIDGES[@]} ))
current=0

for alpha in "${ALPHAS[@]}"; do
  for gamma in "${GAMMAS[@]}"; do
    for ridge in "${RIDGES[@]}"; do
      current=$((current + 1))
      tag="a${alpha}_g${gamma}_r${ridge}"
      temp_model_path="$OUTPUT_BASE/model_${tag}"
      target_eval_dir="$OUTPUT_BASE/evals_${tag}"

      if [[ -f "$target_eval_dir/TOFU_EVAL.json" ]]; then
        echo "Skipping $tag [$current/$total_combinations] - already completed"
        continue
      fi

      echo ""
      echo "========================================"
      echo "Progress: $current/$total_combinations"
      echo "Testing alpha=$alpha, gamma=$gamma, ridge=$ridge"
      echo "========================================"

      echo "Step 1/3: IA-TVC merge..."
      CUDA_VISIBLE_DEVICES="$DEVICE" python scripts/ia_tvc_merge.py \
        --tokenizer_path "$TOKENIZER_PATH" \
        --orig_model_path "$ORIG_MODEL_PATH" \
        --ft_model_path "$FT_MODEL_PATH" \
        --forget_path "$FORGET_PATH" \
        --retain_paths "${retain_paths[@]}" \
        --save_path "$temp_model_path" \
        --alpha "$alpha" \
        --gamma "$gamma" \
        --ridge "$ridge" \
        --forget_metric "$FORGET_METRIC" \
        --filter_type "$FILTER_TYPE" \
        2>&1 | tee -a "$log_file"

      echo "Step 2/3: Evaluating via scripts/tofu_diy_eval.sh..."
      EVAL_MODEL_PATH="$temp_model_path" \
      EVAL_OUTPUT_DIR="$temp_model_path/evals" \
      EVAL_TASK_NAME="tofu_${MODEL}_${FORGET_SPLIT}_IA_TVC_${tag}" \
      EVAL_MODEL="$MODEL" \
      EVAL_FORGET_SPLIT="$FORGET_SPLIT" \
      EVAL_HOLDOUT_SPLIT="$HOLDOUT_SPLIT" \
      EVAL_RETAIN_SPLIT="$RETAIN_SPLIT" \
      EVAL_DEVICE="$DEVICE" \
      bash scripts/tofu_diy_eval.sh 2>&1 | tee -a "$log_file"

      echo "Step 3/3: Preserve evals and optionally remove model weights..."
      if [[ -d "$temp_model_path/evals" ]]; then
        rm -rf "$target_eval_dir"
        mv "$temp_model_path/evals" "$target_eval_dir"
      fi
      if [[ "$KEEP_MODELS" != "1" ]]; then
        rm -rf "$temp_model_path"
      fi
      echo "Finished $tag"
    done
  done
done

summary_file="$OUTPUT_BASE/results_summary.txt"
{
  echo "IA-TVC Hyperparameter Search Results Summary"
  echo "============================================"
  echo "Model: $MODEL"
  echo "Forget split: $FORGET_SPLIT"
  echo "Retain paths: ${retain_paths[*]}"
  echo "Alpha range: ${ALPHAS[*]}"
  echo "Gamma range: ${GAMMAS[*]}"
  echo "Ridge range: ${RIDGES[*]}"
  echo ""
  for alpha in "${ALPHAS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
      for ridge in "${RIDGES[@]}"; do
        tag="a${alpha}_g${gamma}_r${ridge}"
        evals_dir="$OUTPUT_BASE/evals_${tag}"
        if [[ -f "$evals_dir/TOFU_EVAL.json" ]]; then
          echo "$tag: $evals_dir/TOFU_EVAL.json"
        fi
      done
    done
  done
} > "$summary_file"

echo "Summary saved to: $summary_file"
echo "All done."
