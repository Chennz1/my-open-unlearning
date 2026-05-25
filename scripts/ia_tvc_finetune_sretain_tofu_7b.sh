#!/usr/bin/env bash
set -euo pipefail

# Stage 1 for IA-TVC: train small retain-shard task-vector models from theta_base.
# Unlike the original sretain script, the default total retain budget is capped
# at the forget-set size, then split across NUM_RETAIN_SHARDS disjoint shards.
#
# For forget10/retain90 with TOFU_TRIM_BASE=4000:
#   forget_size = 400
#   NUM_RETAIN_SHARDS=4
#   each shard uses 100 retain examples

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export MASTER_PORT
echo "Master Port: $MASTER_PORT"

MODEL=${MODEL:-Llama-2-7b-chat-hf}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/cnz/data/ms-home/Llama-2-7b-chat-hf}
FORGET_SPLIT=${FORGET_SPLIT:-forget10}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain90}
EPOCH=${EPOCH:-10}
CUDA_DEVICES=${CUDA_DEVICES:-0,1}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
TOFU_TRIM_BASE=${TOFU_TRIM_BASE:-4000}
NUM_RETAIN_SHARDS=${NUM_RETAIN_SHARDS:-4}

retain_pct=${RETAIN_SPLIT#retain}
if [[ -z "$retain_pct" || ! "$retain_pct" =~ ^[0-9]+$ ]]; then
  echo "Failed to parse retain percentage from RETAIN_SPLIT=$RETAIN_SPLIT" >&2
  exit 1
fi

forget_size=$(( TOFU_TRIM_BASE * (100 - retain_pct) / 100 ))
if [[ "$forget_size" -le 0 ]]; then
  echo "Computed non-positive forget_size=$forget_size" >&2
  exit 1
fi

shard_size=$(( forget_size / NUM_RETAIN_SHARDS ))
if [[ "$shard_size" -le 0 ]]; then
  echo "NUM_RETAIN_SHARDS=$NUM_RETAIN_SHARDS is too large for forget_size=$forget_size" >&2
  exit 1
fi

echo "Training small retain shards:"
echo "  model=$MODEL"
echo "  base=$BASE_MODEL_PATH"
echo "  retain_split=$RETAIN_SPLIT"
echo "  forget_size_budget=$forget_size"
echo "  num_shards=$NUM_RETAIN_SHARDS"
echo "  shard_size=$shard_size"
echo "  total_retain_used=$(( shard_size * NUM_RETAIN_SHARDS ))"

for shard_id in $(seq 0 $((NUM_RETAIN_SHARDS - 1))); do
  start=$(( shard_id * shard_size ))
  end=$(( start + shard_size ))
  task_name="tofu_${MODEL}_${RETAIN_SPLIT}_sretain_shard${shard_id}_n${shard_size}_epoch${EPOCH}"

  echo ""
  echo "Training retain shard $shard_id: train[$start:$end] -> $task_name"

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" accelerate launch \
    --config_file configs/accelerate/default_config.yaml \
    --main_process_port "$MASTER_PORT" \
    src/train.py experiment=finetune/tofu/default.yaml \
    task_name="$task_name" \
    model="$MODEL" \
    data/datasets@data.train=TOFU_QA_retain \
    data.train.TOFU_QA_retain.args.hf_args.name="$RETAIN_SPLIT" \
    data.train.TOFU_QA_retain.args.hf_args.split="train[${start}:${end}]" \
    model.model_args.pretrained_model_name_or_path="$BASE_MODEL_PATH" \
    trainer.args.learning_rate=1e-5 \
    trainer.args.per_device_train_batch_size="$PER_DEVICE_TRAIN_BATCH_SIZE" \
    trainer.args.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.num_train_epochs="$EPOCH" \
    trainer.args.gradient_checkpointing=true \
    trainer.args.save_strategy=no
done

echo ""
echo "Retain shard models:"
for shard_id in $(seq 0 $((NUM_RETAIN_SHARDS - 1))); do
  echo "  saves/finetune/tofu_${MODEL}_${RETAIN_SPLIT}_sretain_shard${shard_id}_n${shard_size}_epoch${EPOCH}"
done
