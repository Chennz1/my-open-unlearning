#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


per_device_train_batch_size=4
gradient_accumulation_steps=8


model=Llama-2-7b-hf

data_splits=(
    "News"
    # "Books"
)

trainers=(
    # "GradAscent"
    # "GradDiff"
    # "NPO"
    "NullSpace"
)

# #########################################################
# #################### MUSE Unlearning ####################
# #########################################################

# diy_path="saves/finetune/muse_Llama-2-7b-hf_News_forget"
diy_path="vectors/muse_test-7b"
# diy_path="/home/cnz/.cache/huggingface/hub/models--muse-bench--MUSE-news_target/snapshots/a2f39769e9a0b98ec1cdd12f65e9962502208935"

for data_split in "${data_splits[@]}"; do
    for trainer in "${trainers[@]}"; do

        task_name=muse_${model}_${data_split}_${trainer}

        CUDA_VISIBLE_DEVICES=1  python src/eval.py \
        experiment=eval/muse/default.yaml \
        data_split=${data_split} \
        task_name=${task_name} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=${diy_path} \
        paths.output_dir=${diy_path}/evals \
        retain_logs_path=saves/eval/muse_${model}_${data_split}_retrain/MUSE_EVAL.json
    done
done
