#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    "Llama-2-7b-hf"
    # "Llama-2-13b-hf"
)
per_device_train_batch_size=4
gradient_accumulation_steps=2

data_splits=(
    "News"
    # "Books"
)

########################################################################################################################
########################################### FORGET Finetuned MUSE ######################################################
########################################################################################################################

for data_split in "${data_splits[@]}"; do
    dataset_path="muse-bench/MUSE-${data_split}"

    for model in "${models[@]}"; do
        task_name=muse_${model}_${data_split}_forget_e10

        CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
        src/train.py experiment=finetune/muse/default.yaml \
        task_name=${task_name} \
        model=${model} \
        data_split=${data_split} \
        data/datasets@data.train=MUSE_forget \
        model.model_args.pretrained_model_name_or_path=/cnz/data/ms-home/Llama-2-7b-hf/ \
        data.train.MUSE_forget.args.hf_args.path=${dataset_path} \
        data.train.MUSE_forget.args.hf_args.split="forget" \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.num_train_epochs=10 \
        trainer.args.save_strategy=no

    done

done
