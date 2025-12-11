#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


models=(
    # "Llama-3.2-1B-base"
    # "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
    "phi-1_5"
)
per_device_train_batch_size=4 # Effective batch size 32 on two GPUs with gradent_accumulation_steps=8

# Base size used to compute the trimmed count for HF split, e.g. retain90 -> 4000 * (1-0.9) = 400
tofu_trim_base=4000

splits=(
    # "forget01 holdout01 retain99"
    # "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)



########################################################################################################################
########################################### FORGET Finetuned TOFU ######################################################
########################################################################################################################



for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    # Extract retain percentage from token like "retain90" -> 90
    retain_pct=${retain_split#retain}
    if [[ -z "$retain_pct" || ! "$retain_pct" =~ ^[0-9]+$ ]]; then
        echo "Failed to parse retain percentage from: $retain_split" >&2
        exit 1
    fi
    # Compute trim count: base * (100 - retain%)/100
    trim_count=$(( tofu_trim_base * (100 - retain_pct) / 100 ))
    echo "Split=$split -> retain%=$retain_pct => trim_count=$trim_count"W
    
    
    for model in "${models[@]}"; do
        CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
        src/train.py experiment=finetune/tofu/default.yaml \
        task_name=tofu_${model}_${retain_split}_mimic \
        model=${model} \
        data/datasets@data.train=TOFU_QA_retain \
        data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
        data.train.TOFU_QA_retain.args.hf_args.split="train\[:${trim_count}\]" \
        trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.num_train_epochs=10 \
        trainer.args.gradient_checkpointing=true \
        trainer.args.save_strategy=no 


    
        # CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
        # forget_split=${forget_split} \
        # holdout_split=${holdout_split} \
        # task_name=tofu_${model}_${retain_split} \
        # model=${model} \
        # model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_${retain_split}
    done
done

