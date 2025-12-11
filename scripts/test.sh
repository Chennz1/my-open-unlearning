#!/bin/bash


export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
)
trainers_experiments=(
    # "GradAscent unlearn/tofu/default.yaml"
    # "GradDiff unlearn/tofu/default.yaml"
    # "NPO unlearn/tofu/default.yaml"
    "DPO unlearn/tofu/idk.yaml"
    # "RMU  unlearn/tofu/default.yaml"
)
splits=(
    # "forget01 holdout01 retain99"
    # "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)


per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=4


########################################################################################################################
########################################### Unlearn TOFU models ########################################################
######################################################################################################################### eval/tofu/default.yaml \


for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            echo $(echo $trainer_experiment | cut -d' ' -f2)
            task_name=tofu_${model}_${forget_split}_${trainer} 

            # Eval
            CUDA_VISIBLE_DEVICES=0 python src/test.py \
            experiment=$(echo $trainer_experiment | cut -d' ' -f2) \
            forget_split=${forget_split} \
            model=${model} \
            model.model_args.pretrained_model_name_or_path=/home/cnz/.cache/huggingface/hub/Llama-3___2-1B-Instruct \
            task_name=${task_name} 
        done
    done
done

            # holdout_split=${holdout_split} \
