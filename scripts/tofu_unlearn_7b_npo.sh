#!/bin/bash


export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    "Llama-2-7b-chat-hf"
    # "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
)
trainers_experiments=(
    "NPO unlearn/tofu/default.yaml"
    # "GradDiff unlearn/tofu/default.yaml"
    # "GradAscent unlearn/tofu/default.yaml"
    # "SimNPO unlearn/tofu/default.yaml"
    # "DPO unlearn/tofu/idk.yaml"
    # "RMU  unlearn/tofu/default.yaml"
    # "SatImp  unlearn/tofu/default.yaml"
    # "NullSpace unlearn/tofu/default.yaml"
)
splits=(
    # "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    # "forget10 holdout10 retain90"
)


per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=2
epoch=5


########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################


for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            trainer=$(echo $trainer_experiment | cut -d' ' -f1)
            experiment=$(echo $trainer_experiment | cut -d' ' -f2)
            
            task_name=tofu_${model}_${forget_split}_${trainer}_beta0.2_gamma0.2_epoch${epoch}
            model_path=open-unlearning/tofu_${model}_full
            echo ${task_name}: Unlearning ${model_path} using ${trainer}

            # Unlearn
            CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
            src/train.py --config-name=unlearn.yaml \
            experiment=${experiment} \
            trainer=${trainer} \
            task_name=${task_name} \
            model=${model} \
            forget_split=${forget_split} \
            retain_split=${retain_split} \
            model.model_args.pretrained_model_name_or_path=/cnz/data/project/my-open-unlearning/saves/finetune/tofu_Llama-2-7b-chat-hf_full \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            trainer.method_args.beta=0.2 \
            trainer.method_args.gamma=0.2 \
            trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
            trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=true  \
            trainer.args.num_train_epochs=${epoch} \
            trainer.args.save_strategy="no" \
            # trainer.args.save_steps=0.5 
            # trainer.args.save_steps=24 \

            # Eval
            CUDA_VISIBLE_DEVICES=1 python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=${forget_split} \
            holdout_split=${holdout_split} \
            model=${model} \
            task_name=${task_name} \
            model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
            paths.output_dir=saves/unlearn/${task_name}/evals \
            retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
        done
    done
done