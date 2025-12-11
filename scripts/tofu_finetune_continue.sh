#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


models=(
    "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
)
per_device_train_batch_size=4 # Effective batch size 32 on two GPUs with gradent_accumulation_steps=8

splits=(
    # "forget01 holdout01 retain99"
    # "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)



########################################################################################################################
########################################### RETAIN Finetuned TOFU ######################################################
########################################################################################################################

# for split in "${splits[@]}"; do
#     forget_split=$(echo $split | cut -d' ' -f1)
#     holdout_split=$(echo $split | cut -d' ' -f2)
#     retain_split=$(echo $split | cut -d' ' -f3)
    
#     for model in "${models[@]}"; do
#         CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
#         src/train.py experiment=finetune/tofu/default.yaml \
#         task_name=tofu_${model}_${retain_split} \
#         model=${model} \
#         data/datasets@data.train=TOFU_QA_retain \
#         data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
#         trainer.args.per_device_train_batch_size=4 \
#         trainer.args.ddp_find_unused_parameters=true \
#         trainer.args.gradient_checkpointing=true

    
#         # CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
#         # forget_split=${forget_split} \
#         # holdout_split=${holdout_split} \
#         # task_name=tofu_${model}_${retain_split} \
#         # model=${model} \
#         # model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_${retain_split}
#     done
# done


# ########################################################################################################################
# ########################################### FULL Finetuned TOFU models #################################################
# ########################################################################################################################
export HYDRA_FULL_ERROR=0

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        forget_split=$(echo $split | cut -d' ' -f1)
        holdout_split=$(echo $split | cut -d' ' -f2)
        retain_split=$(echo $split | cut -d' ' -f3)

        CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
        src/train.py experiment=finetune/tofu/default.yaml \
        task_name=tofu_${model}_continue \
        model=${model} \
        data/datasets@data.train=TOFU_QA_full \
        data.train.TOFU_QA_full.args.hf_args.name=full \
        model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_full \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true 
        # trainer.args.save_strategy=no 


        # # CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
        # CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerate/single_gpu_config.yaml --main_process_port $MASTER_PORT \
        # src/train.py experiment=finetune/tofu/default.yaml \
        # task_name=tofu_${model}_continue_${retain_split} \
        # model=${model} \
        # data/datasets@data.train=TOFU_QA_retain \
        # data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
        # model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_full \
        # trainer.args.per_device_train_batch_size=4 \
        # trainer.args.ddp_find_unused_parameters=true \
        # trainer.args.gradient_checkpointing=true \
        # trainer.args.save_strategy=epoch 


        # # CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
        # CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file configs/accelerate/single_gpu_config.yaml --main_process_port $MASTER_PORT \
        # src/train.py experiment=finetune/tofu/default.yaml \
        # task_name=tofu_${model}_continue_${forget_split} \
        # model=${model} \
        # data/datasets@data.train=TOFU_QA_forget \
        # data.train.TOFU_QA_forget.args.hf_args.name=${forget_split} \
        # model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_full \
        # trainer.args.per_device_train_batch_size=8 \
        # trainer.args.num_train_epochs=5 \
        # trainer.args.ddp_find_unused_parameters=true \
        # trainer.args.gradient_checkpointing=true \
        # trainer.args.save_strategy=epoch  


        # # Evaluate the full models on each forget split
        # for split in "${splits[@]}"; do
        #     forget_split=$(echo $split | cut -d' ' -f1)
        #     holdout_split=$(echo $split | cut -d' ' -f2)
        #     retain_split=$(echo $split | cut -d' ' -f3)

        #     CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
        #     forget_split=${forget_split} \
        #     holdout_split=${holdout_split} \
        #     task_name=tofu_${model}_full_${forget_split} \
        #     model=${model} \
        #     model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_full \
        #     retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
        #     paths.output_dir=saves/eval/tofu_${model}_full/evals_${forget_split}


        
        # # Evaluate the epoch on each forget split
        # for split in "${splits[@]}"; do
        #     forget_split=$(echo $split | cut -d' ' -f1)
        #     holdout_split=$(echo $split | cut -d' ' -f2)
        #     retain_split=$(echo $split | cut -d' ' -f3)

        #     # CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
        #     # forget_split=${forget_split} \
        #     # holdout_split=${holdout_split} \
        #     # task_name=tofu_${model}_continue_${forget_split} \
        #     # model=${model} \
        #     # model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_continue_${forget_split} \
        #     # retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
        #     # paths.output_dir=saves/finetune/tofu_${model}_continue_${forget_split}/evals

        #     # Evaluate all checkpoints
        #     checkpoint_dir="saves/finetune/tofu_${model}_continue_${forget_split}"
        #     if [ -d "$checkpoint_dir" ]; then
        #     for checkpoint in "$checkpoint_dir"/checkpoint-*; do
        #         if [ -d "$checkpoint" ]; then
        #         checkpoint_name=$(basename "$checkpoint")
        #         echo "Evaluating checkpoint: $checkpoint"
        #         CUDA_VISIBLE_DEVICES=1 python src/eval.py experiment=eval/tofu/default.yaml \
        #         forget_split=${forget_split} \
        #         holdout_split=${holdout_split} \
        #         task_name=tofu_${model}_continue_${forget_split}_${checkpoint_name} \
        #         model=${model} \
        #         model.model_args.pretrained_model_name_or_path="$checkpoint" \
        #         retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
        #         paths.output_dir=saves/finetune/tofu_${model}_continue_${forget_split}/evals_${checkpoint_name}
        #         fi
        #     done
        #     fi
        # done
    done
done