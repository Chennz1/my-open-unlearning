#!/bin/bash


export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    "phi-1_5"
)
splits=(
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

per_device_train_batch_size=4
gradient_accumulation_steps=2
epoch=5


########################################################################################################################
################################### Unified Grid Search for Multiple Methods ##########################################
########################################################################################################################

# Define trainers with their experiments and parameter grids
declare -A trainer_configs

# NPO: β in [0.05, 0.2], γ in [0.2, 0.5]
trainer_configs["NPO"]="unlearn/tofu/default.yaml|0.05,0.1,0.15,0.2|0.2,0.3,0.4,0.5"

# SimNPO: β in [2.0, 3.0], γ in [0.2, 0.3]
trainer_configs["SimNPO"]="unlearn/tofu/default.yaml|2.0,2.25,2.5|0.1,0.125,0.15,0.175,0.2"

# GradDiff: can add other methods here with their parameter ranges
# trainer_configs["GradDiff"]="unlearn/tofu/default.yaml|0.5,1.0|0.5,1.0,1.5"

# GradAscent: single parameter or no grid search
# trainer_configs["GradAscent"]="unlearn/tofu/default.yaml|1.0|1.0"


for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for trainer in "${!trainer_configs[@]}"; do
            # Parse trainer configuration
            config_line="${trainer_configs[$trainer]}"
            experiment=$(echo $config_line | cut -d'|' -f1)
            beta_values=$(echo $config_line | cut -d'|' -f2)
            gamma_values=$(echo $config_line | cut -d'|' -f3)
            
            # Convert comma-separated strings to arrays
            IFS=',' read -ra betas <<< "$beta_values"
            IFS=',' read -ra gammas <<< "$gamma_values"
            
            echo "========================================="
            echo "Starting grid search for ${trainer}"
            echo "Beta values: ${betas[@]}"
            echo "Gamma values: ${gammas[@]}"
            echo "========================================="
            
            for beta in "${betas[@]}"; do
                for gamma in "${gammas[@]}"; do
                    task_name=tofu_${model}_${forget_split}_${trainer}_beta${beta}_gamma${gamma}_epoch${epoch}
                    model_path=/cnz/data/project/my-open-unlearning/saves/finetune/tofu_phi-1_5_full
                    echo ""
                    echo ">>> ${task_name}: Unlearning with ${trainer} (beta=${beta}, gamma=${gamma})"

                    # # Unlearn
                    # CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
                    # src/train.py --config-name=unlearn.yaml \
                    # experiment=${experiment} \
                    # trainer=${trainer} \
                    # task_name=${task_name} \
                    # model=${model} \
                    # forget_split=${forget_split} \
                    # retain_split=${retain_split} \
                    # model.model_args.pretrained_model_name_or_path=${model_path} \
                    # retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
                    # trainer.method_args.beta=${beta} \
                    # trainer.method_args.gamma=${gamma} \
                    # trainer.args.learning_rate=2e-5 \
                    # trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
                    # trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
                    # trainer.args.ddp_find_unused_parameters=true \
                    # trainer.args.gradient_checkpointing=true \
                    # trainer.args.num_train_epochs=${epoch} \
                    # trainer.args.save_strategy="no"

                    # if [ $? -ne 0 ]; then
                    #     echo "ERROR: Training failed for ${task_name}"
                    #     continue
                    # fi

                    # Check if evaluation already exists
                    if [ -d "saves/unlearn/${task_name}/evals" ] && [ -f "saves/unlearn/${task_name}/evals/TOFU_EVAL.json" ]; then
                        echo ">>> Evaluation results already exist for ${task_name}, skipping..."
                    # Check if model exists (training completed)
                    elif [ ! -d "saves/unlearn/${task_name}" ]; then
                        echo ">>> Model not found for ${task_name}, skipping (training may not be completed)..."
                    else
                        # Eval
                        echo ">>> Evaluating ${task_name}..."
                        CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                        experiment=eval/tofu/default.yaml \
                        forget_split=${forget_split} \
                        holdout_split=${holdout_split} \
                        model=${model} \
                        task_name=${task_name} \
                        model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
                        paths.output_dir=saves/unlearn/${task_name}/evals \
                        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

                        if [ $? -ne 0 ]; then
                            echo "ERROR: Evaluation failed for ${task_name}"
                        else
                            # Delete model weights but keep evaluation results
                            echo ">>> Cleaning up model weights for ${task_name}..."
                            if [ -d "saves/unlearn/${task_name}" ]; then
                                # Keep only the evals directory
                                find saves/unlearn/${task_name} -mindepth 1 -maxdepth 1 ! -name 'evals' -exec rm -rf {} +
                                echo "Model weights deleted, evaluation results preserved in saves/unlearn/${task_name}/evals"
                            fi
                        fi
                    fi
                    
                    echo ">>> Completed ${task_name}"
                    echo ""
                done
            done
            
            echo "========================================="
            echo "Completed grid search for ${trainer}"
            echo "========================================="
            echo ""
        done
    done
done

echo "All grid search experiments completed!"
