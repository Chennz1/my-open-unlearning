#!/bin/bash

# Post-processing: Fine-tune final iterative Task Vector models on retain data
# This script takes the final models from iterative task vector experiments
# and fine-tunes them on retain data to potentially improve model utility

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# Configuration
BASE_DIR="saves/iterative_task_vector"
OUTPUT_DIR="saves/iterative_task_vector_retain_finetune"
DEVICE="0"

# Experiments to process
experiments=(
    "forget01 holdout01 retain99"
    "forget10 holdout10 retain90"
)

# Fine-tuning parameters
EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=1e-5

echo "========================================"
echo "POST-PROCESSING: RETAIN FINE-TUNING"
echo "========================================"
echo "Base directory: $BASE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Fine-tuning epochs: $EPOCHS"
echo "Device: $DEVICE"
echo "========================================"

# Create output directory
mkdir -p $OUTPUT_DIR

for experiment in "${experiments[@]}"; do
    forget_split=$(echo $experiment | cut -d' ' -f1)
    holdout_split=$(echo $experiment | cut -d' ' -f2)
    retain_split=$(echo $experiment | cut -d' ' -f3)
    
    echo ""
    echo "========================================"
    echo "PROCESSING EXPERIMENT: $forget_split"
    echo "========================================"
    echo "Forget split: $forget_split"
    echo "Holdout split: $holdout_split"
    echo "Retain split: $retain_split"
    echo "========================================"
    
    # Path to the final iterative model
    final_model_path="$BASE_DIR/$forget_split/iteration_3/updated_combined_model"
    
    # Check if final model exists
    if [ ! -d "$final_model_path" ]; then
        echo "❌ Final model not found: $final_model_path"
        echo "Skipping $forget_split"
        continue
    fi
    
    echo "✅ Found final model: $final_model_path"
    
    # Create output directory for this experiment
    exp_output_dir="$OUTPUT_DIR/$forget_split"
    mkdir -p $exp_output_dir
    
    # Fine-tune on retain data with different epoch counts
    for epochs in 1 2 3; do
        echo ""
        echo "----------------------------------------"
        echo "Fine-tuning with $epochs epochs"
        echo "----------------------------------------"
        
        output_path="$exp_output_dir/retain_finetune_${epochs}epochs"
        
        echo "Running retain fine-tuning..."
        echo "Input model: $final_model_path"
        echo "Output path: $output_path"
        echo "Retain split: $retain_split"
        echo "Epochs: $epochs"
        
        # Fine-tune command
        CUDA_VISIBLE_DEVICES=$DEVICE accelerate launch \
            --config_file configs/accelerate/single_gpu_config.yaml \
            --main_process_port $MASTER_PORT \
            src/train.py experiment=finetune/tofu/default.yaml \
            task_name=iterative_tv_${forget_split}_retain_ft_${epochs}ep \
            model=Llama-3.2-1B-Instruct \
            model.model_args.pretrained_model_name_or_path=$final_model_path \
            data/datasets@data.train=TOFU_QA_retain \
            data.train.TOFU_QA_retain.args.hf_args.name=$retain_split \
            trainer.args.num_train_epochs=$epochs \
            trainer.args.per_device_train_batch_size=$BATCH_SIZE \
            trainer.args.learning_rate=$LEARNING_RATE \
            trainer.args.gradient_checkpointing=true \
            trainer.args.save_strategy=epoch \
            trainer.args.save_only_model=true \
            paths.output_dir=$output_path
        
        if [ $? -eq 0 ]; then
            echo "✅ Fine-tuning completed successfully for $epochs epochs"
            
            # Find the final checkpoint or use the main model directory
            if [ -d "$output_path/checkpoint-$epochs" ]; then
                model_for_eval="$output_path/checkpoint-$epochs"
            else
                model_for_eval="$output_path"
            fi
            
            echo "Evaluating fine-tuned model: $model_for_eval"
            
            # Evaluate the fine-tuned model
            eval_output_dir="$output_path/eval_results"
            mkdir -p $eval_output_dir
            
            CUDA_VISIBLE_DEVICES=$DEVICE python src/eval.py \
                experiment=eval/tofu/default.yaml \
                forget_split=$forget_split \
                holdout_split=$holdout_split \
                task_name=iterative_tv_${forget_split}_retain_ft_${epochs}ep_eval \
                model=Llama-3.2-1B-Instruct \
                model.model_args.pretrained_model_name_or_path=$model_for_eval \
                retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_${retain_split}/TOFU_EVAL.json \
                paths.output_dir=$eval_output_dir
            
            if [ $? -eq 0 ]; then
                echo "✅ Evaluation completed successfully"
            else
                echo "❌ Evaluation failed"
            fi
            
        else
            echo "❌ Fine-tuning failed for $epochs epochs"
        fi
    done
    
    echo "Completed processing for $forget_split"
done

echo ""
echo "========================================"
echo "POST-PROCESSING COMPLETED"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"

# Generate comparison analysis
echo "Generating comparison analysis..."
python analyze_retain_finetune_results.py --results_dir $OUTPUT_DIR

echo "Analysis complete. Check results in: $OUTPUT_DIR"