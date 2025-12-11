#!/bin/bash

# Quick test script for retain fine-tuning
# Tests with just 1 epoch on forget01 experiment

echo "========================================"
echo "RETAIN FINE-TUNING - QUICK TEST"
echo "========================================"

# Configuration
FINAL_MODEL="saves/iterative_task_vector/forget01/iteration_3/updated_combined_model"
OUTPUT_DIR="saves/retain_finetune_test"
FORGET_SPLIT="forget01"
HOLDOUT_SPLIT="holdout01"
RETAIN_SPLIT="retain99"
DEVICE="0"

# Check if final model exists
if [ ! -d "$FINAL_MODEL" ]; then
    echo "❌ Final iterative model not found: $FINAL_MODEL"
    echo "Please run the iterative task vector experiment first."
    exit 1
fi

echo "✅ Found final iterative model: $FINAL_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Testing with $RETAIN_SPLIT data for 1 epoch"

# Export for subprocess
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

echo ""
echo "Starting retain fine-tuning test..."

# Fine-tune for 1 epoch
CUDA_VISIBLE_DEVICES=$DEVICE accelerate launch \
    --config_file configs/accelerate/single_gpu_config.yaml \
    --main_process_port $MASTER_PORT \
    src/train.py experiment=finetune/tofu/default.yaml \
    task_name=iterative_tv_retain_ft_test \
    model=Llama-3.2-1B-Instruct \
    model.model_args.pretrained_model_name_or_path=$FINAL_MODEL \
    data/datasets@data.train=TOFU_QA_retain \
    data.train.TOFU_QA_retain.args.hf_args.name=$RETAIN_SPLIT \
    trainer.args.num_train_epochs=1 \
    trainer.args.per_device_train_batch_size=8 \
    trainer.args.learning_rate=1e-5 \
    trainer.args.gradient_checkpointing=true \
    trainer.args.save_strategy=epoch \
    trainer.args.save_only_model=true \
    paths.output_dir=$OUTPUT_DIR

if [ $? -eq 0 ]; then
    echo "✅ Retain fine-tuning completed successfully!"
    
    # Determine model path for evaluation
    if [ -d "$OUTPUT_DIR/checkpoint-1" ]; then
        MODEL_FOR_EVAL="$OUTPUT_DIR/checkpoint-1"
    else
        MODEL_FOR_EVAL="$OUTPUT_DIR"
    fi
    
    echo "Evaluating fine-tuned model: $MODEL_FOR_EVAL"
    
    # Evaluate the fine-tuned model
    CUDA_VISIBLE_DEVICES=$DEVICE python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=$FORGET_SPLIT \
        holdout_split=$HOLDOUT_SPLIT \
        task_name=iterative_tv_retain_ft_test_eval \
        model=Llama-3.2-1B-Instruct \
        model.model_args.pretrained_model_name_or_path=$MODEL_FOR_EVAL \
        retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_$RETAIN_SPLIT/TOFU_EVAL.json \
        paths.output_dir=$OUTPUT_DIR/eval_results

    if [ $? -eq 0 ]; then
        echo "✅ Evaluation completed successfully!"
        
        # Compare results
        echo ""
        echo "========================================"
        echo "RESULTS COMPARISON"
        echo "========================================"
        
        # Original iterative model results
        ORIG_SUMMARY="saves/iterative_task_vector/forget01/iteration_3/iteration_3_combined_eval/TOFU_SUMMARY.json"
        
        # Fine-tuned model results  
        FT_SUMMARY="$OUTPUT_DIR/eval_results/TOFU_SUMMARY.json"
        
        if [ -f "$ORIG_SUMMARY" ] && [ -f "$FT_SUMMARY" ]; then
            echo "Original Model (Iteration 3):"
            python -c "
import json
with open('$ORIG_SUMMARY', 'r') as f:
    data = json.load(f)
print(f'  Forget Quality: {data.get(\"forget_quality\", \"N/A\")}')
print(f'  Model Utility: {data.get(\"model_utility\", \"N/A\")}')
print(f'  Extraction Strength: {data.get(\"extraction_strength\", \"N/A\")}')
"
            
            echo ""
            echo "After 1 Epoch Retain Fine-tuning:"
            python -c "
import json
with open('$FT_SUMMARY', 'r') as f:
    data = json.load(f)
print(f'  Forget Quality: {data.get(\"forget_quality\", \"N/A\")}')
print(f'  Model Utility: {data.get(\"model_utility\", \"N/A\")}')  
print(f'  Extraction Strength: {data.get(\"extraction_strength\", \"N/A\")}')
"

            echo ""
            echo "Change Analysis:"
            python -c "
import json

with open('$ORIG_SUMMARY', 'r') as f:
    orig = json.load(f)
with open('$FT_SUMMARY', 'r') as f:
    ft = json.load(f)

orig_mu = orig.get('model_utility', 0)
ft_mu = ft.get('model_utility', 0)
mu_change = ((ft_mu - orig_mu) / orig_mu) * 100 if orig_mu > 0 else 0

orig_fq = orig.get('forget_quality', 0) 
ft_fq = ft.get('forget_quality', 0)
fq_change = ((ft_fq - orig_fq) / orig_fq) * 100 if orig_fq > 0 else 0

print(f'  Model Utility Change: {mu_change:+.2f}%')
print(f'  Forget Quality Change: {fq_change:+.2f}%')

if mu_change > 0:
    print('  🎯 Retain fine-tuning improved model utility!')
else:
    print('  📊 Original model had better utility')
"
            
        else
            echo "❌ Could not find summary files for comparison"
        fi
        
    else
        echo "❌ Evaluation failed"
    fi
    
else
    echo "❌ Retain fine-tuning failed"
    exit 1
fi

echo ""
echo "========================================"
echo "TEST COMPLETED"
echo "========================================"
echo "If successful, you can run the full experiment with:"
echo "bash scripts/retain_finetune_iterative_models.sh"