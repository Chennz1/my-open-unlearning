#!/bin/bash

################################################################################
# TVC Filter消融实验脚本
# 目标：测试对模型不同部分应用任务向量的效果
# 固定：α=1.0, β=1.0, 数据比例=1:1
################################################################################

set -e

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

################################################################################
# 配置
################################################################################

MODEL="phi-1_5"
FORGET_SPLIT="forget05"
HOLDOUT_SPLIT="holdout05"
RETAIN_SPLIT="retain95"

LEARNING_RATE=2e-5
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
NUM_TRAIN_EPOCHS=5

TRAIN_GPUS="0,1"
EVAL_GPU="0"

BASE_MODEL_PATH="/cnz/data/hf-home/hub/hub/models--microsoft--phi-1_5/snapshots/77aa61eeac94fbf33d492b9f2744c98b42d5b5eb/"
SAVE_BASE_DIR="/cnz/data/project/my-open-unlearning/saves"
RESULTS_DIR="$SAVE_BASE_DIR/tvc_filter_ablation"

mkdir -p $RESULTS_DIR

# Trim count for retain95 at 1:1 ratio
TRIM_COUNT=200

################################################################################
# Filter配置定义
################################################################################

# 定义所有要测试的filter配置
# 格式: "filter_name:filter_description"
FILTER_CONFIGS=(
    "all_linear:所有线性层(proj+fc+dense+lm_head)"
    "attention_only:仅Attention层(q_proj+k_proj+v_proj+dense)"
    "mlp_only:仅MLP层(fc1+fc2)"
    "qkv_only:仅QKV投影(q_proj+k_proj+v_proj)"
    "output_proj:仅输出投影(dense+fc2)"
    "lm_head_only:仅语言模型头"
    "no_lm_head:除lm_head外所有线性层"
    "attn_and_mlp:Attention和MLP组合"
    "lightweight:轻量级(只q_proj+fc1)"
    "heavy:重量级(所有proj+fc+dense)"
)

################################################################################
# 辅助函数
################################################################################

log_message() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" >> "$RESULTS_DIR/experiment.log"
}

################################################################################
# 训练Forget和Retain模型（只训练一次）
################################################################################

train_base_models() {
    log_message "========================================"
    log_message "训练基础模型（Forget + Retain）"
    log_message "========================================"
    
    local forget_model_path="$SAVE_BASE_DIR/temp/filter_ablation_forget"
    local retain_model_path="$SAVE_BASE_DIR/temp/filter_ablation_retain"
    
    mkdir -p "$SAVE_BASE_DIR/temp"
    
    # 训练Forget模型
    if [ ! -d "$forget_model_path" ]; then
        log_message "训练Forget模型..."
        CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
            --config_file configs/accelerate/default_config.yaml \
            --main_process_port $MASTER_PORT \
            src/train.py experiment=finetune/tofu/default.yaml \
            task_name=filter_ablation_forget \
            model=$MODEL \
            data/datasets@data.train=TOFU_QA_forget \
            data.train.TOFU_QA_forget.args.hf_args.name=$FORGET_SPLIT \
            model.model_args.pretrained_model_name_or_path=$BASE_MODEL_PATH \
            trainer.args.learning_rate=$LEARNING_RATE \
            trainer.args.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
            trainer.args.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.num_train_epochs=$NUM_TRAIN_EPOCHS \
            trainer.args.gradient_checkpointing=true \
            trainer.args.save_strategy=no \
            paths.output_dir=$forget_model_path
    else
        log_message "Forget模型已存在，跳过训练"
    fi
    
    # 训练Retain模型
    if [ ! -d "$retain_model_path" ]; then
        log_message "训练Retain模型 (trim=$TRIM_COUNT)..."
        local split_arg="train\\[:${TRIM_COUNT}\\]"
        
        CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
            --config_file configs/accelerate/default_config.yaml \
            --main_process_port $MASTER_PORT \
            src/train.py experiment=finetune/tofu/default.yaml \
            task_name=filter_ablation_retain \
            model=$MODEL \
            data/datasets@data.train=TOFU_QA_retain \
            data.train.TOFU_QA_retain.args.hf_args.name=$RETAIN_SPLIT \
            data.train.TOFU_QA_retain.args.hf_args.split="$split_arg" \
            model.model_args.pretrained_model_name_or_path=$BASE_MODEL_PATH \
            trainer.args.learning_rate=$LEARNING_RATE \
            trainer.args.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
            trainer.args.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.num_train_epochs=$NUM_TRAIN_EPOCHS \
            trainer.args.gradient_checkpointing=true \
            trainer.args.save_strategy=no \
            paths.output_dir=$retain_model_path
    else
        log_message "Retain模型已存在，跳过训练"
    fi
    
    log_message "基础模型训练完成"
}

################################################################################
# 对单个filter配置进行融合和评估
################################################################################

test_filter_config() {
    local filter_name=$1
    local filter_desc=$2
    
    log_message "========================================"
    log_message "测试Filter: $filter_name"
    log_message "描述: $filter_desc"
    log_message "========================================"
    
    local forget_model_path="$SAVE_BASE_DIR/temp/filter_ablation_forget"
    local retain_model_path="$SAVE_BASE_DIR/temp/filter_ablation_retain"
    local ft_model_path="$SAVE_BASE_DIR/finetune/tofu_${MODEL}_full"
    local merged_model_path="$SAVE_BASE_DIR/temp/merged_${filter_name}"
    
    # 融合模型
    log_message "融合模型 (filter=$filter_name)..."
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/model_merge_filter.py \
        --tokenizer_path $BASE_MODEL_PATH \
        --orig_model_path $BASE_MODEL_PATH \
        --ft_model_path $ft_model_path \
        --forget_path $forget_model_path \
        --retain_mimic_path $retain_model_path \
        --save_path $merged_model_path \
        --filter_name $filter_name \
        --alpha 1.0 \
        --beta 1.0 \
        --device 0
    
    if [ $? -ne 0 ]; then
        log_message "错误: 模型融合失败"
        return 1
    fi
    
    # 评估
    log_message "评估融合模型..."
    local eval_output_dir="$RESULTS_DIR/${filter_name}_eval"
    local retain_logs_path="$SAVE_BASE_DIR/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
    
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=$FORGET_SPLIT \
        holdout_split=$HOLDOUT_SPLIT \
        model=$MODEL \
        task_name=filter_${filter_name} \
        model.model_args.pretrained_model_name_or_path=$merged_model_path \
        paths.output_dir=$eval_output_dir \
        retain_logs_path=$retain_logs_path
    
    # 清理融合模型
    rm -rf "$merged_model_path"
    
    log_message "Filter ${filter_name} 测试完成"
    log_message "========================================"
    echo ""
}

################################################################################
# 主流程
################################################################################

main() {
    log_message "========================================"
    log_message "TVC Filter消融实验开始"
    log_message "========================================"
    
    # 检查必要文件
    if [ ! -f "scripts/model_merge_filter.py" ]; then
        log_message "错误: model_merge_filter.py不存在"
        exit 1
    fi
    
    if [ ! -d "$SAVE_BASE_DIR/finetune/tofu_${MODEL}_full" ]; then
        log_message "错误: 需要先训练full模型"
        exit 1
    fi
    
    # 训练基础模型（只训练一次）
    train_base_models
    
    # 测试每个filter配置
    local total=0
    local successful=0
    
    for config in "${FILTER_CONFIGS[@]}"; do
        filter_name=$(echo $config | cut -d':' -f1)
        filter_desc=$(echo $config | cut -d':' -f2)
        
        total=$((total + 1))
        
        if test_filter_config "$filter_name" "$filter_desc"; then
            successful=$((successful + 1))
        fi
    done
    
    log_message "========================================"
    log_message "所有Filter测试完成！"
    log_message "总配置数: $total"
    log_message "成功: $successful"
    log_message "失败: $((total - successful))"
    log_message "========================================"
    
    # 生成对比报告
    log_message "生成对比报告..."
    python scripts/analyze_filter_results.py \
        --results_dir $RESULTS_DIR \
        --output filter_comparison_report.md
}

# 运行主流程
main

# python scripts/analyze_filter_results.py \
#         --results_dir /cnz/data/project/my-open-unlearning/saves/tvc_filter_ablation \
#         --output filter_comparison_report.md