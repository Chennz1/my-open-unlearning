#!/bin/bash

################################################################################
# TVC (Task Vector Composition) 整合实验脚本 - 存储优化版
# 目标：验证 α=β=1 的无需调参特性，节省存储空间
# 策略：训练 -> 融合 -> 评估 -> 删除，串行执行
################################################################################

set -e  # Exit on error

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

################################################################################
# 实验配置
################################################################################

# 模型配置
models=(
    "phi-1_5"
)

# 数据集分割配置
splits=(
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# 训练超参数
LEARNING_RATE=2e-5
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
NUM_TRAIN_EPOCHS=5
SAVE_STRATEGY=no

# GPU配置
TRAIN_GPUS="0,1"
EVAL_GPU="0"

# 路径配置
BASE_MODEL_PATH="/cnz/data/hf-home/hub/hub/models--microsoft--phi-1_5/snapshots/77aa61eeac94fbf33d492b9f2744c98b42d5b5eb/"
SAVE_BASE_DIR="/cnz/data/project/my-open-unlearning/saves"
RESULTS_DIR="$SAVE_BASE_DIR/tvc_results"

# 创建结果目录
mkdir -p $RESULTS_DIR

################################################################################
# 实验参数配置
################################################################################

# 默认配置
DEFAULT_ALPHA=1.0
DEFAULT_BETA=1.0

# 参数扫描配置（可根据需要调整）
ALPHA_SWEEP=(0.5 0.8 1.0 1.2 1.5 2.0)
BETA_SWEEP=(0.5 0.8 1.0 1.2 1.5 2.0)

# 数据比例扫描配置
DATA_RATIO_CONFIGS=(
    "ratio_1_0.1 0.1"
    "ratio_1_0.5 0.5"
    "ratio_1_1 1.0"
    "ratio_1_2 2.0"
    "ratio_1_5 5.0"
)

TOFU_TRIM_BASE=4000

################################################################################
# 辅助函数
################################################################################

calculate_trim_count() {
    local retain_split=$1
    local ratio_multiplier=$2
    local retain_pct=${retain_split#retain}
    if [[ -z "$retain_pct" || ! "$retain_pct" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Failed to parse retain percentage from: $retain_split" >&2
        return 1
    fi
    local base_trim=$(( TOFU_TRIM_BASE * (100 - retain_pct) / 100 ))
    local trim_count=$(python -c "print(int($base_trim * $ratio_multiplier))")
    echo $trim_count
}

log_message() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" >> "$RESULTS_DIR/experiment.log"
}

################################################################################
# 核心函数：单次完整实验流程
################################################################################

run_single_experiment() {
    local model=$1
    local forget_split=$2
    local retain_split=$3
    local holdout_split=$4
    local alpha=$5
    local beta=$6
    local ratio_mult=$7
    local ratio_label=${8:-"default"}
    local exp_name=$9
    
    # 检测是否已完成（断点续传机制）
    local eval_output_dir_all="$RESULTS_DIR/${exp_name}_all_linear_eval"
    local eval_output_dir_mlp="$RESULTS_DIR/${exp_name}_mlp_only_eval"
    
    local all_completed=false
    local mlp_completed=false
    
    # 检查 all_linear 评估结果
    if [ -d "$eval_output_dir_all" ] && [ -f "$eval_output_dir_all/TOFU_EVAL.json" ]; then
        log_message "检测到已完成的评估 (all_linear): $exp_name"
        all_completed=true
    fi
    
    # 检查 mlp_only 评估结果
    if [ -d "$eval_output_dir_mlp" ] && [ -f "$eval_output_dir_mlp/TOFU_EVAL.json" ]; then
        log_message "检测到已完成的评估 (mlp_only): $exp_name"
        mlp_completed=true
    fi
    
    # 如果两个都完成，直接跳过
    if [ "$all_completed" = true ] && [ "$mlp_completed" = true ]; then
        log_message "实验已完成，跳过: $exp_name"
        log_message "========================================"
        echo ""
        return 0
    fi
    
    log_message "========================================"
    log_message "开始实验: $exp_name"
    log_message "配置: α=$alpha, β=$beta, 比例=$ratio_mult"
    if [ "$all_completed" = true ]; then
        log_message "状态: all_linear已完成，将重新训练mlp_only"
    elif [ "$mlp_completed" = true ]; then
        log_message "状态: mlp_only已完成，将重新训练all_linear"
    else
        log_message "状态: 全新实验"
    fi
    log_message "========================================"
    
    # 计算trim数量
    local trim_count=$(calculate_trim_count "$retain_split" "$ratio_mult")
    
    # 定义临时模型路径
    local forget_model_path="$SAVE_BASE_DIR/temp/forget_${exp_name}"
    local retain_model_path="$SAVE_BASE_DIR/temp/retain_${exp_name}"
    local merged_model_path="$SAVE_BASE_DIR/temp/merged_${exp_name}"
    local ft_model_path="$SAVE_BASE_DIR/finetune/tofu_${model}_full"
    
    # 创建临时目录
    mkdir -p "$SAVE_BASE_DIR/temp"
    
    # 步骤1: 训练Forget模型
    log_message "步骤1: 训练Forget模型"
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
        --config_file configs/accelerate/default_config.yaml \
        --main_process_port $MASTER_PORT \
        src/train.py experiment=finetune/tofu/default.yaml \
        task_name=temp_forget \
        model=$model \
        data/datasets@data.train=TOFU_QA_forget \
        data.train.TOFU_QA_forget.args.hf_args.name=$forget_split \
        model.model_args.pretrained_model_name_or_path=$BASE_MODEL_PATH \
        trainer.args.learning_rate=$LEARNING_RATE \
        trainer.args.per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
        trainer.args.gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.num_train_epochs=$NUM_TRAIN_EPOCHS \
        trainer.args.gradient_checkpointing=true \
        trainer.args.save_strategy=no \
        paths.output_dir=$forget_model_path
    
    if [ $? -ne 0 ]; then
        log_message "错误: Forget模型训练失败"
        return 1
    fi
    
    # 步骤2: 训练Retain模型
    log_message "步骤2: 训练Retain模型 (trim=$trim_count)"
    local split_arg="train\\[:${trim_count}\\]"
    
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
        --config_file configs/accelerate/default_config.yaml \
        --main_process_port $MASTER_PORT \
        src/train.py experiment=finetune/tofu/default.yaml \
        task_name=temp_retain \
        model=$model \
        data/datasets@data.train=TOFU_QA_retain \
        data.train.TOFU_QA_retain.args.hf_args.name=$retain_split \
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
    
    if [ $? -ne 0 ]; then
        log_message "错误: Retain模型训练失败"
        rm -rf "$forget_model_path"
        return 1
    fi
    
    # 步骤3: 模型融合（全参数 & 仅MLP 两种）
    # 3a: 全参数（所有 linear 层）
    log_message "步骤3a: 模型融合 (all_linear)"
    local merged_all_path="$SAVE_BASE_DIR/temp/merged_${exp_name}_all_linear"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/model_merge_filter.py \
        --tokenizer_path $BASE_MODEL_PATH \
        --orig_model_path $BASE_MODEL_PATH \
        --ft_model_path $ft_model_path \
        --forget_path $forget_model_path \
        --retain_mimic_path $retain_model_path \
        --save_path $merged_all_path \
        --filter_name all_linear \
        --alpha $alpha \
        --beta $beta \
        --device 0

    if [ $? -ne 0 ]; then
        log_message "错误: 模型融合失败 (all_linear)"
        rm -rf "$forget_model_path" "$retain_model_path"
        return 1
    fi

    # 3b: 仅 MLP 层
    log_message "步骤3b: 模型融合 (mlp_only)"
    local merged_mlp_path="$SAVE_BASE_DIR/temp/merged_${exp_name}_mlp_only"
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/model_merge_filter.py \
        --tokenizer_path $BASE_MODEL_PATH \
        --orig_model_path $BASE_MODEL_PATH \
        --ft_model_path $ft_model_path \
        --forget_path $forget_model_path \
        --retain_mimic_path $retain_model_path \
        --save_path $merged_mlp_path \
        --filter_name mlp_only \
        --alpha $alpha \
        --beta $beta \
        --device 0

    if [ $? -ne 0 ]; then
        log_message "错误: 模型融合失败 (mlp_only)"
        rm -rf "$forget_model_path" "$retain_model_path" "$merged_all_path"
        return 1
    fi

    # 步骤4: 分别评估两种融合模型
    local retain_logs_path="$SAVE_BASE_DIR/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

    # 4a: 评估 all_linear 模型
    if [ "$all_completed" = false ]; then
        log_message "步骤4a: 评估融合模型 (all_linear)"
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=$forget_split \
            holdout_split=$holdout_split \
            model=$model \
            task_name=${exp_name}_all_linear \
            model.model_args.pretrained_model_name_or_path=$merged_all_path \
            paths.output_dir=$eval_output_dir_all \
            retain_logs_path=$retain_logs_path

        if [ $? -ne 0 ]; then
            log_message "警告: 评估失败 (all_linear)，但继续后续流程"
        else
            log_message "评估成功 (all_linear): $eval_output_dir_all/TOFU_EVAL.json"
        fi
    else
        log_message "步骤4a: 跳过已完成的评估 (all_linear)"
    fi

    # 4b: 评估 mlp_only 模型
    if [ "$mlp_completed" = false ]; then
        log_message "步骤4b: 评估融合模型 (mlp_only)"
        CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split=$forget_split \
            holdout_split=$holdout_split \
            model=$model \
            task_name=${exp_name}_mlp_only \
            model.model_args.pretrained_model_name_or_path=$merged_mlp_path \
            paths.output_dir=$eval_output_dir_mlp \
            retain_logs_path=$retain_logs_path

        if [ $? -ne 0 ]; then
            log_message "警告: 评估失败 (mlp_only)，但继续清理"
        else
            log_message "评估成功 (mlp_only): $eval_output_dir_mlp/TOFU_EVAL.json"
        fi
    else
        log_message "步骤4b: 跳过已完成的评估 (mlp_only)"
    fi

    # 步骤5: 清理临时模型（保留评估结果）
    log_message "步骤5: 清理临时模型"
    rm -rf "$forget_model_path"
    rm -rf "$retain_model_path"
    rm -rf "$merged_all_path"
    rm -rf "$merged_mlp_path"
    
    log_message "实验完成: $exp_name"
    log_message "评估结果保存在: $eval_output_dir"
    log_message "========================================"
    echo ""
    
    return 0
}

################################################################################
# 主流程
################################################################################

main() {
    log_message "========================================"
    log_message "TVC 存储优化实验开始"
    log_message "支持断点续传：检测已完成的评估将自动跳过"
    log_message "========================================"
    
    # 检查必要文件
    if [ ! -f "scripts/model_merge_filter.py" ]; then
        log_message "错误: model_merge_filter.py不存在"
        exit 1
    fi
    
    # 确保full模型存在
    if [ ! -d "$SAVE_BASE_DIR/finetune/tofu_phi-1_5_full" ]; then
        log_message "错误: 需要先训练full模型: tofu_phi-1_5_full"
        exit 1
    fi
    
    local total_experiments=0
    local successful_experiments=0
    local skipped_experiments=0
    
    for split in "${splits[@]}"; do
        forget_split=$(echo $split | cut -d' ' -f1)
        holdout_split=$(echo $split | cut -d' ' -f2)
        retain_split=$(echo $split | cut -d' ' -f3)
        
        for model in "${models[@]}"; do
            # 实验1: 默认配置
            if [[ "${RUN_DEFAULT:-true}" == "true" ]]; then
                total_experiments=$((total_experiments + 1))
                exp_name="tvc_${model}_${forget_split}_default"
                
                # 快速检测是否已完成
                local eval_all="$RESULTS_DIR/${exp_name}_all_linear_eval/TOFU_EVAL.json"
                local eval_mlp="$RESULTS_DIR/${exp_name}_mlp_only_eval/TOFU_EVAL.json"
                if [ -f "$eval_all" ] && [ -f "$eval_mlp" ]; then
                    log_message "跳过已完成实验: $exp_name"
                    skipped_experiments=$((skipped_experiments + 1))
                    successful_experiments=$((successful_experiments + 1))
                elif run_single_experiment "$model" "$forget_split" "$retain_split" "$holdout_split" \
                    "$DEFAULT_ALPHA" "$DEFAULT_BETA" "1.0" "default" "$exp_name"; then
                    successful_experiments=$((successful_experiments + 1))
                fi
            fi
            
            # 实验2: α和β参数扫描
            if [[ "${RUN_COEFF_SWEEP:-false}" == "true" ]]; then
                for alpha in "${ALPHA_SWEEP[@]}"; do
                    for beta in "${BETA_SWEEP[@]}"; do
                        # 跳过默认配置
                        if [[ "$alpha" == "1.0" && "$beta" == "1.0" ]]; then
                            continue
                        fi
                        
                        total_experiments=$((total_experiments + 1))
                        exp_name="tvc_${model}_${forget_split}_alpha${alpha}_beta${beta}"
                        
                        # 快速检测是否已完成
                        local eval_all="$RESULTS_DIR/${exp_name}_all_linear_eval/TOFU_EVAL.json"
                        local eval_mlp="$RESULTS_DIR/${exp_name}_mlp_only_eval/TOFU_EVAL.json"
                        if [ -f "$eval_all" ] && [ -f "$eval_mlp" ]; then
                            log_message "跳过已完成实验: $exp_name"
                            skipped_experiments=$((skipped_experiments + 1))
                            successful_experiments=$((successful_experiments + 1))
                        elif run_single_experiment "$model" "$forget_split" "$retain_split" "$holdout_split" \
                            "$alpha" "$beta" "1.0" "default" "$exp_name"; then
                            successful_experiments=$((successful_experiments + 1))
                        fi
                    done
                done
            fi
            
            # 实验3: 数据比例扫描
            if [[ "${RUN_RATIO_SWEEP:-false}" == "true" ]]; then
                for ratio_config in "${DATA_RATIO_CONFIGS[@]}"; do
                    ratio_label=$(echo $ratio_config | cut -d' ' -f1)
                    ratio_mult=$(echo $ratio_config | cut -d' ' -f2)
                    
                    # 跳过默认比例
                    if [[ "$ratio_mult" == "1.0" ]]; then
                        continue
                    fi
                    
                    total_experiments=$((total_experiments + 1))
                    exp_name="tvc_${model}_${forget_split}_${ratio_label}"
                    
                    # 快速检测是否已完成
                    local eval_all="$RESULTS_DIR/${exp_name}_all_linear_eval/TOFU_EVAL.json"
                    local eval_mlp="$RESULTS_DIR/${exp_name}_mlp_only_eval/TOFU_EVAL.json"
                    if [ -f "$eval_all" ] && [ -f "$eval_mlp" ]; then
                        log_message "跳过已完成实验: $exp_name"
                        skipped_experiments=$((skipped_experiments + 1))
                        successful_experiments=$((successful_experiments + 1))
                    elif run_single_experiment "$model" "$forget_split" "$retain_split" "$holdout_split" \
                        "$DEFAULT_ALPHA" "$DEFAULT_BETA" "$ratio_mult" "$ratio_label" "$exp_name"; then
                        successful_experiments=$((successful_experiments + 1))
                    fi
                done
            fi
        done
    done
    
    # 最终清理
    rm -rf "$SAVE_BASE_DIR/temp"
    
    log_message "========================================"
    log_message "所有实验完成！"
    log_message "总实验数: $total_experiments"
    log_message "成功: $successful_experiments"
    log_message "跳过(已完成): $skipped_experiments"
    log_message "失败: $((total_experiments - successful_experiments))"
    log_message "结果保存在: $RESULTS_DIR"
    log_message "========================================"
    
    # 生成结果摘要
    python scripts/analyze_tvc_results.py \
        --base_dir /cnz/data/project/my-open-unlearning/saves/tvc_results \
        --model phi-1_5 \
        --forget_split forget05 \
        --output tvc_analysis_report.md
}

# 解析命令行参数
RUN_DEFAULT=true
RUN_COEFF_SWEEP=false
RUN_RATIO_SWEEP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --default-only)
            RUN_DEFAULT=true
            RUN_COEFF_SWEEP=false
            RUN_RATIO_SWEEP=false
            shift
            ;;
        --with-coeff-sweep)
            RUN_COEFF_SWEEP=true
            shift
            ;;
        --with-ratio-sweep)
            RUN_RATIO_SWEEP=true
            shift
            ;;
        --all)
            RUN_DEFAULT=true
            RUN_COEFF_SWEEP=true
            RUN_RATIO_SWEEP=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--default-only] [--with-coeff-sweep] [--with-ratio-sweep] [--all]"
            exit 1
            ;;
    esac
done

# 运行主流程
main
