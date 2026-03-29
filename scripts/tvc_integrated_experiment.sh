#!/bin/bash

################################################################################
# TVC (Task Vector Composition) 整合实验脚本
# 目标：验证 α=β=1 的无需调参特性，并进行系统的参数扫描实验
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
    # "Llama-2-7b-chat-hf"
    # "Llama-3.2-1B-Instruct"
)

# 数据集分割配置
splits=(
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# 训练超参数（关键：保持forget和retain训练的一致性）
LEARNING_RATE=2e-5
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
NUM_TRAIN_EPOCHS=5
SAVE_STRATEGY=no

# GPU配置
TRAIN_GPUS="0,1"  # 训练使用的GPU
EVAL_GPU="0"       # 评估使用的GPU

# 路径配置
BASE_MODEL_PATH="/cnz/data/hf-home/hub/hub/models--microsoft--phi-1_5/snapshots/77aa61eeac94fbf33d492b9f2744c98b42d5b5eb/"
SAVE_BASE_DIR="/cnz/data/project/my-open-unlearning/saves"

################################################################################
# 实验参数配置
################################################################################

# 默认配置（主实验）：α=1.0, β=1.0, 数据比例=1:1
DEFAULT_ALPHA=1.0
DEFAULT_BETA=1.0
DEFAULT_RATIO="1:1"

# 参数扫描配置（消融实验）
ALPHA_SWEEP=(0.5 0.8 1.0 1.2 1.5 2.0)
BETA_SWEEP=(0.5 0.8 1.0 1.2 1.5 2.0)

# 数据比例扫描配置 (retain:forget)
# 格式: "比例标签 forget数据比例"
DATA_RATIO_CONFIGS=(
    "ratio_1_0.1 0.1"   # 1:0.1 -> retain占总数的 91%
    "ratio_1_0.5 0.5"   # 1:0.5 -> retain占总数的 67%
    "ratio_1_1 1.0"     # 1:1   -> retain占总数的 50% (default)
    "ratio_1_2 2.0"     # 1:2   -> retain占总数的 33%
    "ratio_1_5 5.0"     # 1:5   -> retain占总数的 17%
)

# TOFU数据集基础大小
TOFU_TRIM_BASE=4000

################################################################################
# 辅助函数
################################################################################

# 计算retain数据集的trim数量
# 参数: $1=retain_split (如 "retain95"), $2=ratio_multiplier (如 1.0 或 0.5)
calculate_trim_count() {
    local retain_split=$1
    local ratio_multiplier=$2
    
    # 提取retain百分比
    local retain_pct=${retain_split#retain}
    if [[ -z "$retain_pct" || ! "$retain_pct" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Failed to parse retain percentage from: $retain_split" >&2
        return 1
    fi
    
    # 计算基础trim数量: base * (100 - retain%)/100
    local base_trim=$(( TOFU_TRIM_BASE * (100 - retain_pct) / 100 ))
    
    # 根据数据比例调整trim数量（使用Python替代bc）
    local trim_count=$(python -c "print(int($base_trim * $ratio_multiplier))")
    
    echo $trim_count
}

# 创建实验记录文件
create_experiment_log() {
    local log_file="$SAVE_BASE_DIR/experiment_log_$(date +%Y%m%d_%H%M%S).txt"
    echo "TVC Integrated Experiment Log" > $log_file
    echo "Started at: $(date)" >> $log_file
    echo "================================" >> $log_file
    echo $log_file
}

EXPERIMENT_LOG=$(create_experiment_log)
echo "实验日志文件: $EXPERIMENT_LOG"

################################################################################
# 阶段 1: 训练基础模型（Forget和Retain Mimic模型）
################################################################################

train_base_models() {
    echo "=========================================="
    echo "阶段 1: 训练基础模型"
    echo "=========================================="
    
    for split in "${splits[@]}"; do
        forget_split=$(echo $split | cut -d' ' -f1)
        holdout_split=$(echo $split | cut -d' ' -f2)
        retain_split=$(echo $split | cut -d' ' -f3)
        
        for model in "${models[@]}"; do
            echo ""
            echo "模型: $model, 分割: $split"
            echo "训练参数: LR=$LEARNING_RATE, Epochs=$NUM_TRAIN_EPOCHS, BS=$PER_DEVICE_TRAIN_BATCH_SIZE"
            echo "记录到日志: $EXPERIMENT_LOG"
            
            # 训练默认数据比例的模型（用于主实验）
            train_forget_model "$model" "$forget_split"
            train_retain_models "$model" "$retain_split" "$forget_split" "1.0"
            
            # 训练不同数据比例的retain模型（用于数据比例消融实验）
            for ratio_config in "${DATA_RATIO_CONFIGS[@]}"; do
                ratio_label=$(echo $ratio_config | cut -d' ' -f1)
                ratio_mult=$(echo $ratio_config | cut -d' ' -f2)
                
                # 跳过默认比例（已经训练过了）
                if [[ "$ratio_mult" == "1.0" ]]; then
                    continue
                fi
                
                echo "训练数据比例变体: $ratio_label (multiplier=$ratio_mult)"
                train_retain_models "$model" "$retain_split" "$forget_split" "$ratio_mult" "$ratio_label"
            done
        done
    done
}

train_forget_model() {
    local model=$1
    local forget_split=$2
    local task_name="tofu_${model}_${forget_split}_epoch${NUM_TRAIN_EPOCHS}"
    
    echo "[训练] Forget模型: $task_name"
    echo "训练 $task_name" >> $EXPERIMENT_LOG
    
    # 检查模型是否已存在
    if [ -d "$SAVE_BASE_DIR/finetune/$task_name" ]; then
        echo "  -> 模型已存在，跳过训练"
        return 0
    fi
    
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
        --config_file configs/accelerate/default_config.yaml \
        --main_process_port $MASTER_PORT \
        src/train.py experiment=finetune/tofu/default.yaml \
        task_name=$task_name \
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
        trainer.args.save_strategy=$SAVE_STRATEGY
    
    echo "  -> 完成"
}

train_retain_models() {
    local model=$1
    local retain_split=$2
    local forget_split=$3
    local ratio_mult=$4
    local ratio_label=${5:-"default"}
    
    # 计算trim数量
    local trim_count=$(calculate_trim_count "$retain_split" "$ratio_mult")
    
    local suffix=""
    if [[ "$ratio_label" != "default" ]]; then
        suffix="_${ratio_label}"
    fi
    
    local task_name="tofu_${model}_${retain_split}_mimic_epoch${NUM_TRAIN_EPOCHS}${suffix}"
    
    echo "[训练] Retain Mimic模型: $task_name (trim=$trim_count)"
    echo "训练 $task_name (trim=$trim_count)" >> $EXPERIMENT_LOG
    
    # 检查模型是否已存在
    if [ -d "$SAVE_BASE_DIR/finetune/$task_name" ]; then
        echo "  -> 模型已存在，跳过训练"
        return 0
    fi
    
    # 构建split参数（Hydra需要转义方括号）
    local split_arg="train\[:${trim_count}\]"
    
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS accelerate launch \
        --config_file configs/accelerate/default_config.yaml \
        --main_process_port $MASTER_PORT \
        src/train.py experiment=finetune/tofu/default.yaml \
        task_name=$task_name \
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
        trainer.args.save_strategy=$SAVE_STRATEGY
    
    echo "  -> 完成"
}

################################################################################
# 阶段 2: 模型合并实验
################################################################################

merge_models_experiments() {
    echo ""
    echo "=========================================="
    echo "阶段 2: 模型合并实验"
    echo "=========================================="
    
    for split in "${splits[@]}"; do
        forget_split=$(echo $split | cut -d' ' -f1)
        holdout_split=$(echo $split | cut -d' ' -f2)
        retain_split=$(echo $split | cut -d' ' -f3)
        
        for model in "${models[@]}"; do
            echo ""
            echo "模型: $model, 分割: $split"
            
            # 实验 2.1: 默认配置 (α=1.0, β=1.0, 数据比例=1:1)
            echo "[主实验] 默认配置: α=$DEFAULT_ALPHA, β=$DEFAULT_BETA, 比例=$DEFAULT_RATIO"
            merge_single_model "$model" "$forget_split" "$retain_split" \
                "$DEFAULT_ALPHA" "$DEFAULT_BETA" "default" "1.0"
            
            # 实验 2.2: α和β参数扫描（固定数据比例=1:1）
            echo "[消融实验] α和β参数扫描"
            for alpha in "${ALPHA_SWEEP[@]}"; do
                for beta in "${BETA_SWEEP[@]}"; do
                    # 跳过默认配置（已经运行过了）
                    if [[ "$alpha" == "1.0" && "$beta" == "1.0" ]]; then
                        continue
                    fi
                    
                    merge_single_model "$model" "$forget_split" "$retain_split" \
                        "$alpha" "$beta" "sweep" "1.0"
                done
            done
            
            # 实验 2.3: 数据比例扫描（固定α=1.0, β=1.0）
            echo "[消融实验] 数据比例扫描"
            for ratio_config in "${DATA_RATIO_CONFIGS[@]}"; do
                ratio_label=$(echo $ratio_config | cut -d' ' -f1)
                ratio_mult=$(echo $ratio_config | cut -d' ' -f2)
                
                # 跳过默认比例（已经运行过了）
                if [[ "$ratio_mult" == "1.0" ]]; then
                    continue
                fi
                
                merge_single_model "$model" "$forget_split" "$retain_split" \
                    "$DEFAULT_ALPHA" "$DEFAULT_BETA" "ratio" "$ratio_mult" "$ratio_label"
            done
        done
    done
}

merge_single_model() {
    local model=$1
    local forget_split=$2
    local retain_split=$3
    local alpha=$4
    local beta=$5
    local exp_type=$6  # "default", "sweep", or "ratio"
    local ratio_mult=$7
    local ratio_label=${8:-""}
    
    # 构建模型路径和保存路径
    local ft_model_path="$SAVE_BASE_DIR/finetune/tofu_${model}_full"
    local forget_model_path="$SAVE_BASE_DIR/finetune/tofu_${model}_${forget_split}_epoch${NUM_TRAIN_EPOCHS}"
    
    local retain_suffix=""
    if [[ "$exp_type" == "ratio" && -n "$ratio_label" ]]; then
        retain_suffix="_${ratio_label}"
    fi
    local retain_model_path="$SAVE_BASE_DIR/finetune/tofu_${model}_${retain_split}_mimic_epoch${NUM_TRAIN_EPOCHS}${retain_suffix}"
    
    # 构建保存路径名称
    local save_name="tvc_${model}_${forget_split}"
    if [[ "$exp_type" == "default" ]]; then
        save_name="${save_name}_default"
    elif [[ "$exp_type" == "sweep" ]]; then
        save_name="${save_name}_alpha${alpha}_beta${beta}"
    elif [[ "$exp_type" == "ratio" ]]; then
        save_name="${save_name}_${ratio_label}"
    fi
    
    local save_path="$SAVE_BASE_DIR/unlearn/$save_name"
    
    echo "  [合并] α=$alpha, β=$beta, 比例=${ratio_mult:-default}"
    echo "合并模型: $save_name" >> $EXPERIMENT_LOG
    
    # 检查合并后的模型是否已存在
    if [ -d "$save_path" ]; then
        echo "    -> 合并模型已存在，跳过"
        return 0
    fi
    
    # 调用Python合并脚本
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python scripts/model_merge_phi.py \
        --tokenizer_path $BASE_MODEL_PATH \
        --orig_model_path $BASE_MODEL_PATH \
        --ft_model_path $ft_model_path \
        --forget_path $forget_model_path \
        --retain_mimic_path $retain_model_path \
        --save_path $save_path \
        --alpha $alpha \
        --beta $beta \
        --device 0
    
    echo "    -> 完成"
}

################################################################################
# 阶段 3: 评估所有合并模型
################################################################################

evaluate_all_models() {
    echo ""
    echo "=========================================="
    echo "阶段 3: 评估所有合并模型"
    echo "=========================================="
    
    for split in "${splits[@]}"; do
        forget_split=$(echo $split | cut -d' ' -f1)
        holdout_split=$(echo $split | cut -d' ' -f2)
        retain_split=$(echo $split | cut -d' ' -f3)
        
        for model in "${models[@]}"; do
            echo ""
            echo "评估模型: $model, 分割: $split"
            
            # 评估所有合并后的模型
            for merged_model_dir in $SAVE_BASE_DIR/unlearn/tvc_${model}_${forget_split}_*; do
                if [ ! -d "$merged_model_dir" ]; then
                    continue
                fi
                
                local merged_model_name=$(basename $merged_model_dir)
                echo "  [评估] $merged_model_name"
                
                evaluate_single_model "$model" "$forget_split" "$holdout_split" \
                    "$retain_split" "$merged_model_dir" "$merged_model_name"
            done
        done
    done
}

evaluate_single_model() {
    local model=$1
    local forget_split=$2
    local holdout_split=$3
    local retain_split=$4
    local model_path=$5
    local task_name=$6
    
    local eval_output_dir="${model_path}/evals"
    local retain_logs_path="$SAVE_BASE_DIR/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
    
    echo "评估 $task_name" >> $EXPERIMENT_LOG
    
    # 检查评估结果是否已存在
    if [ -f "$eval_output_dir/TOFU_EVAL.json" ]; then
        echo "    -> 评估结果已存在，跳过"
        return 0
    fi
    
    CUDA_VISIBLE_DEVICES=$EVAL_GPU python src/eval.py \
        experiment=eval/tofu/default.yaml \
        forget_split=$forget_split \
        holdout_split=$holdout_split \
        model=$model \
        task_name=$task_name \
        model.model_args.pretrained_model_name_or_path=$model_path \
        paths.output_dir=$eval_output_dir \
        retain_logs_path=$retain_logs_path
    
    echo "    -> 完成"
}

################################################################################
# 阶段 4: 结果汇总与分析
################################################################################

summarize_results() {
    echo ""
    echo "=========================================="
    echo "阶段 4: 结果汇总"
    echo "=========================================="
    
    local summary_file="$SAVE_BASE_DIR/experiment_summary_$(date +%Y%m%d_%H%M%S).txt"
    echo "TVC Experiment Summary" > $summary_file
    echo "Generated at: $(date)" >> $summary_file
    echo "========================================" >> $summary_file
    echo "" >> $summary_file
    
    # 汇总所有评估结果
    for split in "${splits[@]}"; do
        forget_split=$(echo $split | cut -d' ' -f1)
        
        for model in "${models[@]}"; do
            echo "模型: $model, 分割: $forget_split" >> $summary_file
            echo "----------------------------------------" >> $summary_file
            
            # 查找所有评估结果
            for eval_file in $SAVE_BASE_DIR/unlearn/tvc_${model}_${forget_split}_*/evals/TOFU_EVAL.json; do
                if [ -f "$eval_file" ]; then
                    local config_name=$(echo $eval_file | grep -oP "tvc_${model}_${forget_split}_\K[^/]+")
                    echo "  配置: $config_name" >> $summary_file
                    
                    # 提取关键指标（需要根据实际评估输出格式调整）
                    # 这里是示例，实际需要根据TOFU_EVAL.json的格式来提取
                    echo "    文件: $eval_file" >> $summary_file
                fi
            done
            echo "" >> $summary_file
        done
    done
    
    echo "结果汇总已保存到: $summary_file"
    echo "" >> $EXPERIMENT_LOG
    echo "实验完成时间: $(date)" >> $EXPERIMENT_LOG
}

################################################################################
# 主流程
################################################################################

main() {
    echo "=========================================="
    echo "TVC 整合实验开始"
    echo "时间: $(date)"
    echo "=========================================="
    echo ""
    
    # 检查必要的目录和文件
    if [ ! -d "configs" ]; then
        echo "错误: configs目录不存在，请确保在项目根目录下运行"
        exit 1
    fi
    
    if [ ! -f "scripts/model_merge_phi.py" ]; then
        echo "错误: model_merge_phi.py不存在"
        exit 1
    fi
    
    # 创建必要的目录
    mkdir -p $SAVE_BASE_DIR/finetune
    mkdir -p $SAVE_BASE_DIR/unlearn
    mkdir -p $SAVE_BASE_DIR/eval
    
    # 执行实验流程
    train_base_models
    merge_models_experiments
    evaluate_all_models
    summarize_results
    
    echo ""
    echo "=========================================="
    echo "TVC 整合实验完成！"
    echo "时间: $(date)"
    echo "实验日志: $EXPERIMENT_LOG"
    echo "=========================================="
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-merge)
            SKIP_MERGE=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --only-default)
            ONLY_DEFAULT=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--skip-train] [--skip-merge] [--skip-eval] [--only-default]"
            exit 1
            ;;
    esac
done

# 根据参数调整执行流程
if [[ "$SKIP_TRAIN" == true ]]; then
    echo "跳过训练阶段"
    train_base_models() { echo "训练阶段已跳过"; }
fi

if [[ "$SKIP_MERGE" == true ]]; then
    echo "跳过合并阶段"
    merge_models_experiments() { echo "合并阶段已跳过"; }
fi

if [[ "$SKIP_EVAL" == true ]]; then
    echo "跳过评估阶段"
    evaluate_all_models() { echo "评估阶段已跳过"; }
fi

if [[ "$ONLY_DEFAULT" == true ]]; then
    echo "仅运行默认配置实验"
    ALPHA_SWEEP=(1.0)
    BETA_SWEEP=(1.0)
    DATA_RATIO_CONFIGS=("ratio_1_1 1.0")
fi

# 运行主流程
main
