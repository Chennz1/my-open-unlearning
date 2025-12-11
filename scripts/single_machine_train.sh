#!/bin/bash

# 单机训练脚本，不使用accelerate和deepspeed
# 设置CUDA设备和环境变量
export CUDA_VISIBLE_DEVICES=0

# 模型配置
models=(
    "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
)

# 训练器和实验配置
trainers_experiments=(
    # "GradAscent unlearn/tofu/default.yaml"
    # "GradDiff unlearn/tofu/default.yaml"
    # "NPO unlearn/tofu/default.yaml"
    # "DPO unlearn/tofu/idk.yaml"
    # "RMU  unlearn/tofu/default.yaml"
    "SteerUnlearn unlearn/tofu/steer_unlearn.yaml"
)

# 数据集分割配置
splits=(
    # "forget01 holdout01 retain99"
    # "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

# 训练超参数
per_device_train_batch_size=1  # 单GPU批次大小
gradient_accumulation_steps=1  # 梯度累积步数，有效批次大小为 4*8=32


########################################################################################################################
########################################### 单机训练 TOFU 模型 ########################################################
########################################################################################################################

for split in "${splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            trainer=$(echo $trainer_experiment | cut -d' ' -f1)
            experiment=$(echo $trainer_experiment | cut -d' ' -f2)
            
            task_name=tofu_${model}_${forget_split}_${trainer} 
            model_path=open-unlearning/tofu_${model}_full
            echo ${task_name}: Unlearning ${model_path} using ${trainer}
            
            echo "=========================================="
            echo "任务: ${task_name}"
            echo "模型: ${model_path}"
            echo "训练器: ${trainer}"
            echo "实验配置: ${experiment}"
            # echo "输出目录: ${output_dir}"
            echo "=========================================="



            # 开始单机训练
            echo "开始训练..."
            python src/train.py \
                --config-name=unlearn.yaml \
                experiment=${experiment} \
                trainer=${trainer} \
                task_name=${task_name} \
                model=${model} \
                forget_split=${forget_split} \
                retain_split=${retain_split} \
                model.model_args.pretrained_model_name_or_path=${model_path} \
                retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
                trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
                trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
                trainer.args.gradient_checkpointing=true

            # eval
            ## 检查训练是否成功完成
            # if [ $? -eq 0 ]; then
            #     echo "训练成功完成！"
            #     echo "模型保存在: ${output_dir}"
            #     echo "日志保存在: logs/${task_name}_train.log"
                
            #     # 可选：运行评估
            #     echo "开始评估..."
            #     python src/eval.py \
            #         experiment=eval/tofu/steer_unlearn.yaml \
            #         forget_split=${forget_split} \
            #         holdout_split=${holdout_split} \
            #         model=${model} \
            #         task_name=${task_name} \
            #         model.model_args.pretrained_model_name_or_path=${output_dir} \
            #         paths.output_dir=${output_dir}/evals \
            #         retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
            #         2>&1 | tee logs/${task_name}_eval.log
                
            #     if [ $? -eq 0 ]; then
            #         echo "评估成功完成！"
            #         echo "评估结果保存在: ${output_dir}/evals"
            #     else
            #         echo "评估失败，请检查日志文件"
            #     fi
            # else
            #     echo "训练失败，请检查日志文件: logs/${task_name}_train.log"
            #     exit 1
            # fi
        done
    done
done

echo "所有训练任务完成！"
