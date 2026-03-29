#!/usr/bin/env python3
"""
TVC实验结果分析脚本 - 适配存储优化版
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import argparse


def extract_metrics(eval_json_path: str) -> Dict[str, float]:
    """从评估JSON文件中提取关键指标"""
    try:
        with open(eval_json_path, 'r') as f:
            data = json.load(f)
        
        metrics = {}
        
        # 常见指标名称
        possible_metrics = [
            'forget_quality',
            'model_utility', 
            'truthfulness',
            'factuality',
            'rouge_score',
            'truth_ratio',
            'forget_loss',
            'retain_loss',
            'accuracy',
            'perplexity'
        ]
        
        for metric in possible_metrics:
            if metric in data:
                metrics[metric] = data[metric]
            elif 'results' in data and metric in data['results']:
                metrics[metric] = data['results'][metric]
        
        return metrics
    
    except Exception as e:
        print(f"警告: 无法从 {eval_json_path} 提取指标: {e}")
        return {}


def parse_exp_name(exp_name: str) -> Dict[str, str]:
    """解析实验名称提取配置信息"""
    config = {'exp_name': exp_name}
    
    parts = exp_name.split('_')
    
    # 提取模型名
    if 'phi' in exp_name:
        config['model'] = 'phi-1_5'
    
    # 提取forget split
    for part in parts:
        if 'forget' in part:
            config['forget_split'] = part
            break
    
    # 提取alpha和beta
    for i, part in enumerate(parts):
        if part.startswith('alpha'):
            config['alpha'] = part.replace('alpha', '')
        elif part.startswith('beta'):
            config['beta'] = part.replace('beta', '')
        elif part == 'ratio' and i + 2 < len(parts):
            config['data_ratio'] = f"{parts[i+1]}:{parts[i+2]}"
    
    # 设置默认值
    config.setdefault('alpha', '1.0')
    config.setdefault('beta', '1.0')
    config.setdefault('data_ratio', '1:1')
    
    # 检测实验类型
    if 'default' in exp_name:
        config['exp_type'] = 'default'
    elif 'ratio' in exp_name:
        config['exp_type'] = 'ratio_ablation'
    else:
        config['exp_type'] = 'coeff_ablation'
    
    return config


def collect_results(results_dir: str) -> List[Dict]:
    """收集所有实验结果"""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"错误: 结果目录不存在: {results_dir}")
        return []
    
    # 查找所有评估目录
    for eval_dir in results_path.glob("*_eval"):
        eval_file = eval_dir / 'TOFU_EVAL.json'
        
        if not eval_file.exists():
            print(f"跳过 {eval_dir.name}: 评估文件不存在")
            continue
        
        # 提取实验名称（去掉_eval后缀）
        exp_name = eval_dir.name.replace('_eval', '')
        
        # 解析配置
        config = parse_exp_name(exp_name)
        
        # 提取指标
        metrics = extract_metrics(str(eval_file))
        
        # 合并
        result = {**config, **metrics}
        results.append(result)
        print(f"收集到结果: {exp_name}")
    
    return results


def generate_report(results: List[Dict], output_path: str):
    """生成实验报告"""
    if not results:
        print("没有结果可以生成报告")
        return
    
    df = pd.DataFrame(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# TVC 实验结果分析报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. 默认配置结果
        default_df = df[df['exp_type'] == 'default']
        if not default_df.empty:
            f.write("## 1. 主实验: 默认配置 (α=1.0, β=1.0, 数据比例=1:1)\n\n")
            f.write(default_df.to_markdown(index=False))
            f.write("\n\n")
        
        # 2. 系数扫描结果
        coeff_df = df[df['exp_type'] == 'coeff_ablation']
        if not coeff_df.empty:
            f.write("## 2. 消融实验: α和β系数扫描\n\n")
            coeff_df['alpha'] = coeff_df['alpha'].astype(float)
            coeff_df['beta'] = coeff_df['beta'].astype(float)
            coeff_df = coeff_df.sort_values(['alpha', 'beta'])
            f.write(coeff_df.to_markdown(index=False))
            f.write("\n\n")
        
        # 3. 数据比例扫描结果
        ratio_df = df[df['exp_type'] == 'ratio_ablation']
        if not ratio_df.empty:
            f.write("## 3. 消融实验: 数据比例扫描\n\n")
            f.write(ratio_df.to_markdown(index=False))
            f.write("\n\n")
        
        f.write("## 4. 总结\n\n")
        f.write(f"- 总实验数: {len(results)}\n")
        f.write(f"- 默认配置实验: {len(default_df)}\n")
        f.write(f"- 系数扫描实验: {len(coeff_df)}\n")
        f.write(f"- 数据比例实验: {len(ratio_df)}\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description='分析TVC实验结果')
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='实验结果目录'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='phi-1_5',
        help='模型名称（用于报告标题）'
    )
    parser.add_argument(
        '--forget_split',
        type=str,
        default='forget05',
        help='Forget数据分割（用于报告标题）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='tvc_analysis_report.md',
        help='输出报告文件名'
    )
    
    args = parser.parse_args()
    
    print(f"收集实验结果...")
    print(f"  结果目录: {args.base_dir}")
    
    # 收集结果
    results = collect_results(args.base_dir)
    
    if not results:
        print("错误: 没有找到任何实验结果")
        sys.exit(1)
    
    print(f"\n找到 {len(results)} 个实验结果")
    
    # 生成报告
    output_path = os.path.join(args.base_dir, args.output)
    print(f"\n生成分析报告: {output_path}")
    generate_report(results, output_path)
    
    # 保存CSV
    csv_path = output_path.replace('.md', '.csv')
    df_all = pd.DataFrame(results)
    df_all.to_csv(csv_path, index=False)
    print(f"完整数据已保存到: {csv_path}")
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()
