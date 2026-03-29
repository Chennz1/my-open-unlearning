#!/usr/bin/env python3
"""
Filter消融实验结果分析脚本
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import argparse


FILTER_DESCRIPTIONS = {
    "all_linear": "所有线性层(proj+fc+dense+lm_head)",
    "attention_only": "仅Attention层(q_proj+k_proj+v_proj+dense)",
    "mlp_only": "仅MLP层(fc1+fc2)",
    "qkv_only": "仅QKV投影(q_proj+k_proj+v_proj)",
    "output_proj": "仅输出投影(dense+fc2)",
    "lm_head_only": "仅语言模型头",
    "no_lm_head": "除lm_head外所有线性层",
    "attn_and_mlp": "Attention和MLP组合",
    "lightweight": "轻量级(只q_proj+fc1)",
    "heavy": "重量级(所有proj+fc+dense)"
}


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
            'perplexity',
            # 关键联合分析相关指标
            'extraction_strength',
            'exact_memory',
            'forget_rouge',
            'retain_rouge',
            'forget_rouge_retain_rouge'
        ]
        
        for metric in possible_metrics:
            value = None

            # 顶层字段
            if metric in data:
                value = data[metric]
            # results 子字段
            elif 'results' in data and isinstance(data['results'], dict) and metric in data['results']:
                value = data['results'][metric]

            if value is None:
                continue

            # 如果是字典，尝试将其中的标量数值展开成多个指标
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        metrics[f"{metric}_{k}"] = v
                # 跳过将整个 dict 作为一个值
                continue

            # 只保留标量数值型指标，避免后续排序出现 dict 比较错误
            if isinstance(value, (int, float)):
                metrics[metric] = value
        
        return metrics
    
    except Exception as e:
        print(f"警告: 无法从 {eval_json_path} 提取指标: {e}")
        return {}


def collect_results(results_dir: str) -> List[Dict]:
    """收集所有filter配置的结果"""
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
        
        # 提取filter名称
        filter_name = eval_dir.name.replace('_eval', '')
        
        # 获取描述
        description = FILTER_DESCRIPTIONS.get(filter_name, "未知配置")
        
        # 提取指标
        metrics = extract_metrics(str(eval_file))
        
        # 合并
        result = {
            'filter_name': filter_name,
            'description': description,
            **metrics
        }
        results.append(result)
        print(f"收集到结果: {filter_name}")
    
    return results


def generate_comparison_report(results: List[Dict], output_path: str):
    """生成对比报告"""
    if not results:
        print("没有结果可以生成报告")
        return
    
    df = pd.DataFrame(results)
    
    # 按filter_name排序
    df = df.sort_values('filter_name')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# TVC Filter消融实验对比报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("## 实验目标\n\n")
        f.write("测试对模型不同部分应用任务向量组合的效果，找出最优的filter配置。\n\n")
        f.write("**固定参数**: α=1.0, β=1.0, 数据比例=1:1\n\n")
        
        f.write("## Filter配置说明\n\n")
        f.write("| Filter名称 | 描述 | 作用范围 |\n")
        f.write("|-----------|------|----------|\n")
        for filter_name, desc in FILTER_DESCRIPTIONS.items():
            f.write(f"| `{filter_name}` | {desc} | ... |\n")
        f.write("\n")
        
        f.write("## 完整结果对比\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        # 如果有足够的指标，生成排名
        metric_cols = [col for col in df.columns if col not in ['filter_name', 'description']]
        
        if metric_cols:
            f.write("## 指标排名\n\n")
            
            for metric in metric_cols[:5]:  # 显示前5个指标
                if metric in df.columns and df[metric].notna().any():
                    f.write(f"### {metric}\n\n")
                    
                    # 根据指标类型决定升序还是降序
                    # 通常forget_quality, model_utility等越高越好
                    # loss类指标越低越好
                    ascending = 'loss' in metric.lower() or 'perplexity' in metric.lower()
                    
                    sorted_df = df.sort_values(metric, ascending=ascending)[['filter_name', 'description', metric]]
                    sorted_df['rank'] = range(1, len(sorted_df) + 1)
                    sorted_df = sorted_df[['rank', 'filter_name', 'description', metric]]
                    
                    f.write(sorted_df.to_markdown(index=False))
                    f.write("\n\n")

        # 关键指标联合分析：extraction_strength / exact_memory / forget_rouge / retain_rouge / forget_rouge_retain_rouge
        key_metrics = [
            'extraction_strength',
            'exact_memory',
            'forget_rouge',
            'retain_rouge',
            'forget_rouge_retain_rouge'
        ]

        existing_key_metrics = [m for m in key_metrics if m in df.columns and df[m].notna().any()]

        if existing_key_metrics:
            f.write("## 关键指标联合分析\n\n")
            f.write("本节集中对 `extraction_strength`、`exact_memory`、`forget_rouge`、`retain_rouge` 等关键指标进行联合对比和综合评价，以帮助选择更均衡的filter配置。\n\n")

            # 1) 原始指标对比表
            f.write("### 1) 原始指标对比\n\n")
            cols_for_table = ['filter_name', 'description'] + existing_key_metrics
            key_df = df[cols_for_table].copy()
            f.write(key_df.to_markdown(index=False))
            f.write("\n\n")

            # 2) 简单相关性分析（若样本足够）
            try:
                metrics_only = key_df[existing_key_metrics].dropna(how='any')
                if metrics_only.shape[0] >= 3 and metrics_only.shape[1] >= 2:
                    corr = metrics_only.corr()
                    f.write("### 2) 指标间相关性\n\n")
                    f.write("下表给出了关键指标之间的皮尔逊相关系数，用于观察它们的一致性或权衡关系：\n\n")
                    f.write(corr.to_markdown())
                    f.write("\n\n")
            except Exception as e:
                f.write(f"(相关性分析失败: {e})\n\n")

            # 3) 归一化综合得分（方向性采用经验假设，可按需修改）
            try:
                f.write("### 3) 归一化综合得分\n\n")
                f.write("我们对各指标做0-1归一化后，根据经验假设给出方向性：\n")
                f.write("- `extraction_strength`、`exact_memory` 越低越好（表示记忆/可提取性越弱）；\n")
                f.write("- `forget_rouge` 越低越好（与被遗忘数据越不相似）；\n")
                f.write("- `retain_rouge` 越高越好（对保留数据的能力越强）；\n")
                f.write("- 若存在 `forget_rouge_retain_rouge`，假设其越高越好（整体权衡指标）。\n\n")

                norm_df = pd.DataFrame(index=df.index)

                for m in existing_key_metrics:
                    series = df[m]
                    if not series.notna().any():
                        continue
                    min_v, max_v = series.min(), series.max()
                    if max_v == min_v:
                        # 所有值相同，跳过该指标
                        continue
                    # 0-1 归一化
                    norm = (series - min_v) / (max_v - min_v)

                    # 方向性调整
                    lower_is_better = (
                        'loss' in m.lower()
                        or 'perplexity' in m.lower()
                        or 'extraction_strength' in m.lower()
                        or 'exact_memory' in m.lower()
                        or ('forget_rouge' in m.lower() and 'retain' not in m.lower())
                    )
                    if lower_is_better:
                        norm = 1.0 - norm

                    norm_df[m] = norm

                if not norm_df.empty:
                    combined_score = norm_df.mean(axis=1)
                    score_df = df[['filter_name', 'description']].copy()
                    score_df['combined_score'] = combined_score
                    score_df = score_df.sort_values('combined_score', ascending=False)

                    f.write("下表给出了基于上述经验方向的简单综合得分（越高越好，仅作辅助参考）：\n\n")
                    f.write(score_df.to_markdown(index=False))
                    f.write("\n\n")
                else:
                    f.write("(可用于综合评分的关键指标不足，跳过该部分)\n\n")
            except Exception as e:
                f.write(f"(综合得分计算失败: {e})\n\n")
        
        f.write("## 分析建议\n\n")
        f.write("### 1. 单一组件效果\n")
        f.write("- **AttentionOnly** vs **MLP Only**: 哪个组件对遗忘效果贡献更大？\n")
        f.write("- **QKV Only** vs **Output Proj**: Attention内部哪部分更关键？\n\n")
        
        f.write("### 2. 组合效果\n")
        f.write("- **Attention + MLP** vs 单独使用: 组合是否优于单一组件？\n")
        f.write("- **All Linear** vs **No LM Head**: LM Head是否应该被包含？\n\n")
        
        f.write("### 3. 轻量级 vs 重量级\n")
        f.write("- **Lightweight** vs **Heavy**: 最小化参数修改能否达到好效果？\n")
        f.write("- 效率与性能的权衡分析\n\n")
        
        f.write("### 4. 最佳实践\n")
        f.write("根据实验结果总结：\n")
        f.write("- [ ] 推荐的默认filter配置\n")
        f.write("- [ ] 不同场景下的filter选择策略\n")
        f.write("- [ ] 参数效率分析\n\n")


def main():
    parser = argparse.ArgumentParser(description='分析Filter消融实验结果')
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='实验结果目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='filter_comparison_report.md',
        help='输出报告文件名'
    )
    
    args = parser.parse_args()
    
    print(f"收集Filter消融实验结果...")
    print(f"  结果目录: {args.results_dir}")
    
    # 收集结果
    results = collect_results(args.results_dir)
    
    if not results:
        print("错误: 没有找到任何实验结果")
        sys.exit(1)
    
    print(f"\n找到 {len(results)} 个filter配置的结果")
    
    # 生成报告
    output_path = os.path.join(args.results_dir, args.output)
    print(f"\n生成对比报告: {output_path}")
    generate_comparison_report(results, output_path)
    
    # 保存CSV
    csv_path = output_path.replace('.md', '.csv')
    df_all = pd.DataFrame(results)
    df_all.to_csv(csv_path, index=False)
    print(f"完整数据已保存到: {csv_path}")
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()
