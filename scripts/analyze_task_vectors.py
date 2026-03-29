"""
任务向量一致性分析脚本

用于分析多次独立微调实验产生的任务向量的一致性
验证假设：从相同基础模型和数据微调产生的任务向量方向一致
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
import argparse
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class TaskVectorAnalyzer:
    """任务向量一致性分析器 - 逐层分析版本"""
    
    def __init__(
        self,
        base_model_path: str,
        finetuned_models_dir: str,
        output_dir: str = "./analysis_results",
        top_k_layers: int = 5
    ):
        """
        初始化分析器
        
        Args:
            base_model_path: 基础模型路径
            finetuned_models_dir: 微调模型目录（包含多个run）
            output_dir: 输出目录
            top_k_layers: 只分析范数最大的K层（节省内存）
        """
        self.base_model_path = Path(base_model_path)
        self.finetuned_models_dir = Path(finetuned_models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_k_layers = top_k_layers
        self.selected_layers = None
        self.base_layer_norms = {}  # 存储base model每层的范数
        
        print(f"基础模型: {self.base_model_path}")
        print(f"微调模型目录: {self.finetuned_models_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"只分析范数最大的 {self.top_k_layers} 层参数（逐层分析）")
    
    def load_model_state_dict(self, model_path: Path) -> Dict[str, torch.Tensor]:
        """加载模型状态字典"""
        print(f"加载模型: {model_path}")
        
        # 尝试不同的文件名
        possible_files = [
            model_path / "pytorch_model.bin",
            model_path / "model.safetensors",
            model_path / "adapter_model.bin",
        ]
        
        # 也检查checkpoint子目录
        for checkpoint_dir in model_path.glob("checkpoint-*"):
            possible_files.extend([
                checkpoint_dir / "pytorch_model.bin",
                checkpoint_dir / "model.safetensors",
            ])
        
        for file_path in possible_files:
            if file_path.exists():
                print(f"  找到模型文件: {file_path}")
                if file_path.suffix == ".bin":
                    return torch.load(file_path, map_location="cpu")
                elif file_path.suffix == ".safetensors":
                    from safetensors.torch import load_file
                    return load_file(str(file_path))
        
        raise FileNotFoundError(f"未找到模型文件在: {model_path}")
    
    def compute_task_vector(
        self, 
        base_state: Dict[str, torch.Tensor],
        finetuned_state: Dict[str, torch.Tensor],
        selected_keys: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算任务向量 τ = θ_ft - θ_base
        
        Args:
            base_state: 基础模型状态
            finetuned_state: 微调模型状态
            selected_keys: 只计算指定的keys（内存优化）
            
        Returns:
            任务向量
        """
        task_vector = {}
        keys_to_process = selected_keys if selected_keys else base_state.keys()
        
        for key in keys_to_process:
            if key in base_state and key in finetuned_state:
                # 转换为float32以避免BFloat16问题
                diff = finetuned_state[key].float() - base_state[key].float()
                task_vector[key] = diff
        return task_vector
    
    def select_top_k_layers(
        self, 
        base_state: Dict[str, torch.Tensor],
        finetuned_model_path: Path,
        k: int = 5
    ) -> List[str]:
        """
        选择任务向量范数最大的K层（仅选择MLP层的proj）
        
        Args:
            base_state: 基础模型状态
            finetuned_model_path: 一个微调模型路径（用于确定重要层）
            k: 选择top k层
            
        Returns:
            选中的参数key列表
        """
        print(f"\n选择范数最大的 {k} 个 MLP proj 层...")
        
        # 加载一个微调模型来计算层重要性
        finetuned_state = self.load_model_state_dict(finetuned_model_path)
        
        # 只计算MLP层的proj参数的范数（任务向量范数）
        layer_norms = {}
        for key in base_state.keys():
            # 只选择 mlp 且包含 proj 的层
            if 'mlp' in key.lower() and 'proj' in key.lower() and key in finetuned_state:
                diff = finetuned_state[key].float() - base_state[key].float()
                norm = torch.norm(diff).item()
                layer_norms[key] = norm
                
                # 同时记录base model该层的范数
                base_norm = torch.norm(base_state[key].float()).item()
                self.base_layer_norms[key] = base_norm
        
        # 释放内存
        del finetuned_state
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if not layer_norms:
            raise ValueError("未找到 MLP proj 层！请检查模型结构")
        
        # 按范数排序，选择top k
        sorted_layers = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)
        top_k_keys = [key for key, norm in sorted_layers[:k]]
        
        print(f"  从 {len(layer_norms)} 个 MLP proj 层中选出 top-{k}:")
        
        total_params = sum(base_state[k].numel() for k in top_k_keys)
        print(f"  总参数量: {total_params:,}")
        
        return top_k_keys
    
    def flatten_task_vector(self, task_vector: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将任务向量展平为一维向量"""
        # 确保所有tensor都是float32类型
        return torch.cat([v.flatten().float() for v in task_vector.values()])
    
    def compute_cosine_similarity(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """计算余弦相似度"""
        # 手动计算余弦相似度以确保正确性
        v1 = v1.flatten().float().cpu()
        v2 = v2.flatten().float().cpu()
        
        # 归一化向量
        v1_normalized = v1 / torch.norm(v1)
        v2_normalized = v2 / torch.norm(v2)
        
        # 计算点积（已归一化的向量点积就是余弦相似度）
        cosine_sim = torch.dot(v1_normalized, v2_normalized)
        
        # Clamp to [-1, 1] to handle numerical precision issues
        cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
        
        return cosine_sim.item()
    
    def _analyze_single_layer(
        self, 
        layer_key: str, 
        layer_vectors: List[torch.Tensor],
        run_info: List[Dict]
    ) -> Dict:
        """分析单层的一致性指标"""
        num_vectors = len(layer_vectors)
        
        # 1. 计算每个向量的范数
        norms = [torch.norm(v.float()).item() for v in layer_vectors]
        avg_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        # 2. Base model该层的范数
        base_norm = self.base_layer_norms[layer_key]
        
        # 3. 计算余弦相似度
        similarities = []
        similarity_matrix = np.eye(num_vectors)
        
        for i in range(num_vectors):
            for j in range(i + 1, num_vectors):
                sim = self.compute_cosine_similarity(layer_vectors[i], layer_vectors[j])
                similarities.append(sim)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        avg_similarity = np.mean(similarities) if similarities else 1.0
        std_similarity = np.std(similarities) if similarities else 0.0
        
        # 4. 计算差值范数
        difference_norms = []
        difference_norm_matrix = np.zeros((num_vectors, num_vectors))
        
        for i in range(num_vectors):
            vec_i = layer_vectors[i].double().cpu()
            for j in range(i + 1, num_vectors):
                vec_j = layer_vectors[j].double().cpu()
                diff = vec_i - vec_j
                diff_norm = torch.norm(diff).item()
                difference_norms.append(diff_norm)
                difference_norm_matrix[i, j] = diff_norm
                difference_norm_matrix[j, i] = diff_norm
        
        avg_diff_norm = np.mean(difference_norms) if difference_norms else 0.0
        std_diff_norm = np.std(difference_norms) if difference_norms else 0.0
        
        # 5. 计算相对差异
        relative_diff = avg_diff_norm / avg_norm if avg_norm > 0 else 0.0
        relative_to_base = avg_norm / base_norm if base_norm > 0 else 0.0
        
        # 6. 计算与平均向量的差异
        mean_vector = torch.stack([v.double().cpu() for v in layer_vectors]).mean(dim=0)
        norms_to_mean = []
        for v in layer_vectors:
            diff = v.double().cpu() - mean_vector
            norms_to_mean.append(torch.norm(diff).item())
        
        avg_norm_to_mean = np.mean(norms_to_mean)
        relative_to_mean = avg_norm_to_mean / avg_norm if avg_norm > 0 else 0.0
        
        # 打印该层结果
        print(f"    Base Model Norm: {base_norm:.4e}")
        print(f"    Avg Task Vector Norm: {avg_norm:.4e} (Ratio to Base: {relative_to_base:.4f})")
        print(f"    Cosine Similarity: {avg_similarity:.4f} ± {std_similarity:.4f}")
        print(f"    Avg Difference Norm: {avg_diff_norm:.4e} (Relative: {relative_diff:.2%})")
        
        return {
            "layer_key": layer_key,
            "base_model_norm": base_norm,
            "task_vector_norms": {
                "values": norms,
                "mean": avg_norm,
                "std": std_norm,
                "ratio_to_base": relative_to_base
            },
            "cosine_similarity": {
                "values": similarities,
                "matrix": similarity_matrix.tolist(),
                "mean": avg_similarity,
                "std": std_similarity,
                "min": np.min(similarities) if similarities else 1.0,
                "max": np.max(similarities) if similarities else 1.0
            },
            "difference_norms": {
                "values": difference_norms,
                "matrix": difference_norm_matrix.tolist(),
                "mean": avg_diff_norm,
                "std": std_diff_norm,
                "min": np.min(difference_norms) if difference_norms else 0.0,
                "max": np.max(difference_norms) if difference_norms else 0.0,
                "relative_to_vector_norm": relative_diff
            },
            "difference_to_mean": {
                "values": norms_to_mean,
                "mean": avg_norm_to_mean,
                "relative": relative_to_mean
            }
        }
    
    def _compute_summary_statistics(self, layer_results: Dict) -> Dict:
        """计算跨层汇总统计"""
        summary = {
            "avg_cosine_similarity": np.mean([r["cosine_similarity"]["mean"] for r in layer_results.values()]),
            "avg_difference_norm_ratio": np.mean([r["difference_norms"]["relative_to_vector_norm"] for r in layer_results.values()]),
            "avg_ratio_to_base": np.mean([r["task_vector_norms"]["ratio_to_base"] for r in layer_results.values()]),
            "layers_with_high_consistency": sum(1 for r in layer_results.values() if r["difference_norms"]["relative_to_vector_norm"] < 0.05),
            "layers_with_medium_consistency": sum(1 for r in layer_results.values() if 0.05 <= r["difference_norms"]["relative_to_vector_norm"] < 0.1),
            "layers_with_low_consistency": sum(1 for r in layer_results.values() if r["difference_norms"]["relative_to_vector_norm"] >= 0.1)
        }
        
        print(f"\n跨层汇总统计:")
        print(f"  平均余弦相似度: {summary['avg_cosine_similarity']:.4f}")
        print(f"  平均相对差异: {summary['avg_difference_norm_ratio']:.2%}")
        print(f"  平均任务向量/基础模型范数比: {summary['avg_ratio_to_base']:.4f}")
        print(f"  高一致性层 (<5%): {summary['layers_with_high_consistency']}/{len(layer_results)}")
        print(f"  中一致性层 (5-10%): {summary['layers_with_medium_consistency']}/{len(layer_results)}")
        print(f"  低一致性层 (≥10%): {summary['layers_with_low_consistency']}/{len(layer_results)}")
        
        return summary
    
    def _prepare_for_json(self, obj):
        """准备对象用于JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def analyze_consistency(self) -> Dict:
        """执行完整的一致性分析 - 逐层分析版本"""
        print("\n" + "=" * 80)
        print("开始任务向量一致性分析（逐层分析）")
        print("=" * 80)
        
        # 1. 加载基础模型
        print("\n[1/6] 加载基础模型...")
        base_state = self.load_model_state_dict(self.base_model_path)
        print(f"  参数数量: {len(base_state)}")
        
        # 2. 查找所有微调模型
        print("\n[2/6] 查找微调模型...")
        run_dirs = sorted(self.finetuned_models_dir.glob("run*"))
        if not run_dirs:
            raise ValueError(f"未找到运行目录在: {self.finetuned_models_dir}")
        print(f"  找到 {len(run_dirs)} 个运行")
        
        # 3. 选择范数最大的K层（使用第一个模型）
        print(f"\n[3/6] 选择范数最大的 {self.top_k_layers} 层...")
        if self.selected_layers is None:
            self.selected_layers = self.select_top_k_layers(
                base_state, 
                run_dirs[0], 
                k=self.top_k_layers
            )
        
        # 4. 计算每个运行的任务向量（保存为字典，不flatten）
        print("\n[4/6] 计算任务向量并保存到磁盘...")
        vectors_cache_dir = self.output_dir / "vectors_cache"
        vectors_cache_dir.mkdir(exist_ok=True)
        
        run_info = []
        
        for i, run_dir in enumerate(tqdm(run_dirs, desc="处理模型")):
            try:
                # 加载微调模型
                finetuned_state = self.load_model_state_dict(run_dir)
                
                # 只计算选中层的任务向量（保持字典格式）
                task_vector = self.compute_task_vector(
                    base_state, 
                    finetuned_state, 
                    selected_keys=self.selected_layers
                )
                
                # 计算每层的范数
                layer_norms = {key: torch.norm(task_vector[key].float()).item() 
                             for key in task_vector.keys()}
                
                # 保存到磁盘（保存字典格式）
                cache_path = vectors_cache_dir / f"vector_{i}.pt"
                torch.save({k: v.cpu() for k, v in task_vector.items()}, cache_path)
                
                run_info.append({
                    "run_id": i,
                    "path": str(run_dir),
                    "layer_norms": layer_norms,
                    "cache_path": str(cache_path)
                })
                
                print(f"  Run {i}: 已计算 {len(self.selected_layers)} 层的任务向量")
                
                # 释放内存
                del finetuned_state, task_vector
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"  警告: Run {i} 处理失败: {e}")
                continue
        
        num_vectors = len(run_info)
        print(f"\n成功加载 {num_vectors} 个任务向量")
        
        # 5. 逐层计算一致性指标
        print("\n[5/6] 逐层计算一致性指标...")
        layer_results = {}
        
        for layer_idx, layer_key in enumerate(tqdm(self.selected_layers, desc="逐层分析")):
            print(f"\n  分析层 {layer_idx+1}/{len(self.selected_layers)}: {layer_key}")
            
            # 提取该层的所有任务向量
            layer_vectors = []
            for info in run_info:
                task_vector_dict = torch.load(info["cache_path"])
                layer_vectors.append(task_vector_dict[layer_key].float().cpu())
                del task_vector_dict
            
            # 计算该层的指标
            layer_metrics = self._analyze_single_layer(
                layer_key, 
                layer_vectors, 
                run_info
            )
            layer_results[layer_key] = layer_metrics
            
            # 释放内存
            del layer_vectors
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 6. 整理结果
        print("\n[6/6] 整理分析结果...")
        results = {
            "num_runs": num_vectors,
            "selected_layers": self.selected_layers,
            "num_selected_layers": len(self.selected_layers),
            "base_layer_norms": self.base_layer_norms,
            "run_info": run_info,
            "layer_results": layer_results
        }
        
        # 计算跨层汇总统计
        results["summary"] = self._compute_summary_statistics(layer_results)
        
        # 保存结果
        result_file = self.output_dir / "analysis_results.json"
        with open(result_file, "w") as f:
            # 将Tensor转为float以便JSON序列化
            json_results = self._prepare_for_json(results)
            json.dump(json_results, f, indent=2)
        print(f"\n结果已保存至: {result_file}")
        
        return results
    
    def visualize_results(self, results: Dict):
        """生成可视化图表 - 逐层分析版本"""
        print("\n生成可视化图表...")
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        
        layer_results = results["layer_results"]
        layer_keys = results["selected_layers"]
        num_layers = len(layer_keys)
        
        # 1. Per-Layer Cosine Similarity Heatmaps
        print("  Generating per-layer cosine similarity heatmaps...")
        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))
        if num_layers == 1:
            axes = [axes]
        
        for idx, (layer_key, ax) in enumerate(zip(layer_keys, axes)):
            sim_matrix = np.array(layer_results[layer_key]["cosine_similarity"]["matrix"])
            sns.heatmap(sim_matrix, annot=True, fmt=".3f", cmap="RdYlGn",
                       vmin=0.95, vmax=1.0, square=True, ax=ax,
                       xticklabels=[f"R{i}" for i in range(len(sim_matrix))],
                       yticklabels=[f"R{i}" for i in range(len(sim_matrix))])
            ax.set_title(f'Layer {idx+1}\n{layer_key.split(".")[-1]}', fontsize=10)
        
        plt.suptitle('Per-Layer Cosine Similarity Matrices', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(plots_dir / "per_layer_similarity.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Per-layer similarity heatmaps")
        
        # 2. Per-Layer Difference Norms Heatmaps
        print("  Generating per-layer difference norms heatmaps...")
        fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4))
        if num_layers == 1:
            axes = [axes]
        
        for idx, (layer_key, ax) in enumerate(zip(layer_keys, axes)):
            diff_matrix = np.array(layer_results[layer_key]["difference_norms"]["matrix"])
            sns.heatmap(diff_matrix, annot=True, fmt=".2e", cmap="YlOrRd",
                       square=True, ax=ax,
                       xticklabels=[f"R{i}" for i in range(len(diff_matrix))],
                       yticklabels=[f"R{i}" for i in range(len(diff_matrix))])
            ax.set_title(f'Layer {idx+1}\n{layer_key.split(".")[-1]}', fontsize=10)
        
        plt.suptitle('Per-Layer Difference Norm Matrices ||τ_i - τ_j||', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(plots_dir / "per_layer_difference_norms.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Per-layer difference norms heatmaps")
        
        # 3. Layer Comparison: Base Norm vs Task Vector Norm
        print("  Generating norm comparison plot...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        layer_names = [f"L{i+1}" for i in range(num_layers)]
        base_norms = [layer_results[k]["base_model_norm"] for k in layer_keys]
        task_norms = [layer_results[k]["task_vector_norms"]["mean"] for k in layer_keys]
        
        x = np.arange(num_layers)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, base_norms, width, label='Base Model Norm', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, task_norms, width, label='Task Vector Norm (Avg)', alpha=0.8, color='coral')
        
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('L2 Norm', fontsize=11)
        ax.set_title('Base Model Norm vs Task Vector Norm (Per Layer)', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "norm_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Norm comparison plot")
        
        # 4. Layer Consistency Metrics Summary
        print("  Generating consistency summary...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        layer_names = [f"L{i+1}" for i in range(num_layers)]
        cosine_sims = [layer_results[k]["cosine_similarity"]["mean"] for k in layer_keys]
        diff_norms = [layer_results[k]["difference_norms"]["mean"] for k in layer_keys]
        relative_diffs = [layer_results[k]["difference_norms"]["relative_to_vector_norm"] for k in layer_keys]
        ratios_to_base = [layer_results[k]["task_vector_norms"]["ratio_to_base"] for k in layer_keys]
        
        # Cosine Similarity
        ax1.bar(layer_names, cosine_sims, alpha=0.7, color='green')
        ax1.set_ylabel('Cosine Similarity', fontsize=11)
        ax1.set_title('Average Cosine Similarity (Per Layer)', fontsize=12)
        ax1.set_ylim([min(cosine_sims) * 0.999, 1.0])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Difference Norms
        ax2.bar(layer_names, diff_norms, alpha=0.7, color='orange')
        ax2.set_ylabel('Difference Norm', fontsize=11)
        ax2.set_title('Average Difference Norm ||τ_i - τ_j|| (Per Layer)', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Relative Difference (consistency metric)
        colors = ['green' if r < 0.05 else 'yellow' if r < 0.1 else 'red' for r in relative_diffs]
        ax3.bar(layer_names, relative_diffs, alpha=0.7, color=colors)
        ax3.axhline(0.05, color='green', linestyle='--', linewidth=1, alpha=0.5, label='<5% (High)')
        ax3.axhline(0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='<10% (Medium)')
        ax3.set_ylabel('Relative Difference', fontsize=11)
        ax3.set_title('Relative Difference (Consistency Metric, Per Layer)', fontsize=12)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Ratio to Base
        ax4.bar(layer_names, ratios_to_base, alpha=0.7, color='purple')
        ax4.set_ylabel('Ratio', fontsize=11)
        ax4.set_title('Task Vector / Base Model Norm Ratio (Per Layer)', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "consistency_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Consistency summary plot")
        
        print(f"\nAll plots saved to: {plots_dir}")
    
    def print_summary(self, results: Dict):
        """打印分析摘要 - 逐层分析版本"""
        print("\n" + "=" * 80)
        print("Per-Layer Task Vector Consistency Analysis Summary")
        print("=" * 80)
        
        print(f"\nNumber of Runs: {results['num_runs']}")
        print(f"Number of Layers Analyzed: {results['num_selected_layers']}")
        
        summary = results["summary"]
        layer_results = results["layer_results"]
        
        print(f"\n【Cross-Layer Summary Statistics】")
        print(f"  Average Cosine Similarity: {summary['avg_cosine_similarity']:.4f}")
        print(f"  Average Relative Difference: {summary['avg_difference_norm_ratio']:.2%}")
        print(f"  Average Task/Base Norm Ratio: {summary['avg_ratio_to_base']:.4f}")
        print(f"\n  Consistency Breakdown:")
        print(f"    High Consistency (<5%):    {summary['layers_with_high_consistency']}/{results['num_selected_layers']}")
        print(f"    Medium Consistency (5-10%): {summary['layers_with_medium_consistency']}/{results['num_selected_layers']}")
        print(f"    Low Consistency (≥10%):     {summary['layers_with_low_consistency']}/{results['num_selected_layers']}")
        
        print(f"\n【Per-Layer Detailed Results】")
        for idx, layer_key in enumerate(results['selected_layers']):
            layer_res = layer_results[layer_key]
            print(f"\n  Layer {idx+1}: {layer_key}")
            print(f"    Base Model Norm:         {layer_res['base_model_norm']:.4e}")
            print(f"    Avg Task Vector Norm:    {layer_res['task_vector_norms']['mean']:.4e}")
            print(f"    Ratio (Task/Base):       {layer_res['task_vector_norms']['ratio_to_base']:.4f}")
            print(f"    Cosine Similarity:       {layer_res['cosine_similarity']['mean']:.4f} ± {layer_res['cosine_similarity']['std']:.4f}")
            print(f"    Difference Norm:         {layer_res['difference_norms']['mean']:.4e}")
            print(f"    Relative Difference:     {layer_res['difference_norms']['relative_to_vector_norm']:.2%}")
            
            # 一致性评估
            rel_diff = layer_res['difference_norms']['relative_to_vector_norm']
            if rel_diff < 0.05:
                status = "✓✓ High Consistency"
            elif rel_diff < 0.1:
                status = "✓ Medium Consistency"
            else:
                status = "✗ Low Consistency"
            print(f"    → {status}")
        
        print(f"\n【Overall Conclusion】")
        avg_rel_diff = summary['avg_difference_norm_ratio']
        
        if avg_rel_diff < 0.01:
            print(f"  ✓✓✓ Strongly Supports Hypothesis 1:")
            print(f"      Task vectors are extremely consistent (<1% difference)")
            print(f"      Parameter changes are strongly determined by the data")
        elif avg_rel_diff < 0.05:
            print(f"  ✓✓ Supports Hypothesis 1:")
            print(f"      Task vectors are highly consistent (<5% difference)")
            print(f"      Common direction of parameter change exists")
        elif avg_rel_diff < 0.1:
            print(f"  ✓ Partially Supports Hypothesis 1:")
            print(f"      Task vectors show moderate consistency (<10% difference)")
            print(f"      Some randomness affects parameter changes")
        else:
            print(f"  ✗ Does Not Support Hypothesis 1:")
            print(f"      Task vectors show significant differences ({avg_rel_diff*100:.1f}%)")
            print(f"      Randomness has substantial impact on parameter changes")
        
        high_ratio = summary['layers_with_high_consistency'] / results['num_selected_layers']
        if high_ratio >= 0.8:
            print(f"  ✓ {high_ratio*100:.0f}% of layers show high consistency")
        elif high_ratio >= 0.5:
            print(f"  ≈ {high_ratio*100:.0f}% of layers show high consistency")
        else:
            print(f"  ✗ Only {high_ratio*100:.0f}% of layers show high consistency")
        
        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Per-Layer Task Vector Consistency Analysis"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base model"
    )
    parser.add_argument(
        "--finetuned-dir",
        type=str,
        required=True,
        help="Directory containing multiple finetuned runs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./task_vector_analysis",
        help="Output directory"
    )
    parser.add_argument(
        "--top-k-layers",
        type=int,
        default=5,
        help="Analyze only top-K layers by norm (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = TaskVectorAnalyzer(
        base_model_path=args.base_model,
        finetuned_models_dir=args.finetuned_dir,
        output_dir=args.output_dir,
        top_k_layers=args.top_k_layers
    )
    
    # Execute analysis
    results = analyzer.analyze_consistency()
    
    # Visualize
    analyzer.visualize_results(results)
    
    # Print summary
    analyzer.print_summary(results)


if __name__ == "__main__":
    main()
