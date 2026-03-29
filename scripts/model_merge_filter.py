import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from param import param
from copy import deepcopy
import gc


def parse_args():
    parser = argparse.ArgumentParser(description="Model merge script with configurable filters")
    
    # Model paths
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--orig_model_path", type=str, required=True)
    parser.add_argument("--ft_model_path", type=str, required=True)
    parser.add_argument("--forget_path", type=str, required=True)
    parser.add_argument("--retain_mimic_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    
    # Filter configuration
    parser.add_argument(
        "--filter_name",
        type=str,
        default="all_linear",
        choices=[
            "all_linear", "attention_only", "mlp_only", "qkv_only",
            "output_proj", "lm_head_only", "no_lm_head", "attn_and_mlp",
            "lightweight", "heavy"
        ],
        help="Which parts of the model to apply task vectors to"
    )
    
    # Coefficients
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=0)
    
    return parser.parse_args()


def get_filter_function(filter_name: str):
    """
    返回对应的filter函数
    
    Phi模型结构：
    - Embedding: embed_tokens
    - Attention: q_proj, k_proj, v_proj, dense
    - MLP: fc1, fc2
    - LayerNorm: input_layernorm, final_layernorm
    - Output: lm_head
    """
    
    filters = {
        # 所有线性层（包括lm_head）
        "all_linear": lambda n, p: any(x in n for x in ["proj", "dense", "fc", "lm_head"]),
        
        # 仅Attention层
        "attention_only": lambda n, p: any(x in n for x in ["q_proj", "k_proj", "v_proj", "dense"]) and "lm_head" not in n,
        
        # 仅MLP层
        "mlp_only": lambda n, p: "fc1" in n or "fc2" in n,
        
        # 仅QKV投影
        "qkv_only": lambda n, p: any(x in n for x in ["q_proj", "k_proj", "v_proj"]),
        
        # 仅输出投影（attention的dense和MLP的fc2）
        "output_proj": lambda n, p: "dense" in n or "fc2" in n,
        
        # 仅lm_head
        "lm_head_only": lambda n, p: "lm_head" in n,
        
        # 除lm_head外的所有线性层
        "no_lm_head": lambda n, p: any(x in n for x in ["proj", "dense", "fc"]) and "lm_head" not in n,
        
        # Attention + MLP（不包括lm_head）
        "attn_and_mlp": lambda n, p: any(x in n for x in ["proj", "dense", "fc"]) and "lm_head" not in n,
        
        # 轻量级：只操作关键投影（q_proj和fc1）
        "lightweight": lambda n, p: "q_proj" in n or "fc1" in n,
        
        # 重量级：所有投影层+MLP+dense（不包括lm_head）
        "heavy": lambda n, p: any(x in n for x in ["proj", "dense", "fc1", "fc2"]) and "lm_head" not in n,
    }
    
    if filter_name not in filters:
        raise ValueError(f"Unknown filter name: {filter_name}")
    
    return filters[filter_name]


def count_filtered_params(model, filter_func):
    """统计被filter选中的参数数量"""
    total_params = 0
    filtered_params = 0
    
    for name, p in model.named_parameters():
        total_params += p.numel()
        if filter_func(name, p):
            filtered_params += p.numel()
    
    return filtered_params, total_params


def main():
    args = parse_args()
    
    print(f"=" * 80)
    print(f"TVC Model Merge with Filter: {args.filter_name}")
    print(f"=" * 80)
    print(f"Configuration:")
    print(f"  Filter: {args.filter_name}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Beta: {args.beta}")
    print(f"  Device: cuda:{args.device}")
    print(f"=" * 80)
    
    # Load models
    print("\nLoading models...")
    orig_model = AutoModelForCausalLM.from_pretrained(
        args.orig_model_path, 
        output_hidden_states=True
    ).eval()
    
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.ft_model_path, 
        output_hidden_states=True
    ).eval()
    
    forget_model = AutoModelForCausalLM.from_pretrained(
        args.forget_path, 
        output_hidden_states=True
    ).eval()
    
    retain_mimic_model = AutoModelForCausalLM.from_pretrained(
        args.retain_mimic_path, 
        output_hidden_states=True
    ).eval()
    
    # Get filter function
    filter_func = get_filter_function(args.filter_name)
    
    # Create param objects and apply filter
    print("\nApplying filter...")
    orig_param = param(orig_model)
    orig_param.filter(filter_func)
    
    ft_param = param(ft_model)
    ft_param.filter(filter_func)
    
    forget_param = param(forget_model)
    forget_param.filter(filter_func)
    
    retain_mimic_param = param(retain_mimic_model)
    retain_mimic_param.filter(filter_func)
    
    # Count filtered parameters
    filtered_count, total_count = count_filtered_params(orig_model, filter_func)
    filter_ratio = (filtered_count / total_count) * 100
    
    print(f"\nFilter Statistics:")
    print(f"  Total parameters: {total_count:,}")
    print(f"  Filtered parameters: {filtered_count:,}")
    print(f"  Filter ratio: {filter_ratio:.2f}%")
    
    # Calculate task vectors
    print("\nCalculating task vectors...")
    forget_tv = forget_param - orig_param
    retain_tv = retain_mimic_param - orig_param
    ft_tv = ft_param - orig_param
    
    # Merge models
    print(f"\nMerging models...")
    print(f"  Formula: θ_new = θ_ft - α * τ_forget + β * τ_retain")
    print(f"  Applied to: {filter_ratio:.2f}% of parameters (via filter)")
    new_model = ft_model - (forget_tv * args.alpha) + (retain_tv * args.beta)
    
    # Apply to actual model
    fft_model = deepcopy(ft_model)
    new_model.assign(fft_model)
    
    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save merged model
    print(f"\nSaving merged model to {args.save_path}...")
    fft_model.save_pretrained(args.save_path)
    print("Model saved successfully!")
    print(f"=" * 80)


if __name__ == "__main__":
    main()
