import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from param import param
from copy import deepcopy
import gc


def parse_args():
    parser = argparse.ArgumentParser(description="Model merge script for unlearning")
    
    # Model paths
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/home/cnz/.cache/huggingface/hub/Llama-3___2-1B-Instruct",
        help="Path to tokenizer"
    )
    parser.add_argument(
        "--orig_model_path",
        type=str,
        default="/home/cnz/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B",
        help="Path to original model"
    )
    parser.add_argument(
        "--ft_model_path",
        type=str,
        default="/home/cnz/.cache/huggingface/hub/models--open-unlearning--tofu_Llama-3.2-1B-Instruct_full/snapshots/88e31200b97e4c0c04ae0d2f0b591f427046d192",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--forget_path",
        type=str,
        default="/home/cnz/project/open-unlearning/saves/finetune/tofu_Llama-3.2-1B-Instruct_forget10",
        help="Path to forget model"
    )
    parser.add_argument(
        "--retain_mimic_path",
        type=str,
        default="/home/cnz/project/open-unlearning/saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90_mimic",
        help="Path to retain mimic model"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/cnz/project/open-unlearning/vectors/test_model_10",
        help="Path to save merged model"
    )
    
    # Device settings
    parser.add_argument(
        "--device",
        type=int,
        default=1,
        help="CUDA device number"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading models with the following configuration:")
    print(f"  Tokenizer: {args.tokenizer_path}")
    print(f"  Original model: {args.orig_model_path}")
    print(f"  Fine-tuned model: {args.ft_model_path}")
    print(f"  Forget model: {args.forget_path}")
    print(f"  Retain mimic model: {args.retain_mimic_path}")
    print(f"  Save path: {args.save_path}")
    print(f"  Device: cuda:{args.device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, 
        use_fast=True, 
        padding_side="left", 
        legacy=False, 
        token=True
    )

    
    # Load models
    print("\nLoading original model...")
    orig_model = AutoModelForCausalLM.from_pretrained(
        args.orig_model_path, 
        output_hidden_states=True
    ).eval()
    
    print("Loading fine-tuned model...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.ft_model_path, 
        output_hidden_states=True
    ).eval()
    
    print("Loading forget model...")
    forget_model = AutoModelForCausalLM.from_pretrained(
        args.forget_path, 
        output_hidden_states=True
    ).eval()
    
    print("Loading retain mimic model...")
    retain_mimic_model = AutoModelForCausalLM.from_pretrained(
        args.retain_mimic_path, 
        output_hidden_states=True
    ).eval()
    
    device = torch.device(f"cuda:{args.device}")
    
    # Define filter function
    filter_func = lambda n, p: True if "proj" in n else False
    
    # Create param objects and apply filter
    print("\nProcessing model parameters...")
    orig_param = param(orig_model)
    orig_param.filter(filter_func)
    
    ft_param = param(ft_model)
    ft_param.filter(filter_func)
    
    forget_param = param(forget_model)
    forget_param.filter(filter_func)
    
    retain_mimic_param = param(retain_mimic_model)
    retain_mimic_param.filter(filter_func)
    
    # Calculate task vectors
    print("Calculating task vectors...")
    forget_tv = forget_param - orig_param
    retain_tv = retain_mimic_param - orig_param
    ft_tv = ft_param - orig_param
    
    # Merge models
    print("Merging models...")
    new_model = ft_model - forget_tv + retain_tv
    
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


if __name__ == "__main__":
    main()


# python model_merge.py \
#   --tokenizer_path /path/to/tokenizer \
#   --orig_model_path /path/to/orig_model \
#   --ft_model_path /path/to/ft_model \
#   --forget_path /path/to/forget_model \
#   --retain_mimic_path /path/to/retain_model \
#   --save_path /path/to/save \
#   --device 0