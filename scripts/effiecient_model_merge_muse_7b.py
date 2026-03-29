import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Model merge script for unlearning")
    
    # Model paths
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/cnz/data/ms-home/Llama-2-7b-chat-hf/",
        # default="/cnz/data/ms-home/Llama-2-7b-chat-hf",
        help="Path to tokenizer"
    )
    parser.add_argument(
        "--orig_model_path",
        type=str,
        default="/cnz/data/ms-home/Llama-2-7b-chat-hf/",
        # default="/cnz/data/ms-home/Llama-2-7b-chat-hf",
        help="Path to original model"
    )
    parser.add_argument(
        "--ft_model_path",
        type=str,
        default="/cnz/data/hf-home/hub/models--muse-bench--MUSE-news_target/snapshots/a2f39769e9a0b98ec1cdd12f65e9962502208935",
        # default="/cnz/data/project/my-open-unlearning/saves/finetune/tofu_Llama-2-7b-chat-hf_full",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--forget_path",
        type=str,
        default="/cnz/data/project/my-open-unlearning/saves/finetune/muse_Llama-2-7b-hf_News_forget_e10",
        help="Path to forget model"
    )
    parser.add_argument(
        "--retain_mimic_path",
        type=str,
        default="/cnz/data/project/my-open-unlearning/saves/finetune/muse_Llama-2-7b-hf_News_retain1_e10",
        # default="/cnz/data/project/my-open-unlearning/saves/finetune/tofu_Llama-2-7b-chat-hf_retain90_mimic",
        help="Path to retain mimic model"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/cnz/data/project/my-open-unlearning/saves/unlearn/test_model_muse_7b",
        help="Path to save merged model"
    )
    
    # Device settings
    parser.add_argument(
        "--device",
        type=int,
        default=1,
        help="CUDA device number"
    )
    
    # Task vector coefficients
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Coefficient for forget task vector (default: 1.0)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Coefficient for retain task vector (default: 1.0)"
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
    print(f"  Alpha (forget coefficient): {args.alpha}")
    print(f"  Beta (retain coefficient): {args.beta}")
    
    # Define filter function
    filter_func = lambda n, p: True if "up_proj" in n or "down_proj" in n or "gate_proj" in n or "fc" in n else False
    # filter_func = lambda n, p: True 
    # filter_func = lambda n, p: "mlp" in n
    
    print("\nLoading fine-tuned model (base for merging)...")
    # Load base model
    # We load to CPU to save memory. If you have enough VRAM, you can move to GPU.
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.ft_model_path, 
        torch_dtype=torch.float16,
        output_hidden_states=True
    ).eval()
    
    # Helper function to apply updates
    def apply_update(model_path, scale, name):
        if abs(scale) < 1e-9:
            print(f"Skipping {name} as scale is 0")
            return

        print(f"\nLoading {name} from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            output_hidden_states=True
        ).eval()
        
        print(f"Applying update from {name} with scale {scale}...")
        with torch.no_grad():
            for n, p in model.named_parameters():
                if filter_func(n, p):
                    if n in ft_model.state_dict():
                        # In-place update: param = param + scale * other_param
                        # Ensure we are on the same device (likely CPU)
                        ft_model.state_dict()[n].add_(p.to(ft_model.device), alpha=scale)
                    else:
                        print(f"Warning: Parameter {n} not found in base model")
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Finished processing {name}")

    # Formula: new = ft - alpha * (forget - orig) + beta * (retain - orig)
    #              = ft - alpha * forget + alpha * orig + beta * retain - beta * orig
    #              = ft - alpha * forget + beta * retain + (alpha - beta) * orig
    
    # 1. Apply forget model (-alpha)
    apply_update(args.forget_path, -args.alpha, "forget model")
    
    # 2. Apply retain model (+beta)
    apply_update(args.retain_mimic_path, args.beta, "retain mimic model")
    
    # 3. Apply original model (alpha - beta)
    apply_update(args.orig_model_path, args.alpha - args.beta, "original model")
    
    # Save merged model
    print(f"\nSaving merged model to {args.save_path}...")
    ft_model.save_pretrained(args.save_path)
    
    # Try to save tokenizer as well
    try:
        print("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        tokenizer.save_pretrained(args.save_path)
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")
        
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