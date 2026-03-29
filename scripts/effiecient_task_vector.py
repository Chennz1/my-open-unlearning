import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


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
        "--save_path",
        type=str,
        default="/cnz/data/project/my-open-unlearning/saves/unlearn/test_model_muse_7b",
        help="Path to save merged model"
    )

    # Coefficients
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Coefficient for new = ft - alpha*(forget - ft) (default: 1.0)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading models with the following configuration:")
    print(f"  Tokenizer: {args.tokenizer_path}")
    print(f"  Fine-tuned model: {args.ft_model_path}")
    print(f"  Forget model: {args.forget_path}")
    print(f"  Save path: {args.save_path}")
    print(f"  Alpha: {args.alpha}")
    
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

    # Merge formula (only this one):
    #   new = ft - alpha*(forget - ft)
    #       = (1 + alpha)*ft - alpha*forget
    if abs(args.alpha) < 1e-9:
        print("Alpha is 0, skipping merge (new == ft).")
    else:
        print("\nLoading forget model...")
        forget_model = AutoModelForCausalLM.from_pretrained(
            args.forget_path,
            torch_dtype=torch.float16,
            output_hidden_states=True,
        ).eval()

        print(f"Applying merge: new = (1+alpha)*ft - alpha*forget (alpha={args.alpha})")
        ft_params = dict(ft_model.named_parameters())
        with torch.no_grad():
            for name, forget_param in forget_model.named_parameters():
                if not filter_func(name, forget_param):
                    continue
                if name not in ft_params:
                    print(f"Warning: Parameter {name} not found in ft model")
                    continue
                # new = (1+alpha)*ft - alpha*forget
                ft_param = ft_params[name]
                ft_param.mul_(1.0 + args.alpha)
                ft_param.add_(forget_param.to(ft_param.device), alpha=-args.alpha)

        del forget_model
        gc.collect()
        torch.cuda.empty_cache()
    
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


# python effiecient_task_vector.py \
#   --tokenizer_path /path/to/tokenizer \
#   --ft_model_path /path/to/ft_model \
#   --forget_path /path/to/forget_model \
#   --save_path /path/to/save \
#   --alpha 1.0