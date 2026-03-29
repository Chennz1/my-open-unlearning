import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Model merge script for unlearning with retain")
    
    # Model paths
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--ft_model_path", type=str, required=True, help="Path to target (fine-tuned) model")
    parser.add_argument("--forget_path", type=str, required=True, help="Path to forget reinforce model")
    parser.add_argument("--retain_path", type=str, required=True, help="Path to retain reinforce model")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save merged model")

    # Coefficients
    parser.add_argument("--alpha", type=float, default=1.0, help="Coefficient for forget task vector")
    parser.add_argument("--beta", type=float, default=1.0, help="Coefficient for retain task vector")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading models with configuration:")
    print(f"  Target model: {args.ft_model_path}")
    print(f"  Forget model: {args.forget_path}")
    print(f"  Retain model: {args.retain_path}")
    print(f"  Alpha: {args.alpha}, Beta: {args.beta}")
    
    filter_func = lambda n, p: True if "up_proj" in n or "down_proj" in n or "gate_proj" in n or "fc" in n else False
    
    print("\nLoading target model (base)...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.ft_model_path, 
        torch_dtype=torch.float16,
        output_hidden_states=True
    ).eval()
    
    ft_params = dict(ft_model.named_parameters())
    
    # Formula: Target - alpha*(Forget - Target) + beta*(Retain - Target)
    # Target + alpha*Target - alpha*Forget + beta*Retain - beta*Target
    # (1 + alpha - beta)*Target - alpha*Forget + beta*Retain
    
    # To save memory, modify target inplace:
    # target = (1 + alpha - beta) * target
    
    coef_target = 1.0 + args.alpha - args.beta
    print(f"\nApplying target coefficient: {coef_target}")
    with torch.no_grad():
        for name, ft_param in ft_model.named_parameters():
            if not filter_func(name, None):
                continue
            ft_param.mul_(coef_target)
            
    print("\nLoading forget model...")
    forget_model = AutoModelForCausalLM.from_pretrained(args.forget_path, torch_dtype=torch.float16).eval()
    with torch.no_grad():
        for name, forget_param in forget_model.named_parameters():
            if not filter_func(name, param:=None):
                continue
            if name in ft_params:
                ft_params[name].add_(forget_param.to(ft_params[name].device), alpha=-args.alpha)
                
    del forget_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nLoading retain model...")
    retain_model = AutoModelForCausalLM.from_pretrained(args.retain_path, torch_dtype=torch.float16).eval()
    with torch.no_grad():
        for name, retain_param in retain_model.named_parameters():
            if not filter_func(name, param:=None):
                continue
            if name in ft_params:
                ft_params[name].add_(retain_param.to(ft_params[name].device), alpha=args.beta)
                
    del retain_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\nSaving merged model to {args.save_path}...")
    ft_model.save_pretrained(args.save_path)
    try:
        AutoTokenizer.from_pretrained(args.tokenizer_path).save_pretrained(args.save_path)
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")
        
    print("Done!")

if __name__ == "__main__":
    main()
