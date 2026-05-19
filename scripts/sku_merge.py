import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory-efficient SKU task-vector negation merge script"
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer. Defaults to --orig_model_path when omitted.",
    )
    parser.add_argument(
        "--orig_model_path",
        type=str,
        required=True,
        help="Path to the original reference/base model theta_o.",
    )
    parser.add_argument(
        "--acquired_model_path",
        type=str,
        required=True,
        help="Path to the SKU acquisition-stage model theta_a.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the merged SKU unlearned model.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scaling coefficient in theta_u = theta_o - alpha * (theta_a - theta_o).",
    )
    parser.add_argument(
        "--filter_type",
        type=str,
        default="all",
        choices=["mlp", "attn", "all"],
        help="Which parameters to merge: mlp, attn, or all.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device number kept for interface compatibility with existing scripts.",
    )

    return parser.parse_args()


def resolve_dtype(dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def get_filter_func(filter_type):
    if filter_type == "mlp":
        return lambda n, p: (
            "mlp" in n
            or "up_proj" in n
            or "down_proj" in n
            or "gate_proj" in n
            or ".fc" in n
        )
    if filter_type == "attn":
        return lambda n, p: (
            "attn" in n
            or "self_attn" in n
            or "q_proj" in n
            or "k_proj" in n
            or "v_proj" in n
            or "o_proj" in n
        )
    if filter_type == "all":
        return lambda n, p: torch.is_floating_point(p) or torch.is_complex(p)
    raise ValueError(f"Unknown filter type: {filter_type}")


def main():
    args = parse_args()
    tokenizer_path = args.tokenizer_path or args.orig_model_path
    torch_dtype = resolve_dtype(args.torch_dtype)
    filter_func = get_filter_func(args.filter_type)

    print("Loading models with the following configuration:")
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  Original model: {args.orig_model_path}")
    print(f"  Acquired model: {args.acquired_model_path}")
    print(f"  Save path: {args.save_path}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Filter type: {args.filter_type}")
    print(f"  Dtype: {args.torch_dtype}")
    print(f"  Device: cuda:{args.device}")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
    )

    # Keep the original model as the writable destination so we avoid creating an
    # additional full model copy during the merge stage.
    print("Loading original model...")
    orig_model = AutoModelForCausalLM.from_pretrained(
        args.orig_model_path,
        torch_dtype=torch_dtype,
        output_hidden_states=True,
    ).eval()

    print("Loading acquired model...")
    acquired_model = AutoModelForCausalLM.from_pretrained(
        args.acquired_model_path,
        torch_dtype=torch_dtype,
        output_hidden_states=True,
    ).eval()

    print("\nApplying memory-efficient SKU negation merge...")
    orig_params = dict(orig_model.named_parameters())
    merged_count = 0
    skipped_count = 0

    with torch.no_grad():
        for name, acquired_param in acquired_model.named_parameters():
            if name not in orig_params:
                print(f"Warning: parameter missing in original model: {name}")
                skipped_count += 1
                continue

            orig_param = orig_params[name]
            if not filter_func(name, orig_param):
                skipped_count += 1
                continue

            if not (torch.is_floating_point(orig_param) or torch.is_complex(orig_param)):
                skipped_count += 1
                continue

            # SKU stage 2:
            #   theta_u = theta_o - alpha * (theta_a - theta_o)
            # We update theta_o in-place to avoid constructing an extra full model
            # copy or an explicit task-vector param object.
            orig_param.mul_(1.0 + args.alpha)
            orig_param.add_(acquired_param.to(orig_param.device), alpha=-args.alpha)
            merged_count += 1

    print(f"Merged parameters: {merged_count}")
    print(f"Skipped parameters: {skipped_count}")

    del acquired_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nSaving merged model to {args.save_path}...")
    orig_model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
