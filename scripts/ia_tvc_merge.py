import argparse
import gc
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interference-aware retain-subspace TVC model merge"
    )
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--orig_model_path", type=str, required=True)
    parser.add_argument("--ft_model_path", type=str, required=True)
    parser.add_argument("--forget_path", type=str, required=True)
    parser.add_argument("--retain_paths", nargs="+", required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="Relative ridge multiplied by mean diag(T_R^T T_R).",
    )
    parser.add_argument("--absolute_ridge", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument(
        "--forget_metric",
        choices=["none", "identity", "rank1"],
        default="rank1",
        help="Cheap proxy for the forget-reactivation penalty.",
    )
    parser.add_argument(
        "--filter_type",
        choices=["mlp", "attn", "all_linear", "no_lm_head", "all"],
        default="mlp",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--clip_weight_abs", type=float, default=5.0)
    parser.add_argument("--trust_remote_code", action="store_true")
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
        keys = ("mlp", "up_proj", "down_proj", "gate_proj", "fc1", "fc2", ".fc")
        return lambda name, param: any(key in name for key in keys)
    if filter_type == "attn":
        keys = ("self_attn", "attn", "q_proj", "k_proj", "v_proj", "o_proj", "dense")
        return lambda name, param: any(key in name for key in keys)
    if filter_type == "all_linear":
        keys = ("proj", "dense", "fc", "lm_head")
        return lambda name, param: any(key in name for key in keys)
    if filter_type == "no_lm_head":
        keys = ("proj", "dense", "fc")
        return lambda name, param: any(key in name for key in keys) and "lm_head" not in name
    if filter_type == "all":
        return lambda name, param: torch.is_floating_point(param)
    raise ValueError(f"Unknown filter type: {filter_type}")


def load_model(path, torch_dtype, trust_remote_code):
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    ).eval()


def selected(name, param, filter_func):
    return torch.is_floating_point(param) and filter_func(name, param)


def compute_projection_system(orig_model, forget_model, retain_models, filter_func, alpha):
    orig_params = dict(orig_model.named_parameters())
    forget_params = dict(forget_model.named_parameters())
    retain_params = [dict(model.named_parameters()) for model in retain_models]

    k_retain = len(retain_models)
    gram = torch.zeros((k_retain, k_retain), dtype=torch.float64)
    rhs = torch.zeros(k_retain, dtype=torch.float64)
    forget_norm_sq = torch.tensor(0.0, dtype=torch.float64)
    selected_tensors = 0
    selected_elements = 0
    skipped = 0

    with torch.no_grad():
        for name, orig_param in orig_params.items():
            if not selected(name, orig_param, filter_func):
                continue
            if name not in forget_params or any(name not in rp for rp in retain_params):
                skipped += 1
                continue

            orig_data = orig_param.detach().to(dtype=torch.float32)
            forget_data = forget_params[name].detach().to(dtype=torch.float32)
            forget_vec = (forget_data - orig_data).reshape(-1)

            retain_vecs = []
            for rp in retain_params:
                retain_data = rp[name].detach().to(dtype=torch.float32)
                retain_vecs.append((retain_data - orig_data).reshape(-1))

            forget_norm_sq += torch.dot(forget_vec, forget_vec).double()
            for i, retain_vec_i in enumerate(retain_vecs):
                rhs[i] += torch.dot(retain_vec_i, forget_vec).double()
                for j in range(i, k_retain):
                    value = torch.dot(retain_vec_i, retain_vecs[j]).double()
                    gram[i, j] += value
                    if i != j:
                        gram[j, i] += value

            selected_tensors += 1
            selected_elements += orig_param.numel()

    if selected_tensors == 0:
        raise RuntimeError("No parameters matched --filter_type.")

    return {
        "gram": gram,
        "rhs": rhs * alpha,
        "rhs_unscaled": rhs,
        "forget_norm_sq": forget_norm_sq,
        "selected_tensors": selected_tensors,
        "selected_elements": selected_elements,
        "skipped_tensors": skipped,
    }


def solve_weights(system, args):
    gram = system["gram"]
    rhs = system["rhs"]
    lhs = gram.clone()

    if args.forget_metric == "identity":
        lhs = lhs + args.gamma * gram
    elif args.forget_metric == "rank1":
        denom = system["forget_norm_sq"].clamp_min(1e-12)
        leakage = torch.outer(system["rhs_unscaled"], system["rhs_unscaled"]) / denom
        lhs = lhs + args.gamma * leakage
    elif args.forget_metric != "none":
        raise ValueError(f"Unknown forget metric: {args.forget_metric}")

    diag_scale = torch.diag(gram).mean().clamp_min(1.0)
    actual_ridge = args.ridge * diag_scale + args.absolute_ridge
    lhs = lhs + actual_ridge * torch.eye(lhs.shape[0], dtype=lhs.dtype)

    try:
        weights = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        weights = torch.linalg.lstsq(lhs, rhs).solution

    if args.clip_weight_abs and args.clip_weight_abs > 0:
        weights = weights.clamp(-args.clip_weight_abs, args.clip_weight_abs)

    return weights, actual_ridge


def apply_merge(ft_model, orig_model, forget_model, retain_models, filter_func, alpha, weights):
    orig_params = dict(orig_model.named_parameters())
    forget_params = dict(forget_model.named_parameters())
    retain_params = [dict(model.named_parameters()) for model in retain_models]

    merged = 0
    skipped = 0
    orig_coef = alpha - float(weights.sum().item())

    with torch.no_grad():
        for name, ft_param in ft_model.named_parameters():
            if not selected(name, ft_param, filter_func):
                continue
            if name not in orig_params or name not in forget_params:
                skipped += 1
                continue
            if any(name not in rp for rp in retain_params):
                skipped += 1
                continue

            device = ft_param.device
            dtype = ft_param.dtype
            ft_param.add_(orig_params[name].to(device=device, dtype=dtype), alpha=orig_coef)
            ft_param.add_(forget_params[name].to(device=device, dtype=dtype), alpha=-alpha)
            for weight, rp in zip(weights.tolist(), retain_params):
                ft_param.add_(rp[name].to(device=device, dtype=dtype), alpha=float(weight))
            merged += 1

    return merged, skipped


def main():
    args = parse_args()
    torch_dtype = resolve_dtype(args.torch_dtype)
    tokenizer_path = args.tokenizer_path or args.orig_model_path
    filter_func = get_filter_func(args.filter_type)

    print("IA-TVC merge configuration:")
    print(f"  base/original: {args.orig_model_path}")
    print(f"  target/full:    {args.ft_model_path}")
    print(f"  forget model:   {args.forget_path}")
    print(f"  retain models:  {args.retain_paths}")
    print(f"  save path:      {args.save_path}")
    print(f"  alpha:          {args.alpha}")
    print(f"  gamma:          {args.gamma}")
    print(f"  ridge:          {args.ridge}")
    print(f"  forget metric:  {args.forget_metric}")
    print(f"  filter type:    {args.filter_type}")

    print("\nLoading models...")
    orig_model = load_model(args.orig_model_path, torch_dtype, args.trust_remote_code)
    ft_model = load_model(args.ft_model_path, torch_dtype, args.trust_remote_code)
    forget_model = load_model(args.forget_path, torch_dtype, args.trust_remote_code)
    retain_models = [
        load_model(path, torch_dtype, args.trust_remote_code) for path in args.retain_paths
    ]

    print("\nComputing retain-subspace projection system...")
    system = compute_projection_system(
        orig_model, forget_model, retain_models, filter_func, args.alpha
    )
    weights, actual_ridge = solve_weights(system, args)

    print("\nProjection statistics:")
    print(f"  selected tensors:  {system['selected_tensors']}")
    print(f"  selected elements: {system['selected_elements']:,}")
    print(f"  skipped tensors:   {system['skipped_tensors']}")
    print(f"  forget norm sq:    {system['forget_norm_sq'].item():.6e}")
    print(f"  actual ridge:      {actual_ridge.item():.6e}")
    for idx, weight in enumerate(weights.tolist()):
        print(f"  retain weight[{idx}]: {weight:.6f}")

    print("\nApplying merge:")
    print("  theta_u = theta_target - alpha*tau_forget + sum_k w_k*tau_retain_k")
    merged, skipped = apply_merge(
        ft_model, orig_model, forget_model, retain_models, filter_func, args.alpha, weights
    )
    print(f"  merged tensors:  {merged}")
    print(f"  skipped tensors: {skipped}")

    del orig_model, forget_model, retain_models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nSaving merged model to {args.save_path}...")
    os.makedirs(args.save_path, exist_ok=True)
    ft_model.save_pretrained(args.save_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=args.trust_remote_code
        )
        tokenizer.save_pretrained(args.save_path)
    except Exception as exc:
        print(f"Warning: tokenizer save failed: {exc}")

    metadata = {
        "method": "IA-TVC",
        "formula": "theta_target - alpha*tau_forget + sum_k w_k*tau_retain_k",
        "args": vars(args),
        "weights": [float(w) for w in weights.tolist()],
        "actual_ridge": float(actual_ridge.item()),
        "selected_tensors": int(system["selected_tensors"]),
        "selected_elements": int(system["selected_elements"]),
        "forget_norm_sq": float(system["forget_norm_sq"].item()),
        "gram": system["gram"].tolist(),
        "rhs": system["rhs"].tolist(),
    }
    with open(os.path.join(args.save_path, "ia_tvc_merge_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
