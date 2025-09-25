# -*- coding: utf-8 -*-
import math
import copy
import logging
from typing import Dict, List, Tuple, Optional
from torch.nn import functional as F
import random
import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------
# Utility: find a submodule by its dotted name, and return (parent, attr_name, module)
# ---------------------------
def find_module(root: nn.Module, dotted: str) -> Tuple[nn.Module, str, nn.Module]:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1], getattr(parent, parts[-1])


# ---------------------------
# LinearSVD layer
# ---------------------------
class LinearSVD(nn.Module):
    """
    A robust custom layer that decomposes a linear operation into two low-rank matrices.
    """
    def __init__(
        self,
        orig_linear: nn.Linear,
        rank: int,
    ):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.rank = rank

        # Bias first
        if orig_linear.bias is not None:
            self.bias = nn.Parameter(orig_linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        # Placeholder factors (will be replaced)
        A = torch.empty(self.out_features, self.rank, device=orig_linear.weight.device, dtype=orig_linear.weight.dtype)
        B = torch.empty(self.rank, self.in_features, device=orig_linear.weight.device, dtype=orig_linear.weight.dtype)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(B, a=math.sqrt(5))

        self.linear_A = nn.Parameter(A)
        self.linear_B = nn.Parameter(B)

        self._cache_X: Optional[torch.Tensor]     = None  # [N, in]
        self._cache_delta: Optional[torch.Tensor] = None  # [N, out]

    def _bw_hook(self, grad_out: torch.Tensor):
        # grad_out shape == output shape == preact grad δ
        # 只存一份 detach 后的缓存；按需转成 float32 更稳
        self._cache_delta = grad_out.detach()

    def forward(self, input: torch.Tensor):
        # y = (A) @ (B @ x) + b
        self._cache_X = input.detach()
        intermediate = f.linear(input, self.linear_B)
        Z = F.linear(intermediate, self.linear_A, bias=self.bias)

        Z.register_hook(self._bw_hook)
        return Z
    
    def pop_cached_X_delta(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        X, delta = self._cache_X, self._cache_delta
        self._cache_X = None
        self._cache_delta = None
        return X, delta

    def get_cached_X_delta(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._cache_X, self._cache_delta

    def __repr__(self):
        bias_str = f", bias={hasattr(self, 'bias') and self.bias is not None}"
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}{bias_str})"

    def get_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.linear_A, self.linear_B


# ---------------------------
# IPSVD-aware core: per-layer closed form to compute A,B from W, H, Delta
# ---------------------------
@torch.no_grad()
def IPSVD_factors_from_probes(
    W: torch.Tensor,      # [out, in] 仅用于 dtype/device
    H: torch.Tensor,      # [in, N]
    Delta: torch.Tensor,  # [out, N]
    rank: int,
    ridge: float = 1e-3,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = torch.device(device) if device is not None else (W.device if W.is_cuda else torch.device("cpu"))

    H32 = H.to(torch.float32, copy=False).to(dev)
    D32 = Delta.to(torch.float32, copy=False).to(dev)

    if H32.numel() == 0 or D32.numel() == 0 or H32.size(1) == 0:
        out_f, in_f = W.shape
        r = max(1, min(rank, in_f, out_f))
        A = torch.zeros(out_f, r, dtype=torch.float32, device="cpu")
        B = torch.zeros(r, in_f, dtype=torch.float32, device="cpu")
        return A.contiguous(), B.contiguous()

    if torch.linalg.matrix_norm(D32).item() == 0.0:
        D32 = D32 + 1e-8 * torch.randn_like(D32)

    U_H, S_H, _ = torch.linalg.svd(H32, full_matrices=False)   # [in, N] = U_H diag(S_H) V_H^T
    U_D, S_D, _ = torch.linalg.svd(D32, full_matrices=False)   # [out, N] = U_D diag(S_D) V_D^T

    N = H32.size(1)
    eps = 1e-6
    S_H = torch.clamp(S_H, min=eps)
    S_D = torch.clamp(S_D, min=eps)

    D_H = torch.sqrt(S_H**2 / max(N, 1) + ridge)               # [in_rank]
    D_D = torch.sqrt(S_D**2 / max(N, 1) + ridge)               # [out_rank]

    W32 = W.to(torch.float32, copy=True).to(dev)

    core = (U_D.mT @ W32 @ U_H)
    core = (D_D[:, None] * core) * D_H[None, :]

    P, S_core, Qh = torch.linalg.svd(core, full_matrices=False)
    r = int(min(rank, S_core.shape[0]))
    r = max(r, 1)
    P_r = P[:, :r]                      # [out_rank, r]
    S_r = S_core[:r]                    # [r]
    Q_r = Qh[:r, :].mT                  # [in_rank, r]

    S_r_sqrt = S_r.sqrt()

    # A = U_D * D_D^{-1} @ P_r @ diag(S^{1/2})                 -> [out, r]
    A32 = (U_D * (1.0 / D_D)[None, :]) @ (P_r * S_r_sqrt[None, :])

    # B = diag(S^{1/2}) @ Q_r^T @ D_H^{-1} @ U_H^T             -> [r, in]
    B32 = (S_r_sqrt[:, None] * Q_r.mT) @ ((1.0 / D_H)[:, None] * U_H.mT)

    A_cpu = A32.to("cpu", dtype=torch.float32, copy=False).contiguous()
    B_cpu = B32.to("cpu", dtype=torch.float32, copy=False).contiguous()
    return A_cpu, B_cpu

from typing import Dict, List
def _resolve_target_map(model: nn.Module,
                                     target_modules: List[str]) -> Dict[str, nn.Linear]:
    mapping: Dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(sfx) for sfx in target_modules):
            mapping[name] = module
    if not mapping:
        logger.warning("[IPSVD] No matching Linear modules found by suffix.")
    return mapping

def collect_IPSVD_probes(
    model: nn.Module,
    dataloader,
    target_modules: Dict[str, nn.Linear],
    device: str,
    final_probes_per_layer: int,
    total_sequences_hint: Optional[int] = None,
    k_per_sequence: Optional[int] = None,
    move_to_cpu: bool = True,
    dtype: torch.dtype = torch.float32,
    domain_tag: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:

    def _log(msg: str):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    model.eval()
    _acts: Dict[str, torch.Tensor] = {}
    _grads: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _pick_token_indices_per_seq(attn_mask_row: torch.Tensor, k: int, device=None) -> torch.Tensor:
        valid = torch.nonzero(attn_mask_row.bool(), as_tuple=False).squeeze(-1)
        if valid.numel() <= k:
            out = valid
        else:
            perm = torch.randperm(valid.numel(), device=valid.device)[:k]
            out = valid[perm]
        if device is not None and out.device != device:
            out = out.to(device, non_blocking=True)
        return out

    # ---- hooks ----
    def _fw_hook(module, inp, out):
        _acts[module._IPSVD_name] = inp[0].detach()

    def _bw_hook(module, grad_in, grad_out):
        _grads[module._IPSVD_name] = grad_out[0].detach()

    handles = []
    target_modules = _resolve_target_map(model, target_modules)
    for name, mod in target_modules.items():
        mod._IPSVD_name = name
        handles.append(mod.register_forward_hook(_fw_hook))
        handles.append(mod.register_full_backward_hook(_bw_hook))

    H_data = {name: [] for name in target_modules}
    D_data = {name: [] for name in target_modules}

    if k_per_sequence is None:
        if total_sequences_hint is None:
            try:
                total_sequences_hint = len(dataloader.dataset)
            except Exception:
                total_sequences_hint = 1
        k_per_sequence = max(1, final_probes_per_layer // max(1, total_sequences_hint))

    pbar = tqdm(dataloader, desc="Collecting (H, Δ)")
    for batch in pbar:
        model.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = input_ids.masked_fill(attention_mask == 0, -100)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()  

        B = input_ids.size(0)

        for name in target_modules:
            h = _acts[name]   # [B, T, in] / [B, in]
            d = _grads[name]  # [B, T, out] / [B, out]

            if h.dim() == 2:
                h_sel = h                      # [B, in]
                d_sel = d                      # [B, out]
            else:
                picks_h, picks_d = [], []
                for b in range(B):
                    idx = _pick_token_indices_per_seq(attention_mask[b], k_per_sequence)
                    if idx.numel() == 0:
                        continue
                    idx = idx.to(h.device)
                    picks_h.append(h[b, idx, :])  # [m, in]
                    picks_d.append(d[b, idx, :])  # [m, out]
                if len(picks_h) == 0:
                    continue
                h_sel = torch.cat(picks_h, dim=0)  # [sum_m, in]
                d_sel = torch.cat(picks_d, dim=0)  # [sum_m, out]

            H_data[name].append(h_sel.to(dtype).T.contiguous())  # [in, n_batch]
            D_data[name].append(d_sel.to(dtype).T.contiguous())  # [out, n_batch]

        _acts.clear(); _grads.clear()

    for h in handles:
        h.remove()

    layer_probes: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in target_modules:
        if len(H_data[name]) == 0 or len(D_data[name]) == 0:
            continue

        H_cat = torch.cat(H_data[name], dim=1)
        D_cat = torch.cat(D_data[name], dim=1)
        N = min(final_probes_per_layer, H_cat.size(1), D_cat.size(1))
        H_cat = H_cat[:, :N]
        D_cat = D_cat[:, :N]

        if move_to_cpu:
            H_cat = H_cat.cpu()
            D_cat = D_cat.cpu()

        in_f, out_f = H_cat.size(0), D_cat.size(0)
        layer_probes[name] = {"H": H_cat, "Delta": D_cat, "in": in_f, "out": out_f}

        tag = f"[{domain_tag}] " if domain_tag else ""
        _log(f"{tag}[IPSVD] Layer {name}: collected N={N} probe columns (in={in_f}, out={out_f}).")

    return layer_probes

# ---------------------------
# High-level: compute IPSVD-aware factors for all target layers and initialize a proxy
# ---------------------------
def estimate_rank(m: int, n: int, sparsity: float) -> int:
    assert 0 <= sparsity <= 1, "sparsity has to be in [0,1]."
    
    total_params = m * n
    target_params = (1 - sparsity) * total_params
    r = target_params / (m + n)
    
    r = max(1, min(int(round(r)), min(m, n)))
    return r

def init_proxy_model_with_IPSVD(
    base_model: nn.Module,
    loader_src: torch.utils.data.DataLoader,
    init_method: str = "IPSVD",
    target_modules: List[str] = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj'),
    sparsity: float = 0.3,
    min_rank_multiple: int = 128,
    ridge_lambda: float = 1e-3,
    max_probes_per_layer: int = 1024,
    freeze_non_md_param: bool = True,
) -> nn.Module:

    device = next(base_model.parameters()).device
    proxy_model = copy.deepcopy(base_model)

    base_targets: Dict[str, nn.Linear] = {
        name: mod for name, mod in base_model.named_modules()
        if isinstance(mod, nn.Linear) and any(name.endswith(sfx) for sfx in target_modules)
    }
    if not base_targets:
        logger.warning("No base target linear modules found; abort init.")
        return proxy_model

    probes_src: Dict[str, Dict[str, torch.Tensor]] = {}
    if init_method == "IPSVD":
        logger.info("[IPSVD] init_method=IPSVD → collecting probes from SRC...")
        with torch.enable_grad():
            probes_src = collect_IPSVD_probes(
                model=base_model,
                dataloader=loader_src,
                target_modules=list(target_modules),
                final_probes_per_layer=max_probes_per_layer,
                device=device,
                domain_tag="SRC",
            )
    else:
        logger.info("[IPSVD] init_method=RANDOM or SVD → skip probe collection; use RANDOM A/B init.")

    trainable_AB: List[nn.Parameter] = []
    replaced = 0

    for bname, bmod in tqdm(base_targets.items()):
        in_f, out_f = bmod.in_features, bmod.out_features
        max_rank = min(in_f, out_f)
        raw_rank = max(1, estimate_rank(in_f, out_f, sparsity))
        layer_rank = ((raw_rank + min_rank_multiple - 1) // min_rank_multiple) * min_rank_multiple if min_rank_multiple > 1 else raw_rank

        if layer_rank >= max_rank:
            logger.info(f"[INIT] {bname}: rank={layer_rank} >= max_rank={max_rank}, keep full-rank; skip replacement.")
            continue

        try:
            parent, attr, smod = find_module(proxy_model, bname)
        except Exception:
            smod = None
            tail = ".".join(bname.split(".")[-2:])
            parent = attr = None
            for sname, m in proxy_model.named_modules():
                if sname.endswith(tail) and isinstance(m, nn.Linear):
                    try:
                        parent, attr, _ = find_module(proxy_model, sname)
                    except Exception:
                        parent = None; attr = None
                    smod = m; bname = sname; break

        if smod is None or not isinstance(smod, nn.Linear) or parent is None or attr is None:
            logger.warning(f"[INIT] Cannot find matching nn.Linear in proxy for {bname}; skip.")
            continue

        if init_method == "IPSVD":
            # IPSVD init：collect H、Delta
            if bname not in probes_src:
                logger.warning(f"[IPSVD] No probe for {bname}; fallback RANDOM init.")
                new_module = LinearSVD(orig_linear=smod, rank=layer_rank)
            else:
                H = probes_src[bname]["H"]         # [in, N]
                Delta = probes_src[bname]["Delta"] # [out, N]
                try:
                    A_cpu, B_cpu = IPSVD_factors_from_probes(
                        W=bmod.weight.data, H=H, Delta=Delta, rank=layer_rank, ridge=ridge_lambda
                    )
                    new_module = LinearSVD(orig_linear=smod, rank=layer_rank).to(bmod.weight.device)
                    new_module.linear_A.data.copy_(A_cpu.to(dtype=new_module.linear_A.dtype))
                    new_module.linear_B.data.copy_(B_cpu.to(dtype=new_module.linear_B.dtype))
                    logger.info(f"[IPSVD] {bname}: init by IPSVD (rank={layer_rank}, N={H.size(1)}).")
                except Exception as e:
                    logger.warning(f"[IPSVD] Factor computation failed for {bname} ({e}); fallback RANDOM init.")
                    new_module = LinearSVD(orig_linear=smod, rank=layer_rank).to(bmod.weight.device)
        elif init_method == "SVD":
            try:
                U, S, Vt = torch.linalg.svd(bmod.weight.data, full_matrices=False)
                A_cpu = U[:, :layer_rank] @ torch.diag(S[:layer_rank])
                B_cpu = Vt[:layer_rank, :]
                new_module = LinearSVD(orig_linear=smod, rank=layer_rank).to(bmod.weight.device)
                new_module.linear_A.data.copy_(A_cpu.to(dtype=new_module.linear_A.dtype))
                new_module.linear_B.data.copy_(B_cpu.to(dtype=new_module.linear_B.dtype))
                logger.info(f"[INIT] {bname}: init by SVD (rank={layer_rank}).")
            except Exception as e:
                logger.warning(f"[SVD] Factor computation failed for {bname} ({e}); fallback RANDOM init.")
                new_module = LinearSVD(orig_linear=smod, rank=layer_rank).to(bmod.weight.device)
        else:
            new_module = LinearSVD(orig_linear=smod, rank=layer_rank)
            logger.info(f"[INIT] {bname}: RANDOM A/B (rank={layer_rank}).")

        if smod.bias is not None and getattr(new_module, "bias", None) is None:
            new_module.bias = nn.Parameter(smod.bias.data.clone(), requires_grad=False)

        setattr(parent, attr, new_module)
        replaced += 1

        new_module.linear_A.requires_grad_(True)
        new_module.linear_B.requires_grad_(True)
        if new_module.bias is not None:
            new_module.bias.requires_grad_(False)
        trainable_AB.extend([new_module.linear_A, new_module.linear_B])

    logger.info(f"[INIT] Total replaced layers: {replaced}")

    if freeze_non_md_param:
        total, trainable = 0, 0
        trainable_set = {p for p in trainable_AB}
        for p in proxy_model.parameters():
            total += p.numel()
            if p in trainable_set:
                p.requires_grad_(True); trainable += p.numel()
            else:
                p.requires_grad_(False)
        ratio = 100.0 * (trainable / max(1, total))
        logger.info(f"[INIT] Trainable params after replacement (A/B only): {trainable}/{total} ({ratio:.2f}%)")
    else:
        logger.info("[INIT] freeze_non_md_param=False → keep default requires_grad (A/B are trainable; others unchanged).")

    return proxy_model

def save_proxy_model(model: nn.Module, save_path: str):
    """
    Saves the clean weights and biases from all LinearSVD layers in a model.
    """
    svd_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, LinearSVD):
            # CORRECT: Save the .weight and .bias tensors from the submodules
            # Using consistent and clear keys.
            svd_state_dict[f"{name}.linear_A"] = module.linear_A.data.cpu()
            svd_state_dict[f"{name}.linear_B"] = module.linear_B.data.cpu()
            if module.bias is not None:
                svd_state_dict[f"{name}.bias"] = module.bias.data.cpu()

    torch.save(svd_state_dict, save_path)
    logger.info(f"Saved clean SVD parameters to: {save_path}")

def load_proxy_model(model: nn.Module, load_path: str):
    """
    Loads clean weights and biases into all LinearSVD layers in a model.
    """
    svd_state_dict = torch.load(load_path, map_location='cpu')
    logger.info(f"Loading SVD parameters from: {load_path}")

    for name, module in model.named_modules():
        if isinstance(module, LinearSVD):
            # CORRECT: Use keys that match the save function
            a_key = f"{name}.linear_A"
            b_key = f"{name}.linear_B"
            bias_key = f"{name}.bias"

            # Check if keys exist and load data into the correct tensors
            if a_key in svd_state_dict:
                # Get the target device from the module itself
                device = module.linear_A.device
                module.linear_A.data.copy_(svd_state_dict[a_key].to(device))

            if b_key in svd_state_dict:
                device = module.linear_B.device
                module.linear_B.data.copy_(svd_state_dict[b_key].to(device))

            if bias_key in svd_state_dict and module.bias is not None:
                device = module.bias.device
                module.bias.data.copy_(svd_state_dict[bias_key].to(device))

    logger.info(f"Loaded clean SVD parameters from: {load_path}")

class InferenceOnlySVDLinear(nn.Module):
    """
    A hyper-efficient, inference-only version of the decomposed linear layer.
    It contains no noise logic, no dynamic operations, and its forward pass is
    a simple, sequential call to two nn.Linear layers, making it maximally
    compatible with JIT compilation and other optimizations.
    """
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True, dtype=torch.bfloat16):
        super().__init__()
        self.linear_B = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.linear_A = nn.Linear(rank, out_features, bias=bias, dtype=dtype)
        
        self._equiv_hooks_enabled = False
        self._equiv_accumulate = True
        self._equiv_accum_dtype = None
        self._last_input = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The fastest possible implementation: direct, sequential calls.
        return self.linear_A(self.linear_B(x))

def disable_ab_param_grads(model: nn.Module):
    real = getattr(model, "module", model)
    for m in real.modules():
        if isinstance(m, InferenceOnlySVDLinear):
            m.linear_A.weight.requires_grad_(False)
            m.linear_B.weight.requires_grad_(False)

def load_compressed_model_for_inference(
    base_model_architecture: nn.Module, 
    svd_weights_path: str
) -> nn.Module:
    """
    Loads SVD weights into a base model architecture, replacing target layers
    with the hyper-efficient InferenceOnlySVDLinear module.

    Args:
        base_model_architecture: An instance of the model architecture (e.g., from AutoModelForCausalLM.from_config).
        svd_weights_path: Path to the .pt file saved by save_proxy_model.
    
    Returns:
        A new model optimized for fast inference.
    """
    logger.info(f"Building fast inference model from SVD weights at: {svd_weights_path}")
    
    inference_model = copy.deepcopy(base_model_architecture)
    svd_state_dict = torch.load(svd_weights_path, map_location='cpu')

    # Group weights by layer name
    layer_weights = {}
    for key, value in svd_state_dict.items():
        # Key format: "model.layers.0.self_attn.q_proj.linear_A.weight"
        # We want to group by "model.layers.0.self_attn.q_proj"
        base_name = key.rsplit('.', 1)[0]
        if base_name not in layer_weights:
            layer_weights[base_name] = {}
        layer_weights[base_name][key] = value

    for name, weights_dict in layer_weights.items():
        try:
            # Get info from the loaded weights
            weight_a = weights_dict[f"{name}.linear_A"]
            weight_b = weights_dict[f"{name}.linear_B"]
            bias = weights_dict.get(f"{name}.bias", None)

            out_features, rank = weight_a.shape
            _, in_features = weight_b.shape
            
            parent_module, child_name, original_module = find_module(inference_model, name)
            target_device = next(original_module.parameters()).device
            # Create the new, efficient module
            new_module = InferenceOnlySVDLinear(
                in_features=in_features,
                out_features=out_features,
                rank=rank,
                bias=(bias is not None),
                dtype=next(original_module.parameters()).dtype
            )
            
            # Load the weights into the new module
            new_module.linear_A.weight.data.copy_(weight_a.contiguous())
            new_module.linear_B.weight.data.copy_(weight_b.contiguous())
            if bias is not None:
                new_module.linear_A.bias.data.copy_(bias.contiguous())

            new_module.to(target_device)
            # Replace the original nn.Linear in the inference model
            setattr(parent_module, child_name, new_module)
            logger.info(f"Replaced layer '{name}' with InferenceOnlySVDLinear.")

        except Exception as e:
            logger.error(f"Failed to replace layer {name}: {e}", exc_info=True)

    return inference_model
