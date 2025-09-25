
import torch
from torch import nn
from torch.autograd import grad as autograd_grad
import torch.nn.functional as F

from typing import List, Tuple
import logging
import os
from tqdm import tqdm
from utils.init_with_ipsvd import LinearSVD, save_proxy_model
# --- Basic Setup ---
logger = logging.getLogger(__name__)

def get_target_layer_pairs(target_model: nn.Module, proxy_model: nn.Module, target_modules: list[str]) -> list[tuple[nn.Module, LinearSVD]]:
    """
    Finds corresponding wrapped target layers and
    proxy layers (LinearSVD) based on module names containing target substrings.

    Args:
        target_model: target model.
        proxy_model: proxy model (with LinearSVD layers).
        target_modules: List of substrings identifying target layers.

    Returns:
        A list of tuples: [(target_wrapped_layer, proxy_svd_layer), ...]
    """
    pairs = []
    target_layers = {}
    proxy_layers = {}

    # Find all relevant layers in both models
    for name, module in target_model.named_modules():
        is_target = any(target in name.split('.')[-1] for target in target_modules)
        if is_target:
            target_layers[name] = module

    for name, module in proxy_model.named_modules():
        is_target = isinstance(module, LinearSVD) and any(target in name.split('.')[-1] for target in target_modules)
        if is_target:
            proxy_layers[name] = module

    # Match layers by name
    for name, target_layer in target_layers.items():
        if name in proxy_layers:
            proxy_layer = proxy_layers[name]
            # Basic check: ensure the original dimensions match
            if (target_layer.in_features == proxy_layer.in_features and
                target_layer.out_features == proxy_layer.out_features):
                 if proxy_layer.rank > 0: # Only pair if proxy layer is decomposed
                    pairs.append((target_layer, proxy_layer))
                 else:
                     logger.warning(f"Skipping pair for {name}: proxy layer has rank 0.")
            else:
                logger.warning(f"Dimension mismatch for layer {name}. target: ({target_layer.in_features}, {target_layer.out_features}), proxy: ({proxy_layer.in_features}, {proxy_layer.out_features})")
        else:
            logger.warning(f"No corresponding LinearSVD layer found in proxy for target layer: {name}")

    logger.info(f"Found {len(pairs)} corresponding layer pairs for noise injection.")
    return pairs

def _sdpa_math_ctx():
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend  # PyTorch ≥2.4+
        return sdpa_kernel(SDPBackend.MATH)
    except Exception:
        # PyTorch 2.0–2.3
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        )

def train_with_gradient_alignment(
    target_model: nn.Module,
    proxy_model: nn.Module,
    layer_pairs: List[Tuple[nn.Module, nn.Module]],  # (target_layer, proxy_LinearSVD)
    train_dataloader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    save_path: str,
    lambda_anchor: float = 0.1,  # KD weight
    gradient_accumulation_steps: int = 1,
    max_steps: int = -1,
    log_interval: int = 100,
    temperature: float = 2.0,
    logger: logging.Logger = logging.getLogger(__name__),
    eps_norm: float = 1e-12,           # numeric epsilon in norms
):
    os.makedirs(save_path, exist_ok=True)

    def _supervised_loss(model_outputs, batch):
        if hasattr(model_outputs, "loss") and model_outputs.loss is not None:
            return model_outputs.loss, model_outputs.logits
        if "labels" not in batch:
            raise RuntimeError("Batch must include 'labels' if model does not return .loss")
        logits = model_outputs.logits
        labels = batch["labels"]
        vocab = logits.size(-1)
        loss = F.cross_entropy(
            logits.view(-1, vocab),
            labels.view(-1),
            ignore_index=-100
        )
        return loss, logits

    # KD loss
    kl_loss_fn = nn.KLDivLoss(reduction='none', log_target=True)
    try:
        proxy_model.set_attn_implementation("eager")
    except Exception:
        if hasattr(proxy_model, "config"):
            proxy_model.config.attn_implementation = "eager"

    steps_per_epoch = max(1, len(train_dataloader) // max(1, gradient_accumulation_steps))
    total_steps = steps_per_epoch * epochs

    logger.info(f"[Stage 2] Starting Gradient Alignment for {epochs} epochs (ALL layers).")
    for p in target_model.parameters():
        p.requires_grad_(False)
    for t_layer, _ in layer_pairs:
        if hasattr(t_layer, "linear") and hasattr(t_layer.linear, "weight"):
            t_layer.linear.weight.requires_grad_(True)
        elif hasattr(t_layer, "weight"):
            t_layer.weight.requires_grad_(True)

    target_model.eval()
    proxy_model.train()

    completed_steps = 0
    progress_bar = tqdm(range(total_steps), desc="Stage 2")

    epoch_total_loss = epoch_align_loss = epoch_kd_loss = 0.0
    batches_cnt = 0

    t_weights = []
    for t_layer, _ in layer_pairs:
        t_weights.append(t_layer.weight)

    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            if max_steps > 0 and completed_steps >= max_steps:
                break

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            try:
                target_outputs = target_model(**batch)
                with _sdpa_math_ctx():
                    proxy_outputs = proxy_model(**batch)

                l_target, t_logits = _supervised_loss(target_outputs, batch)
                l_surr,   p_logits = _supervised_loss(proxy_outputs, batch)

                # ===== target mode gradients =====
                t_grads = autograd_grad(
                    l_target, t_weights,
                    retain_graph=False, create_graph=False, allow_unused=True
                )
                t_grads = [(g if g is not None else torch.zeros_like(p)) for g, p in zip(t_grads, t_weights)]

                # ===== Collect proxy A,B params from each LinearSVD =====
                p_A_params, p_B_params = [], []
                for _, p_layer in layer_pairs:  # layer_pairs: List[(target_linear, proxy_LinearSVD)]
                    A, B = p_layer.get_factors()          # -> nn.Parameter, nn.Parameter
                    p_A_params.append(A)
                    p_B_params.append(B)

                # ===== proxy grads wrt A,B (enable higher-order so GA loss can backprop) =====
                p_params = p_A_params + p_B_params
                p_grads  = autograd_grad(
                    l_surr, p_params,
                    retain_graph=True, create_graph=True, allow_unused=True
                )

                # Split and zero-fill if a factor is frozen/unused
                nA = len(p_A_params)
                gA_p_list = list(p_grads[:nA])
                gB_p_list = list(p_grads[nA:])
                for i in range(nA):
                    if gA_p_list[i] is None:
                        gA_p_list[i] = torch.zeros_like(p_A_params[i])
                    if gB_p_list[i] is None:
                        gB_p_list[i] = torch.zeros_like(p_B_params[i])

                # ===== Factor-space GA: project target and align with proxy grads =====
                eps_norm   = 1e-8  # numerical stability for cosine
                master_dev = p_logits.device
                l_align_sum, layer_used = torch.zeros((), device=master_dev), 0

                for (t_lin, p_layer), Gt, A, B, gA_s, gB_s in zip(
                    layer_pairs, t_grads, p_A_params, p_B_params, gA_p_list, gB_p_list
                ):  
                    # target targets in factor space (detach so target graph isn’t touched)
                    # Shapes: Gt [m,n], A [m,r], B [r,n]
                    gA_t = (Gt.detach() @ B.transpose(0, 1))      # [m, r]
                    gB_t = (A.transpose(0, 1) @ Gt.detach())      # [r, n]

                    # Cosine losses (simple, scale-robust-ish).
                    loss_A = 1.0 - F.cosine_similarity(gA_s.float().reshape(-1),
                                                    gA_t.float().reshape(-1),
                                                    dim=0, eps=eps_norm)
                    loss_B = 1.0 - F.cosine_similarity(gB_s.float().reshape(-1),
                                                    gB_t.float().reshape(-1),
                                                    dim=0, eps=eps_norm)
                    loss_l = 0.5 * (loss_A + loss_B)

                    l_align_sum = l_align_sum + loss_l.to(master_dev, non_blocking=True)
                    layer_used += 1

                l_align = l_align_sum / max(layer_used, 1)

                # ===== KD =====
                with torch.cuda.amp.autocast(False):
                    t_logp = F.log_softmax(t_logits.float().detach() / temperature, dim=-1)
                    p_logp = F.log_softmax(p_logits.float() / temperature, dim=-1)
                loss_per_token = kl_loss_fn(p_logp, t_logp).sum(dim=-1)
                tok_mask = (batch["labels"] != -100).float() if "labels" in batch else batch["attention_mask"].float()
                valid = tok_mask.sum().clamp_min(1.0)
                l_kd = (loss_per_token * tok_mask).sum() / valid
                l_kd = l_kd * (temperature ** 2)

                # ===== Backward =====
                loss_total = l_align + lambda_anchor * l_kd
                (loss_total / gradient_accumulation_steps).backward()

                # ===== Logging =====
                epoch_total_loss += float(loss_total.detach())
                epoch_align_loss += float(l_align.detach())
                epoch_kd_loss    += float(l_kd.detach())
                batches_cnt      += 1

            except Exception as e:
                logger.error(f"[Stage 2] Epoch {epoch+1}, Step {step}, Train Error: {e}", exc_info=True)
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.zero_grad(set_to_none=True)
                continue

            # ===== Optimization =====
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
                progress_bar.update(1)

                if completed_steps % log_interval == 0 and batches_cnt > 0:
                    avg_total = epoch_total_loss / batches_cnt
                    avg_align = epoch_align_loss / batches_cnt
                    avg_kd    = epoch_kd_loss / batches_cnt
                    logger.info(f"[Stage 2] Step {completed_steps}/{total_steps}: "
                                f"Total {avg_total:.4f} | Align {avg_align:.4f} | KD {avg_kd:.4f}")

            if max_steps > 0 and completed_steps >= max_steps:
                break

        if batches_cnt > 0:
            logger.info(f"[Stage 2] End of Epoch {epoch+1}/{epochs} — "
                        f"Avg Total: {epoch_total_loss/batches_cnt:.4f} "
                        f"(Align {epoch_align_loss/batches_cnt:.4f}, KD {epoch_kd_loss/batches_cnt:.4f})")

    progress_bar.close()
    logger.info("--- Training Finished ---")
    final_ckpt = os.path.join(save_path, "final_pytorch_model.bin")
    save_proxy_model(proxy_model, final_ckpt)
    logger.info(f"Final proxy model saved to {final_ckpt}")