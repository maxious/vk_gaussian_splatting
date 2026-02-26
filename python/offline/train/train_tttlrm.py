"""
Train tttLRM model (single-GPU MVP).

Combines Long-LRM's training loop with tttLRM's model architecture.
As noted by tttLRM authors: "Training can be easily done by combining
LongLRM and our inference code (the sequence parallel part)."

Usage:
    python -m offline.train.train_tttlrm \
        --config third_party/tttlrm/configs/dl3dv_full.yaml \
        --dataset /path/to/manifests_index.json \
        --output-dir ./tttlrm_checkpoints \
        --train-steps 10000

Run from the python/ directory so third_party modules are importable.
"""

import argparse
import copy
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

# ---------------------------------------------------------------------------
# sys.path setup — tttLRM modules import from "utils", "model", "data" etc.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_TTTLRM_ROOT = str(_SCRIPT_DIR.parent.parent / "third_party" / "tttlrm")
_LONGLRM_ROOT = str(_SCRIPT_DIR.parent.parent / "third_party" / "longlrm")

# ---------------------------------------------------------------------------
# Setup tttLRM imports — must happen before importing any tttLRM code.
# tttLRM has top-level "utils", "model", "data" packages that conflict with
# other packages in sys.path (e.g. sam_3d_body's "utils").  We force the
# tttLRM directory first and register the utils package explicitly.
# ---------------------------------------------------------------------------
import importlib
import importlib.util
import types as _types

# Put tttlrm first in sys.path
if _TTTLRM_ROOT in sys.path:
    sys.path.remove(_TTTLRM_ROOT)
sys.path.insert(0, _TTTLRM_ROOT)

# Evict any conflicting cached modules
for _prefix in ("utils", "model", "data"):
    for _mod_name in list(sys.modules.keys()):
        if _mod_name == _prefix or _mod_name.startswith(_prefix + "."):
            del sys.modules[_mod_name]
importlib.invalidate_caches()

# Pre-register tttLRM's utils package so sub-imports work
_utils_pkg = _types.ModuleType("utils")
_utils_pkg.__path__ = [str(Path(_TTTLRM_ROOT) / "utils")]
_utils_pkg.__package__ = "utils"
sys.modules["utils"] = _utils_pkg

for _submod, _filename in [
    ("utils.sp_support", "utils/sp_support.py"),
    ("utils.ddp_utils", "utils/ddp_utils.py"),
    ("utils.metrics", "utils/metrics.py"),
    ("utils.camera_utils", "utils/camera_utils.py"),
]:
    _filepath = Path(_TTTLRM_ROOT) / _filename
    if _filepath.exists() and _submod not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_submod, _filepath)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_submod] = _mod
        _spec.loader.exec_module(_mod)

from utils import sp_support as _sp  # noqa: E402


def _patch_sp_support() -> None:
    """Replace distributed SP functions with single-GPU no-ops."""
    _sp._SP_GROUP = None

    def _get_sp_world_size():
        return 1

    def _get_sp_rank():
        return 0

    def _get_sp_replicas():
        return 1

    def _get_sp_replica_id():
        return 0

    def _is_sp():
        return False

    def _sp_all_reduce(tensor, op=None):
        return tensor

    def _sp_all_gather(x, gather_dim=2, length=-1):
        if length != -1:
            x = _sp.slice_tensor(x, gather_dim, 0, length)
        return x

    def _sp_input_broadcast_scatter(x, scatter_dim=1, different_size=False):
        if different_size:
            return x, tuple(x.shape)
        return x

    def _sp_broadcast(x):
        return x

    def _sp_broadcast_different_size(x):
        return x, tuple(x.shape)

    def _sp_gather_scatter(x, gather_dim=2, scatter_dim=1):
        return x

    def _sp_local_scatter(x, scatter_dim=1):
        return x

    def _init_sp_group(sp_size):
        pass

    _sp.get_sp_world_size = _get_sp_world_size
    _sp.get_sp_rank = _get_sp_rank
    _sp.get_sp_replicas = _get_sp_replicas
    _sp.get_sp_replica_id = _get_sp_replica_id
    _sp.is_sp = _is_sp
    _sp.sp_all_reduce = _sp_all_reduce
    _sp.sp_all_gather = _sp_all_gather
    _sp.sp_input_broadcast_scatter = _sp_input_broadcast_scatter
    _sp.sp_broadcast = _sp_broadcast
    _sp.sp_broadcast_different_size = _sp_broadcast_different_size
    _sp.sp_gather_scatter = _sp_gather_scatter
    _sp.sp_local_scatter = _sp_local_scatter
    _sp.init_sp_group = _init_sp_group


_patch_sp_support()

# ---------------------------------------------------------------------------
# Now safe to import tttLRM model & dataset
# ---------------------------------------------------------------------------
import omegaconf  # noqa: E402
from easydict import EasyDict as edict  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import (  # noqa: E402
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from model.model import tttLRM  # noqa: E402
from data.dataset_scene import Dataset  # noqa: E402

logger = logging.getLogger("train_tttlrm")

# ---------------------------------------------------------------------------
# Helpers (ported from Long-LRM utils.py)
# ---------------------------------------------------------------------------


def create_optimizer(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
) -> torch.optim.AdamW:
    decay_params, nodecay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1 or name.endswith(".bias") or getattr(param, "_no_weight_decay", False):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


def create_scheduler(optimizer, total_steps: int, warmup_steps: int, scheduler_type: str = "cosine"):
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif scheduler_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(optimizer, warmup_steps)
    raise ValueError(f"Invalid scheduler type: {scheduler_type}")


def auto_resume_helper(checkpoint_dir: str) -> str | None:
    """Return the path to the most recent valid checkpoint, or None."""
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")],
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True,
    )
    for ckpt in checkpoints:
        path = os.path.join(checkpoint_dir, ckpt)
        try:
            torch.load(path, map_location="cpu", weights_only=False)
            return path
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def save_visualization(output_dir: str, step: int, data: dict, result, model: tttLRM) -> None:
    """Save input / rendered / target image grids and optionally a PLY."""
    vis_dir = os.path.join(output_dir, f"vis_{step:09d}")
    os.makedirs(vis_dir, exist_ok=True)

    try:
        # Input images grid  (B, V, C, H, W) → pick batch 0
        input_imgs = result.input.image[0]  # (V, C, H, W)
        grid_input = torchvision.utils.make_grid(input_imgs, nrow=min(8, input_imgs.size(0)), padding=2)
        torchvision.utils.save_image(grid_input, os.path.join(vis_dir, "input.png"))

        # Rendered vs target
        if result.render is not None and result.target is not None:
            rendered = result.render.render[0]  # (V, C, H, W)
            target = result.target.image[0]  # (V, C, H, W)
            n = min(rendered.size(0), 8)
            comparison = torch.cat([target[:n], rendered[:n]], dim=0)
            grid_cmp = torchvision.utils.make_grid(comparison, nrow=n, padding=2)
            torchvision.utils.save_image(grid_cmp, os.path.join(vis_dir, "target_vs_render.png"))

        # Save Gaussian PLY
        gaussians = result.gaussians
        gaussian_b0 = {k: v[0] for k, v in gaussians.items()}
        model.save_gaussian_ply(gaussian_b0, os.path.join(vis_dir, "gaussians.ply"))
    except Exception as e:
        logger.warning("Visualization save failed at step %d: %s", step, e)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def run_training(args: argparse.Namespace) -> None:
    # ---- Logging ----------------------------------------------------------
    log_fmt = "[%(asctime)s %(name)s] %(levelname)s %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    os.makedirs(args.output_dir, exist_ok=True)
    handlers.append(logging.FileHandler(os.path.join(args.output_dir, "train.log"), mode="a"))
    logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers)

    # ---- Config -----------------------------------------------------------
    cfg = omegaconf.OmegaConf.load(args.config)
    cfg = edict(omegaconf.OmegaConf.to_container(cfg, resolve=True))

    # Force single-GPU settings
    cfg.sp_size = 1
    cfg.training.dataset_path = args.dataset
    cfg.training.batch_size_per_gpu = args.batch_size

    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.num_input_views is not None:
        cfg.training.num_input_views = args.num_input_views
        # Keep virtual views == input views for simplicity (matches default full ttt_scan)
        cfg.training.num_virtual_views = args.num_input_views

    # Ensure required config keys have defaults
    cfg.training.setdefault("beta1", 0.9)
    cfg.training.setdefault("beta2", 0.95)
    cfg.training.setdefault("weight_decay", 0.05)
    cfg.training.setdefault("warmup", 500)
    cfg.training.setdefault("warmup_steps", cfg.training.warmup)
    cfg.training.setdefault("grad_clip_norm", 1.0)
    cfg.training.setdefault("grad_accum_steps", args.grad_accum)
    cfg.training.setdefault("print_every", 20)
    cfg.training.setdefault("vis_every", 250)
    cfg.training.setdefault("checkpoint_every", 500)
    cfg.training.setdefault("scheduler_type", "cosine")
    cfg.training.setdefault("prefetch_factor", 2)
    cfg.training.setdefault("frame_method", "mean_cam")
    cfg.training.setdefault("target_has_input", True)
    cfg.training.setdefault("l2_loss_weight", 1.0)
    cfg.training.setdefault("perceptual_loss_weight", 0.5)
    cfg.training.setdefault("opacity_loss_weight", 0.01)
    cfg.training.setdefault("depth_loss_weight", 0.0)
    cfg.setdefault("use_tf32", True)
    cfg.setdefault("use_amp", True)
    cfg.setdefault("amp_dtype", "bf16")

    grad_accum_steps = cfg.training.grad_accum_steps
    train_steps = args.train_steps  # param-update steps
    total_fwd_steps = train_steps * grad_accum_steps
    warmup_steps = int(cfg.training.warmup_steps)

    logger.info("Config:\n%s", cfg)
    logger.info(
        "train_steps=%d, grad_accum=%d, total_fwd=%d, warmup=%d, bs=%d, lr=%.2e",
        train_steps, grad_accum_steps, total_fwd_steps, warmup_steps,
        cfg.training.batch_size_per_gpu, cfg.training.lr,
    )

    # ---- Device & reproducibility ----------------------------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = cfg.use_tf32
    torch.backends.cudnn.allow_tf32 = cfg.use_tf32

    # ---- Dataset & DataLoader ---------------------------------------------
    dataset = Dataset(cfg)
    logger.info("Dataset size: %d scenes", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size_per_gpu,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=cfg.training.prefetch_factor if args.num_workers > 0 else None,
    )

    # ---- Model ------------------------------------------------------------
    model = tttLRM(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %.2fM total, %.2fM trainable", num_params / 1e6, num_trainable / 1e6)

    # ---- Optimizer / Scheduler / Scaler -----------------------------------
    optimizer = create_optimizer(
        model, cfg.training.weight_decay, cfg.training.lr,
        (cfg.training.beta1, cfg.training.beta2),
    )
    scheduler = create_scheduler(
        optimizer, train_steps, warmup_steps,
        cfg.training.get("scheduler_type", "cosine"),
    )
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(cfg.amp_dtype, torch.bfloat16)
    enable_scaler = cfg.use_amp and cfg.amp_dtype == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=enable_scaler)

    # ---- Resume -----------------------------------------------------------
    start_update_step = 0
    resume_path = args.resume or auto_resume_helper(args.output_dir)
    if resume_path is not None:
        logger.info("Resuming from %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        status = model.load_state_dict(ckpt["model"], strict=False)
        logger.info("Model load status: %s", status)
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                start_update_step = ckpt.get("param_update_steps_done", 0)
                logger.info("Restored optimizer & scheduler at step %d", start_update_step)
            except Exception:
                logger.warning("Could not restore optimizer/scheduler — starting fresh")
        del ckpt
        torch.cuda.empty_cache()
    else:
        logger.info("No checkpoint found — training from scratch")

    # ---- Wandb (optional) -------------------------------------------------
    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project="tttlrm-training",
                config=copy.deepcopy(dict(cfg)),
                resume="allow",
            )
            logger.info("Wandb initialised: %s", wandb_run.url)
        except Exception as e:
            logger.warning("Wandb init failed: %s — continuing without it", e)

    # ---- Collect trainable params for grad ops ----------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # ---- Training loop ----------------------------------------------------
    model.train()
    optimizer.zero_grad(set_to_none=True)

    fwd_step = start_update_step * grad_accum_steps
    update_step = start_update_step
    dataloader_iter = iter(dataloader)

    while update_step < train_steps:
        # Fetch batch (restart dataloader on exhaustion)
        try:
            data = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            data = next(dataloader_iter)

        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

        t0 = time.time()

        # Forward + loss
        with torch.autocast(device_type="cuda", enabled=cfg.use_amp, dtype=amp_dtype):
            result = model(data)
            loss = result.loss_metrics.loss / grad_accum_steps

        # Backward
        scaler.scale(loss).backward()
        fwd_step += 1

        is_update_step = (fwd_step % grad_accum_steps == 0)

        if is_update_step:
            update_step += 1

            scaler.unscale_(optimizer)

            # NaN-to-num on gradients
            with torch.no_grad():
                for p in trainable_params:
                    if p.grad is not None:
                        p.grad.nan_to_num_(nan=0.0, posinf=1e-3, neginf=-1e-3)

            total_grad_norm = 0.0
            grad_clip = cfg.training.grad_clip_norm
            if grad_clip > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip).item()

            skip = False
            if total_grad_norm > grad_clip * 5.0:
                logger.warning("Step %d: grad norm %.4f too large — skipping", update_step, total_grad_norm)
                skip = True

            if not skip:
                scaler.step(optimizer)
                scaler.update()
            else:
                scaler.update()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            iter_time = time.time() - t0

            # ---- Logging --------------------------------------------------
            if update_step % cfg.training.print_every == 0 or update_step <= start_update_step + 10:
                lm = result.loss_metrics
                psnr_val = lm.get("psnr_real", lm.get("psnr", torch.tensor(0.0)))
                if isinstance(psnr_val, torch.Tensor):
                    psnr_val = psnr_val.item()
                loss_val = lm.loss.item()
                lr_now = optimizer.param_groups[0]["lr"]
                mem_mb = torch.cuda.max_memory_allocated() / 1024**2
                logger.info(
                    "Step %d/%d | loss=%.4f psnr=%.2f lr=%.2e grad=%.3f time=%.2fs mem=%.0fMB",
                    update_step, train_steps, loss_val, psnr_val,
                    lr_now, total_grad_norm, iter_time, mem_mb,
                )
                if wandb_run is not None:
                    log_dict = {
                        "train/loss": loss_val,
                        "train/psnr": psnr_val,
                        "train/l2_loss": lm.l2_loss.item(),
                        "train/perceptual_loss": lm.perceptual_loss.item(),
                        "train/opacity_loss": lm.opacity_loss.item(),
                        "lr": lr_now,
                        "grad_norm": total_grad_norm,
                        "iter_time": iter_time,
                    }
                    wandb_run.log(log_dict, step=update_step)

            # ---- Checkpoint -----------------------------------------------
            if update_step % cfg.training.checkpoint_every == 0 or update_step == train_steps:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_{update_step:09d}.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "param_update_steps_done": update_step,
                    },
                    ckpt_path,
                )
                logger.info("Saved checkpoint: %s", ckpt_path)

            # ---- Visualisation --------------------------------------------
            if update_step % cfg.training.vis_every == 0:
                save_visualization(args.output_dir, update_step, data, result, model)
                logger.info("Saved visualisation at step %d", update_step)

    logger.info("Training complete — %d update steps done.", update_step)
    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tttLRM model (single-GPU MVP)")
    parser.add_argument("--config", type=str, required=True, help="Base tttLRM config YAML (e.g. dl3dv_full.yaml)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset index JSON")
    parser.add_argument("--output-dir", type=str, default="./tttlrm_checkpoints", help="Checkpoint output dir")
    parser.add_argument("--resume", type=str, default=None, help="Resume from specific checkpoint path")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--train-steps", type=int, default=10000, help="Total param-update steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num-input-views", type=int, default=None, help="Override num_input_views (also sets num_virtual_views)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda / cpu)")
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
