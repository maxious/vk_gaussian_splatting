import logging
import os
import sys
from pathlib import Path

import torch

# Add Any4D to path
ANY4D_ROOT = Path(__file__).parent / "any4d"
if str(ANY4D_ROOT) not in sys.path:
    sys.path.insert(0, str(ANY4D_ROOT))

# Now we can import any4d
try:
    import hydra
    from any4d.models import init_model
    from any4d.utils.image import load_images
    from any4d.utils.inference import loss_of_one_batch_multi_view
    from any4d.utils.misc import seed_everything
    from any4d.utils.moge_inference import load_moge_model

    ANY4D_AVAILABLE = True
except ImportError as e:
    ANY4D_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Define dummies to prevent import errors in consumers
    load_images = None
    loss_of_one_batch_multi_view = None
    load_moge_model = None
    init_model = None


logger = logging.getLogger(__name__)


def check_any4d_available():
    if not ANY4D_AVAILABLE:
        logger.error(f"Any4D not available: {IMPORT_ERROR}")
        logger.error("Please run: git clone https://github.com/Any-4D/Any4D python/offline/any4d")
        logger.error("And install dependencies: pip install hydra-core natsort rerun-sdk")
        return False
    return True


def init_hydra_config(config_path, overrides=None):
    """Initialize Hydra config pointing to Any4D configs."""
    # Hydra needs a relative path from the calling script or a module path.
    # Since we are hacking the path, we can try using the absolute path with search_path_dir

    # Reset hydra instance
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).split(".")[0]

    # We initialize with the directory of the config file
    hydra.initialize_config_dir(config_dir=os.path.abspath(config_dir), version_base=None)

    if overrides is not None:
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    else:
        cfg = hydra.compose(config_name=config_name)

    return cfg


def load_any4d_model(checkpoint_path=None, device="cuda"):
    """Load Any4D model."""
    if not check_any4d_available():
        raise ImportError("Any4D not found")

    if checkpoint_path is None:
        # Default path
        checkpoint_path = Path("checkpoints/any4d_4v_combined.pth")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Config setup
    config_dir = ANY4D_ROOT / "configs"
    train_config_path = config_dir / "train.yaml"

    overrides = [
        "machine=local",
        "model=any4d",
        "model.encoder.uses_torch_hub=false",
        "model/task=images_only",
    ]

    cfg = init_hydra_config(str(train_config_path), overrides=overrides)

    # Initialize model
    model = init_model(cfg.model.model_str, cfg.model.model_config)
    model.to(device)

    # Load weights
    logger.info(f"Loading weights from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    return model


def run_inference(model, image_paths, resolution=(518, 336), device="cuda"):
    """Run Any4D inference on a list of images."""
    # Load MoGe model for depth (Any4D dependency)
    moge_model = load_moge_model(device=device)

    # Load images as views
    # load_images expects a list of paths
    # It returns a list of dictionaries (one per view)
    views = load_images(
        image_paths,
        size=resolution,
        verbose=False,
        norm_type="dinov2",
        patch_size=14,
        compute_moge_mask=True,
        moge_model=moge_model,
        binary_mask_path=None,
    )

    # Move views to device
    # load_images might return tensors on CPU
    # loss_of_one_batch_multi_view handles device movement?
    # Let's check Any4D code or just trust sample_inference from demo script

    # Run inference
    with torch.no_grad():
        results = loss_of_one_batch_multi_view(
            views,
            model,
            None,
            device,
            use_amp=True,
        )

    return results, views
