"""Download InfiniDepth model checkpoints."""

import os
import sys

# Disable xet transfer for compatibility
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Setup logging to file
log_file = open("download_log.txt", "w")


def log(msg):
    print(msg, flush=True)
    log_file.write(msg + "\n")
    log_file.flush()


log("Starting InfiniDepth model download...")

INFINIDEPTH_REPO = "ritianyu/InfiniDepth"
CHECKPOINTS = [
    ("infinidepth.ckpt", "Main depth model"),
    ("infinidepth_gs.ckpt", "Gaussian splatting predictor"),
    ("moge2.pt", "MoGe-2 metric depth"),
    ("skyseg.onnx", "Sky segmentation"),
]

try:
    import huggingface_hub

    log(f"huggingface_hub version: {huggingface_hub.__version__}")

    # Enable verbose logging
    huggingface_hub.logging.set_verbosity_debug()

    from huggingface_hub import hf_hub_download

    for filename, description in CHECKPOINTS:
        log(f"Downloading {description} ({filename})...")
        try:
            path = hf_hub_download(
                repo_id=INFINIDEPTH_REPO,
                filename=filename,
                resume_download=True,
                force_download=False,
            )
            log(f"SUCCESS: {path}")
        except Exception as e:
            log(f"WARNING: Failed to download {filename}: {e}")
            if "skyseg" not in filename:
                raise

    log("SUCCESS! All InfiniDepth checkpoints downloaded.")

except Exception as e:
    log(f"ERROR: {e}")
    import traceback

    log(traceback.format_exc())
    sys.exit(1)
finally:
    log_file.close()
