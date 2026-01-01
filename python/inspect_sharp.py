import torch
import torch
import numpy as np
import cv2
import ssl
from sharp.models import create_predictor, PredictorParams
from sharp.cli.predict import predict_image

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]


def inspect_sharp_output():
    print("Initializing SHARP...")
    params = PredictorParams()
    predictor = create_predictor(params)

    # Use cached load if possible, or download
    # url = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    # state_dict = torch.hub.load_state_dict_from_url(url, progress=True)

    # Use local path provided by user
    model_path = r"S:\ComfyUI\models\sharp\sharp_2572gikvuh.pt"
    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")

    predictor.load_state_dict(state_dict)
    predictor.eval().to("cuda")

    # Create dummy image
    H, W = 518, 518
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[:] = 128

    # Focal length (arbitrary for test)
    f_px = 500.0

    print("Running inference...")
    with torch.no_grad():
        gaussians = predict_image(predictor, image, f_px, device=torch.device("cuda"))

    print("Output Type:", type(gaussians))
    print("Attributes:", dir(gaussians))

    if hasattr(gaussians, "means"):
        print("Means shape:", gaussians.means.shape)  # type: ignore[attr-defined]
    if hasattr(gaussians, "colors"):
        print("Colors shape:", gaussians.colors.shape)  # type: ignore[attr-defined]
    if hasattr(gaussians, "scales"):
        print("Scales shape:", gaussians.scales.shape)  # type: ignore[attr-defined]
    if hasattr(gaussians, "rotations"):
        print("Rotations shape:", gaussians.rotations.shape)  # type: ignore[attr-defined]
    if hasattr(gaussians, "opacities"):
        print("Opacities shape:", gaussians.opacities.shape)  # type: ignore[attr-defined]

    # Check color space utils
    from sharp.utils import cs_utils  # type: ignore[attr-defined]

    print("cs_utils available:", dir(cs_utils))


if __name__ == "__main__":
    inspect_sharp_output()
