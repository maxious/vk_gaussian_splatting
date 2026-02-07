from typing import Optional, Union
import torch
from diffusers import AutoencoderKLLTXVideo
from diffusers.pipelines.ltx.pipeline_ltx_condition import retrieve_latents
from diffusers.utils import export_to_video, load_video
from PIL import Image
from OmnimatteZero import OmnimatteZero


def tensor_video_to_pil_images(video_tensor):
    """
    Converts a PyTorch tensor representing a video to a list of PIL Images.

    Args:
        video_tensor (torch.Tensor): A tensor of shape (1, frames, height, width, 3).
                                     Corresponds to batch size, frames, height, width, and RGB channels.

    Returns:
        List[Image.Image]: List of frames as PIL Images.
    """
    # Remove the batch dimension (shape: (frames, height, width, 3))
    video_tensor = video_tensor.squeeze(0)

    # Ensure the tensor is on CPU and convert to NumPy
    video_numpy = video_tensor.cpu().numpy()

    # Convert each frame to a PIL Image
    pil_images = [Image.fromarray(frame.astype("uint8")) for frame in video_numpy]

    return pil_images


class MyAutoencoderKLLTXVideo(AutoencoderKLLTXVideo):
    def forward_encode(
        self, sample, temb=None, sample_posterior=False, return_dict=True, generator=None
    ):
        posterior = self.encode(sample).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        if not return_dict:
            return (z,)
        return z

    def decode_latents(self, latents, temb=None):
        return self.decode(latents, temb).sample
