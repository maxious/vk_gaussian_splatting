#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import safetensors
import torch
import torch.utils.checkpoint
from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer

from model.feature_extractors import SpatialDino
from model.dit import DiT


class DummyT5Encoder(torch.nn.Module):
    """Returns zero embeddings for empty prompts - saves 5GB download."""

    def __init__(self, hidden_size=2048, device="cuda", dtype=torch.float16):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.zeros(
            batch_size, seq_len, self.hidden_size, device=self.device, dtype=self.dtype
        )
        return (hidden_states,)


def load_model(
    cfg, checkpoint_path, device="cuda:0", weight_dtype=torch.float16, use_dummy_t5=False
):
    if cfg.eval.scheduler == "DDPM":
        noise_scheduler = DDPMScheduler.from_pretrained(
            cfg.model.scheduler_url, subfolder="scheduler"
        )
    elif cfg.eval.scheduler == "DDIM":
        noise_scheduler = DDIMScheduler.from_pretrained(
            cfg.model.scheduler_url, subfolder="scheduler"
        )

    feature_extractor = SpatialDino(
        freeze_weights=True,
        model_type="dinov2_vitb14",
        num_patches_x=cfg.modalities.rgb.width,
        num_patches_y=cfg.modalities.rgb.width,
    )

    model_id = cfg.model.decoder_url
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=weight_dtype)
    tokenizer = BertTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = BertModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    tokenizer_2 = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer_2")

    if use_dummy_t5:
        print("Using DummyT5Encoder (empty prompts only) - skipping 5GB download")
        text_encoder_2 = DummyT5Encoder(hidden_size=2048, device=device, dtype=weight_dtype)
    else:
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=weight_dtype
        )

    cfg.used_modalities = {
        key: cfg.modalities[key]
        for key in ["rgb", "ray", "depth", "local_caption", "global_caption"]
    }
    model = DiT(modalities=cfg.used_modalities, **cfg.model)
    if os.path.splitext(checkpoint_path)[-1] == ".safetensors":
        state_dict = safetensors.torch.load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path)["module"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Loaded model from:", checkpoint_path)
    print("missing_keys", missing_keys)
    print("unexpected_keys", unexpected_keys)

    vae.to(device, dtype=weight_dtype)
    feature_extractor.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    text_encoder_2.to(device, dtype=weight_dtype)
    model.to(device, dtype=weight_dtype)

    models = {
        "model": model,
        "noise_scheduler": noise_scheduler,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "tokenizer_2": tokenizer_2,
        "text_encoder_2": text_encoder_2,
        "vae": vae,
        "feature_extractor": feature_extractor,
    }

    return models
