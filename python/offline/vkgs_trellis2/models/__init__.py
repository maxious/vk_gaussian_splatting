import importlib

__attributes = {
    "SparseStructureEncoder": "sparse_structure_vae",
    "SparseStructureDecoder": "sparse_structure_vae",
    "SparseStructureFlowModel": "sparse_structure_flow",
    "SLatFlowModel": "structured_latent_flow",
    "ElasticSLatFlowModel": "structured_latent_flow",
    "SparseUnetVaeEncoder": "sc_vaes.sparse_unet_vae",
    "SparseUnetVaeDecoder": "sc_vaes.sparse_unet_vae",
    "FlexiDualGridVaeEncoder": "sc_vaes.fdg_vae",
    "FlexiDualGridVaeDecoder": "sc_vaes.fdg_vae",
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, base_repo: str | None = None, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to checkpoint. Can be either local path or a Hugging Face model name.
               For HuggingFace: "org/repo_name" or "org/repo_name/subpath/model_name"
               If path starts with "ckpts/", it resolves to base_repo/path
        base_repo: Base HuggingFace repo for resolving relative paths (e.g., "microsoft/TRELLIS.2-4B")
        **kwargs: Additional arguments for model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file

    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        repo_id: str
        model_name: str | None

        if path.startswith("ckpts/"):
            if not base_repo:
                raise ValueError(
                    f"base_repo must be provided when path starts with 'ckpts/': {path}"
                )
            repo_id = base_repo
            model_name = path
        elif "/" in path:
            path_parts = path.split("/")
            if len(path_parts) >= 2:
                repo_id = "/".join(path_parts[:2])
                model_name = "/".join(path_parts[2:]) if len(path_parts) > 2 else None
            else:
                raise ValueError(f"Invalid path format: {path}")
        else:
            repo_id = path
            model_name = None

        if model_name:
            config_file = hf_hub_download(repo_id, f"{model_name}.json")
            model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")
        else:
            config_file = hf_hub_download(repo_id, "pipeline.json")
            model_file = hf_hub_download(repo_id, "model.safetensors")

    with open(config_file, "r") as f:
        config = json.load(f)
    model = __getattr__(config["name"])(**config["args"], **kwargs)
    model.load_state_dict(load_file(model_file), strict=False)

    return model


# For Pylance
if __name__ == "__main__":
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel

    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
