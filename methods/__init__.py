import timm

from .glam import GLAM


def get_method(config):
    """Build the configured GLAM experiment."""
    method_name = config["active_method"]["name"]
    if method_name != "glam":
        raise ValueError(f"Unsupported method: {method_name}. This repository implements only 'glam'.")

    backbone_name = config["backbone"]
    if not backbone_name.startswith("vit"):
        raise ValueError(
            "GLAM currently supports timm Vision Transformer backbones because its "
            "LoRA adapters target the fused qkv projections."
        )

    base_model = timm.create_model(
        backbone_name,
        pretrained=True,
        num_classes=0,
    )
    return GLAM(base_model, config)
