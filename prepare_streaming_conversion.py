#!/usr/bin/env python3
"""
Prepare a streaming-friendly fine-tuning config by comparing non-streaming settings
with cache-aware streaming defaults, then writing a new config file.

This script does NOT train. It only prepares config and prints next steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf

BASE_CONFIG = Path("nemo_scripts/finetune_config_custom_tokenizer.yaml")
OUT_CONFIG = Path("nemo_scripts/finetune_config_streaming_custom_tokenizer.yaml")

# Streaming-related overrides derived from NeMo cache-aware streaming configs
STREAMING_OVERRIDES: Dict[str, Any] = {
    "model": {
        # Streaming config expects no global normalization for mel features
        "preprocessor": {
            "normalize": "NA",
        },
        "encoder": {
            # Causal downsampling and chunked attention are required for streaming
            "causal_downsampling": True,
            "att_context_style": "chunked_limited",
            "att_context_size": [140, 27],
            "conv_context_size": "causal",
        },
        # Disable eval loss to avoid long eval OOM in streaming configs
        "compute_eval_loss": False,
    },
}


def get_nested(cfg: Any, keys: List[str]) -> Any:
    node = cfg
    for key in keys:
        if not OmegaConf.is_config(node) or key not in node:
            return "<not set>"
        node = node[key]
    return node


def compare_settings(base_cfg: Any) -> None:
    print("\nStreaming parameter comparison (reference vs current):")
    print("-" * 70)
    for path, target_value in flatten_dict(STREAMING_OVERRIDES).items():
        keys = path.split(".")
        current_value = get_nested(base_cfg, keys)
        print(f"{path}")
        print(f"  current : {current_value}")
        print(f"  target  : {target_value}\n")


def flatten_dict(d: Dict[str, Any], parent: str = "") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for key, value in d.items():
        path = f"{parent}.{key}" if parent else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, path))
        else:
            items[path] = value
    return items


def main() -> None:
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"Base config not found: {BASE_CONFIG}")

    base_cfg = OmegaConf.load(BASE_CONFIG)

    compare_settings(base_cfg)

    # Merge overrides into base config
    streaming_cfg = OmegaConf.merge(base_cfg, STREAMING_OVERRIDES)

    OUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(streaming_cfg, OUT_CONFIG)

    print("\nPrepared streaming config:")
    print(f"  {OUT_CONFIG}")

    print("\nNext steps (streaming conversion):")
    print("1) Fine-tune with streaming overrides (recommended):")
    print("   python finetune_with_custom_tokenizer.py \\")
    print("     model.preprocessor.normalize=NA \\")
    print("     model.encoder.causal_downsampling=true \\")
    print("     model.encoder.att_context_style=chunked_limited \\")
    print("     model.encoder.att_context_size='[140,27]' \\")
    print("     model.encoder.conv_context_size=causal")
    print("\n2) Or use the generated config as reference:")
    print(f"   {OUT_CONFIG}")

    print("\n3) Evaluate streaming with cache-aware script:")
    print("   python speech_to_text_cache_aware_streaming_infer.py \\")
    print("     model_path=<your_streaming_model>.nemo \\")
    print("     dataset_manifest=data/manifests/test.json \\")
    print("     chunk_size=100 shift_size=50 left_chunks=2 online_normalization=true")


if __name__ == "__main__":
    main()
