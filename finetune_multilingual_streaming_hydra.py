#!/usr/bin/env python3
"""
Fine-tune multilingual FastConformer on Quran ASR using Hydra/NeMo.
This is the recommended approach using NeMo's exp_manager and Hydra.
"""

import sys
from pathlib import Path
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.utils import model_utils
from omegaconf import OmegaConf, open_dict
import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager


def main():
    """Fine-tune model using Hydra + exp_manager."""
    
    project_root = Path(__file__).parent
    
    # Load config
    config = OmegaConf.load(project_root / "nemo_scripts" / "fastconformer_ctc_bpe_streaming_quran.yaml")
    
    # Resolve paths to absolute
    config.model.train_ds.manifest_filepath = str((project_root / "data" / "manifests" / "train.json").resolve())
    config.model.validation_ds.manifest_filepath = str((project_root / "data" / "manifests" / "val.json").resolve())
    config.model.test_ds.manifest_filepath = str((project_root / "data" / "manifests" / "test.json").resolve())
    config.model.tokenizer.dir = str((project_root / "data").resolve())
    config.exp_manager.exp_dir = str((project_root / "nemo_experiments").resolve())
    
    print("\n" + "="*70)
    print("Multilingual FastConformer Fine-tuning on Quran Data")
    print("="*70)
    print(f"Model: stt_en_fastconformer_hybrid_large_streaming_multi.nemo")
    print(f"Config: Official NeMo streaming (cache-aware)")
    print(f"Train manifest: {config.model.train_ds.manifest_filepath}")
    print(f"Val manifest: {config.model.validation_ds.manifest_filepath}")
    print(f"Tokenizer dir: {config.model.tokenizer.dir}")
    print(f"Exp dir: {config.exp_manager.exp_dir}")
    print("="*70 + "\n")
    
    # Initialize experiment manager and trainer
    exp_dir = exp_manager(config)
    
    # Create trainer
    trainer = pl.Trainer(**config.trainer)
    
    # Load pretrained model
    model_path = project_root / "pretrained_models" / "stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
    print(f"\nLoading model: {model_path}")
    model = EncDecHybridRNNTCTCBPEModel.restore_from(str(model_path), map_location="cpu")
    
    # Update config with new paths
    with open_dict(model.cfg):
        model.cfg.train_ds = config.model.train_ds
        model.cfg.validation_ds = config.model.validation_ds
        model.cfg.test_ds = config.model.test_ds
        model.cfg.tokenizer = config.model.tokenizer
        model.cfg.optim = config.model.optim
        model.cfg.exp_manager = config.exp_manager
    
    # Setup datasets
    model.setup_training_data(cfg=model.cfg.train_ds)
    model.setup_validation_data(cfg=model.cfg.validation_ds)
    
    print(f"\n✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
    print(f"✓ Training data: {config.model.train_ds.manifest_filepath}")
    print(f"✓ Validation data: {config.model.validation_ds.manifest_filepath}")
    
    # Fine-tune
    print("\n" + "="*70)
    print("Starting fine-tuning...")
    print("="*70 + "\n")
    
    trainer.fit(model)
    
    print("\n" + "="*70)
    print("Fine-tuning complete!")
    print(f"Experiment: {exp_dir}")
    print(f"View: tensorboard --logdir={exp_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
