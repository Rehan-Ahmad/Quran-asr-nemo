#!/usr/bin/env python3
"""
Fine-tune multilingual FastConformer streaming model on Quran ASR data.
Uses official NeMo config with Quran-specific dataset paths.
"""

import argparse
import logging
from pathlib import Path
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.utils import model_utils
from nemo.core.config import hydra_runner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def finetune_model():
    """Fine-tune multilingual FastConformer model on Quran dataset."""
    
    # Paths
    project_root = Path(__file__).parent.parent
    model_checkpoint = project_root / "pretrained_models" / "stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
    config_file = project_root / "nemo_scripts" / "fastconformer_ctc_bpe_streaming_quran.yaml"
    train_manifest = project_root / "data" / "manifests" / "train.json"
    val_manifest = project_root / "data" / "manifests" / "val.json"
    tokenizer_dir = project_root / "data"
    
    # Validation
    for path, name in [(model_checkpoint, "Model checkpoint"),
                       (config_file, "Config file"),
                       (train_manifest, "Training manifest"),
                       (val_manifest, "Validation manifest")]:
        if not path.exists():
            logger.error(f"❌ {name} not found: {path}")
            exit(1)
    
    logger.info("="*60)
    logger.info("Fine-tune Multilingual FastConformer on Quran Data")
    logger.info("="*60)
    logger.info(f"Model:     {model_checkpoint.name}")
    logger.info(f"Config:    {config_file.name}")
    logger.info(f"Train:     {train_manifest}")
    logger.info(f"Val:       {val_manifest}")
    logger.info(f"Tokenizer: {tokenizer_dir}")
    logger.info("="*60)
    
    # Load config
    logger.info("\nLoading configuration...")
    config = OmegaConf.load(config_file)
    
    # Ensure all paths are absolute
    config.model.train_ds.manifest_filepath = str(train_manifest.resolve())
    config.model.validation_ds.manifest_filepath = str(val_manifest.resolve())
    config.model.tokenizer.dir = str(tokenizer_dir.resolve())
    config.exp_manager.exp_dir = str((project_root / "nemo_experiments").resolve())
    
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")
    
    # Initialize Lightning trainer with config
    logger.info("\nInitializing trainer...")
    trainer = pl.Trainer(**config.trainer)
    
    # Load model from checkpoint
    logger.info(f"\nLoading model from: {model_checkpoint}")
    model = EncDecHybridRNNTCTCBPEModel.restore_from(
        str(model_checkpoint),
        map_location="cpu"
    )
    
    # Update model config with new dataset paths
    logger.info("\nUpdating model configuration...")
    with open_dict(model.cfg):
        model.cfg.train_ds = config.model.train_ds
        model.cfg.validation_ds = config.model.validation_ds
        model.cfg.test_ds = config.model.test_ds
        model.cfg.tokenizer = config.model.tokenizer
        model.cfg.optim = config.model.optim
    
    # Update dataset configs
    model.setup_training_data(cfg=model.cfg.train_ds)
    model.setup_validation_data(cfg=model.cfg.validation_ds)
    
    logger.info("\n✓ Model loaded and configured")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Fine-tune
    logger.info("\n" + "="*60)
    logger.info("Starting fine-tuning...")
    logger.info("="*60)
    
    trainer.fit(model, ckpt_path=None)
    
    logger.info("\n" + "="*60)
    logger.info("Fine-tuning complete!")
    logger.info(f"Experiment: nemo_experiments/{config.name}")
    logger.info("View results: python launch_tensorboard.py")
    logger.info("Evaluate:     python evaluate_model.py")
    logger.info("="*60)


if __name__ == "__main__":
    finetune_model()
