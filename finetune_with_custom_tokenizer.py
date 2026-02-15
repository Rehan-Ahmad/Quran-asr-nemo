#!/usr/bin/env python3
"""
Fine-tune NeMo ASR model with custom tokenizer.
This replaces the pretrained tokenizer with one trained on Quran dataset.
"""

import os
import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf, open_dict
from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager


def load_model_with_custom_tokenizer(trainer, cfg):
    """
    Load pretrained model and REPLACE its tokenizer with custom one.
    
    This is critical for handling Quranic diacritics that the pretrained
    tokenizer doesn't support.
    """
    # Load pretrained model
    pretrained_path = cfg.init_from_nemo_model
    logging.info(f"Loading pretrained model from: {pretrained_path}")
    
    # Restore base model
    asr_model = ASRModel.restore_from(
        restore_path=pretrained_path,
        trainer=trainer,
        map_location=f'cuda:{trainer.global_rank}' if torch.cuda.is_available() else 'cpu',
    )
    
    logging.info(f"Model architecture: {type(asr_model).__name__}")
    
    # IMPORTANT: Replace tokenizer with custom one
    custom_tokenizer_dir = cfg.model.tokenizer.dir
    
    if custom_tokenizer_dir and os.path.exists(custom_tokenizer_dir):
        logging.info(f"\n{'='*70}")
        logging.info(f"REPLACING TOKENIZER WITH CUSTOM ONE")
        logging.info(f"{'='*70}")
        logging.info(f"Custom tokenizer directory: {custom_tokenizer_dir}")
        
        # Update model's tokenizer configuration
        with open_dict(asr_model.cfg):
            asr_model.cfg.tokenizer.dir = custom_tokenizer_dir
            asr_model.cfg.tokenizer.type = cfg.model.tokenizer.type
        
        # Reinitialize tokenizer and decoder with new vocabulary
        asr_model.change_vocabulary(
            new_tokenizer_dir=custom_tokenizer_dir,
            new_tokenizer_type=cfg.model.tokenizer.type,
        )
        
        logging.info(f"✓ Tokenizer replaced successfully")
        logging.info(f"  New vocabulary size: {asr_model.decoder.vocab_size}")
        logging.info(f"  Tokenizer type: {cfg.model.tokenizer.type}")
        logging.info(f"{'='*70}\n")
    else:
        logging.warning(f"Custom tokenizer not found at: {custom_tokenizer_dir}")
        logging.warning(f"Using pretrained tokenizer (may not handle all diacritics)")
    
    return asr_model


def setup_dataloaders(asr_model, cfg):
    """Configure data loaders with Hydra config."""
    logging.info("\nSetting up data loaders...")
    
    # Convert Hydra config to dict config
    config_dict = model_utils.convert_model_config_to_dict_config(cfg.model)
    
    # Update model's internal config with resolved values
    with open_dict(asr_model.cfg):
        # Update dataset configs
        if 'train_ds' in config_dict:
            asr_model.cfg.train_ds = config_dict.train_ds
        if 'validation_ds' in config_dict:
            asr_model.cfg.validation_ds = config_dict.validation_ds
        if 'test_ds' in config_dict:
            asr_model.cfg.test_ds = config_dict.test_ds

        # Update preprocessor settings if provided
        if 'preprocessor' in config_dict and 'normalize' in config_dict.preprocessor:
            if hasattr(asr_model.cfg, 'preprocessor'):
                asr_model.cfg.preprocessor.normalize = config_dict.preprocessor.normalize
        
        # Ensure tokenizer dir is set (even if None for built-in)
        if asr_model.cfg.tokenizer.dir == "???":
            asr_model.cfg.tokenizer.dir = cfg.model.tokenizer.dir
    
    # Setup model with new data configs
    asr_model.setup_training_data(asr_model.cfg.train_ds)
    asr_model.setup_validation_data(asr_model.cfg.validation_ds)
    
    if hasattr(asr_model.cfg, 'test_ds') and asr_model.cfg.test_ds.manifest_filepath:
        asr_model.setup_test_data(asr_model.cfg.test_ds)
    
    logging.info("✓ Data loaders configured")


def setup_optimizer(asr_model, cfg):
    """Configure optimizer and scheduler."""
    logging.info("\nSetting up optimizer...")
    
    # Update model's optimizer config
    with open_dict(asr_model.cfg):
        asr_model.cfg.optim = cfg.model.optim
    
    # Setup optimization
    asr_model.setup_optimization(optim_config=OmegaConf.to_container(cfg.model.optim, resolve=True))
    
    logging.info(f"✓ Optimizer: {cfg.model.optim.name}")
    logging.info(f"  Learning rate: {cfg.model.optim.lr}")
    logging.info(f"  Scheduler: {cfg.model.optim.sched.name}")


@hydra_runner(config_path="nemo_scripts", config_name="finetune_config_custom_tokenizer")
def main(cfg):
    """Main training function."""
    logging.info(f"\n{'='*70}")
    logging.info(f"Fine-tuning with Custom Tokenizer")
    logging.info(f"{'='*70}\n")
    
    # Validate paths
    if not cfg.init_from_nemo_model or not os.path.exists(cfg.init_from_nemo_model):
        raise FileNotFoundError(f"Pretrained model not found: {cfg.init_from_nemo_model}")
    
    if not os.path.exists(cfg.model.tokenizer.dir):
        raise FileNotFoundError(f"Custom tokenizer not found: {cfg.model.tokenizer.dir}")
    
    # Create trainer (disable built-in logger/checkpointing so exp_manager can configure them)
    with open_dict(cfg.trainer):
        cfg.trainer.logger = False
        cfg.trainer.enable_checkpointing = False
    trainer = pl.Trainer(**cfg.trainer)
    
    # Configure experiment manager (logging, checkpointing)
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Load model and replace tokenizer
    asr_model = load_model_with_custom_tokenizer(trainer, cfg)
    
    # Setup data loaders
    setup_dataloaders(asr_model, cfg)
    
    # Setup optimizer
    setup_optimizer(asr_model, cfg)
    
    # Start training
    logging.info(f"\n{'='*70}")
    logging.info(f"Starting Training")
    logging.info(f"{'='*70}\n")
    
    trainer.fit(asr_model)
    
    logging.info(f"\n{'='*70}")
    logging.info(f"Training Complete!")
    logging.info(f"{'='*70}\n")


if __name__ == '__main__':
    main()
