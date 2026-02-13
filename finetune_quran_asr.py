#!/usr/bin/env python3
"""
Fine-tune pretrained Arabic FastConformer on Quran dataset.
Based on: https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/speech_to_text_finetune.py
"""

import time
from typing import Union
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.trainer_utils import resolve_trainer_cfg


def get_base_model(trainer: pl.Trainer, cfg: DictConfig) -> ASRModel:
    """Load pretrained model for fine-tuning."""
    asr_model = None
    nemo_model_path = cfg.get('init_from_nemo_model', None)
    pretrained_name = cfg.get('init_from_pretrained_model', None)
    
    if nemo_model_path is not None and pretrained_name is not None:
        raise ValueError("Only pass `init_from_nemo_model` or `init_from_pretrained_model` but not both")
    elif nemo_model_path is None and pretrained_name is None:
        raise ValueError("Both `init_from_nemo_model` and `init_from_pretrained_model` cannot be None")
    elif nemo_model_path is not None:
        logging.info(f"Loading model from {nemo_model_path}")
        asr_model = ASRModel.restore_from(restore_path=nemo_model_path)
    elif pretrained_name is not None:
        logging.info(f"Loading pretrained model: {pretrained_name}")
        # Sync across ranks to avoid duplicate downloads
        num_ranks = trainer.num_devices * trainer.num_nodes
        if num_ranks > 1:
            if is_global_rank_zero():
                asr_model = ASRModel.from_pretrained(model_name=pretrained_name)
            else:
                wait_time = int(cfg.get('exp_manager', {}).get('seconds_to_sleep', 60))
                if wait_time < 60:
                    wait_time = 60
                logging.info(f"Waiting {wait_time}s for model download to finish...")
                time.sleep(wait_time)
        asr_model = ASRModel.from_pretrained(model_name=pretrained_name)
    
    asr_model.set_trainer(trainer)
    return asr_model


def setup_dataloaders(asr_model: ASRModel, cfg: DictConfig) -> ASRModel:
    """Setup training, validation, and test dataloaders."""
    from omegaconf import open_dict
    
    # Convert Hydra config to dict
    cfg_dict = model_utils.convert_model_config_to_dict_config(cfg)
    
    # Update model's internal config with valid values to avoid missing mandatory value errors
    # The model was loaded from a .nemo file which may have ??? placeholders
    with open_dict(asr_model.cfg):
        asr_model.cfg.train_ds = cfg_dict.model.train_ds
        asr_model.cfg.validation_ds = cfg_dict.model.validation_ds
        asr_model.cfg.test_ds = cfg_dict.model.test_ds
        # Also update tokenizer if it has ??? values
        if hasattr(asr_model.cfg, 'tokenizer') and asr_model.cfg.tokenizer:
            if asr_model.cfg.tokenizer.get('dir') is None or asr_model.cfg.tokenizer.get('dir') == '???':
                asr_model.cfg.tokenizer.dir = None  # Use model's built-in tokenizer
    
    logging.info(f"Updated model.cfg.train_ds: {OmegaConf.to_yaml(asr_model.cfg.train_ds)}")
    
    # Now setup dataloaders using the model's updated config
    asr_model.setup_training_data(asr_model.cfg.train_ds)
    asr_model.setup_multiple_validation_data(asr_model.cfg.validation_ds)
    if hasattr(asr_model.cfg, 'test_ds') and asr_model.cfg.test_ds.get('manifest_filepath') is not None:
        asr_model.setup_multiple_test_data(asr_model.cfg.test_ds)
    
    return asr_model


@hydra_runner(config_path="nemo_scripts", config_name="finetune_config")
def main(cfg: DictConfig):
    """Main fine-tuning function."""
    logging.info(f'Hydra config:\n{OmegaConf.to_yaml(cfg)}')
    
    # Setup trainer
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))
    
    # Load model
    asr_model = get_base_model(trainer, cfg)
    logging.info("Reusing vocabulary from pretrained model")
    
    # Setup data
    asr_model = setup_dataloaders(asr_model, cfg)
    
    # Setup optimizer
    asr_model.setup_optimization(cfg.model.optim)
    
    # Setup SpecAug if present
    if hasattr(cfg.model, 'spec_augment') and cfg.model.spec_augment is not None:
        asr_model.spec_augment = ASRModel.from_config_dict(cfg.model.spec_augment)
    
    # Train
    logging.info("Starting fine-tuning...")
    trainer.fit(asr_model)
    
    logging.info("✅ Fine-tuning complete!")


if __name__ == '__main__':
    main()
