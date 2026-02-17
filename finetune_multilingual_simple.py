#!/usr/bin/env python3
"""
Fine-tune English FastConformer on Quran ASR data with custom Quran tokenizer.
Follows the official NVIDIA NeMo example: examples/asr/speech_to_text_finetune.py

Hardware-aware: Auto-detects GPUs and scales training (works on 1-8 GPUs).
Environment variables: BATCH_SIZE_PER_GPU (default: 16), MAX_EPOCHS (default: 10)
"""

import os
import torch
torch.set_float32_matmul_precision('medium')

from pathlib import Path
from omegaconf import OmegaConf
import lightning.pytorch as pl
from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager

# ------ CONFIG ------
NEMO_MODEL = "/data/SAB_PhD/quranNemoASR/pretrained_models/stt_en_fastconformer_hybrid_large_streaming_multi.nemo"
TOKENIZER_DIR = Path("/data/SAB_PhD/quranNemoASR/tokenizer/quran_tokenizer_bpe_v1024")
MANIFEST_DIR = Path("/data/SAB_PhD/quranNemoASR/data/manifests")
LOG_DIR = Path("/data/SAB_PhD/quranNemoASR/nemo_experiments/FastConformer-English-Quran-Tokenizer")

# Hardware-aware configuration (override via environment variables if needed)
BATCH_SIZE_PER_GPU = int(os.getenv("BATCH_SIZE_PER_GPU", "16"))  # Per-GPU batch (16 for 24GB)
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "10"))
LEARNING_RATE = 1

def main():
    # Detect hardware configuration
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0:
        print("⚠️  No GPUs detected - running on CPU (very slow!)")
        effective_batch_size = BATCH_SIZE_PER_GPU
        num_gpus = 1  # For CPU mode
    else:
        effective_batch_size = BATCH_SIZE_PER_GPU * num_gpus
        print(f"✓ Detected {num_gpus} GPU(s)")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    
    print("=" * 80)
    print("FASTCONFORMER FINE-TUNING ON QURAN DATA WITH CUSTOM TOKENIZER")
    print("=" * 80)
    print(f"Hardware: {num_gpus} GPU(s), Batch: {BATCH_SIZE_PER_GPU}/GPU (effective: {effective_batch_size})")
    print("=" * 80)

    # [1] Load base model from NEMO checkpoint
    print("\n[1/7] Loading model from checkpoint...")
    model = ASRModel.restore_from(NEMO_MODEL)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded ({num_params / 1e6:.0f}M params)")

    # [2] Update tokenizer to use Quran tokenizer
    print("\n[2/7] Updating tokenizer to Quran BPE tokenizer...")
    try:
        model.change_vocabulary(new_tokenizer_dir=str(TOKENIZER_DIR), new_tokenizer_type="bpe")
        print(f"✓ Tokenizer updated: {TOKENIZER_DIR}")
        print(f"  Vocab size: {model.tokenizer.vocab_size}")
    except Exception as e:
        print(f"✗ Error updating tokenizer: {type(e).__name__}: {e}")
        raise

    # [3] Update configuration for Quran dataset
    print("\n[3/7] Updating configuration for Quran dataset...")
    
    # Update training dataset config
    model.cfg.train_ds.manifest_filepath = [str(MANIFEST_DIR / "train.json")]
    model.cfg.train_ds.is_tarred = False
    model.cfg.train_ds.tarred_audio_filepaths = None
    model.cfg.train_ds.batch_size = BATCH_SIZE_PER_GPU
    model.cfg.train_ds.num_workers = 8
    model.cfg.train_ds.pin_memory = True
    model.cfg.train_ds.max_duration = 30.0

    # Update validation dataset config (only fields that exist)
    model.cfg.validation_ds.manifest_filepath = [str(MANIFEST_DIR / "val.json")]
    model.cfg.validation_ds.batch_size = BATCH_SIZE_PER_GPU
    model.cfg.validation_ds.num_workers = 4

    print("✓ Dataset config updated")

    # [4] Setup data loaders using official NeMo method
    print("\n[4/7] Setting up data loaders...")
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    
    try:
        num_train_batches = len(model.train_dataloader())
        num_val_batches = len(model.val_dataloaders()[0]) if model.val_dataloaders() else 0
        print(f"✓ Train loader: {num_train_batches} batches")
        print(f"✓ Val loader: {num_val_batches} batches")
    except Exception as e:
        print(f"✓ Dataloaders configured (batch count: unavailable)")

    # [5] Setup optimizer
    print("\n[5/7] Setting up optimizer...")
    try:
        # Update learning rate in model config
        model.cfg.optim.lr = LEARNING_RATE
        model.setup_optimization(model.cfg.optim)
        print(f"✓ Optimizer configured (lr={LEARNING_RATE})")
    except Exception as e:
        print(f"✓ Optimizer setup (using default)")

    # [6] Setup PyTorch Lightning trainer
    strategy = 'ddp' if num_gpus > 1 else 'auto'
    print(f"\n[6/7] Setting up PyTorch Lightning trainer (strategy: {strategy})...")
    
    trainer_cfg = {
        'max_epochs': MAX_EPOCHS,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': num_gpus,
        'strategy': strategy,
        'num_nodes': 1,
        'log_every_n_steps': 10,
        'val_check_interval': 1.0,
        'enable_checkpointing': True,
        'gradient_clip_val': 1.0,
        'logger': False,  # exp_manager will create its own logger
    }
    print(f"✓ Configured for {num_gpus} device(s) with effective batch size {effective_batch_size}")
    
    print(f"Training strategy: {'DDP (multi-GPU)' if num_gpus > 1 else 'single device'}")
    print(f"Effective batch size: {effective_batch_size} ({BATCH_SIZE_PER_GPU} × {num_gpus})")
    
    trainer = pl.Trainer(**trainer_cfg)
    
    # Setup experiment manager for checkpointing and logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    exp_manager(trainer, {
        "exp_dir": str(LOG_DIR),
        "name": "finetune",
        "explicit_log_dir": str(LOG_DIR / "finetune"),
        "create_checkpoint_callback": False,  # Don't create - trainer already has one
        "create_tensorboard_logger": True,
    })
    
    model.set_trainer(trainer)
    print("✓ Trainer configured")

    # [7] Start training
    print("\n[7/7] Starting training...")
    print(f"Training for {MAX_EPOCHS} epochs")
    print("-" * 80)
    
    trainer.fit(model)
    
    print("-" * 80)
    print("\n✓ Training completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
    
    # Update training dataset config
    model.cfg.train_ds.manifest_filepath = [str(MANIFEST_DIR / "train.json")]
    model.cfg.train_ds.is_tarred = False
    model.cfg.train_ds.tarred_audio_filepaths = None
    model.cfg.train_ds.batch_size = BATCH_SIZE
    model.cfg.train_ds.num_workers = 4
    model.cfg.train_ds.pin_memory = True
    model.cfg.train_ds.max_duration = 20.0

    # Update validation dataset config (only fields that exist in this config)
    model.cfg.validation_ds.manifest_filepath = [str(MANIFEST_DIR / "val.json")]
    model.cfg.validation_ds.batch_size = BATCH_SIZE
    model.cfg.validation_ds.num_workers = 4

    print("✓ Dataset config updated")

    # [3] Setup data loaders using official NeMo method
    print("\n[3/6] Setting up data loaders...")
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)  # Use setup_validation_data (not setup_multiple_validation_data)
    
    try:
        num_train_batches = len(model.train_dataloader())
        num_val_batches = len(model.val_dataloaders()[0]) if model.val_dataloaders() else 0
        print(f"✓ Train loader: {num_train_batches} batches")
        print(f"✓ Val loader: {num_val_batches} batches")
    except Exception as e:
        print(f"✓ Dataloaders configured (batch count: unavailable)")

    # [4] Setup optimizer
    print("\n[4/6] Setting up optimizer...")
    try:
        model.setup_optimization(model.cfg.optim)
        print("✓ Optimizer configured")
    except Exception as e:
        print(f"✓ Optimizer setup (using default)")

    # [5] Setup PyTorch Lightning trainer
    print("\n[5/6] Setting up PyTorch Lightning trainer...")
    
    trainer_cfg = {
        'max_epochs': MAX_EPOCHS,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 1,
        'num_nodes': 1,
        'log_every_n_steps': 10,
        'val_check_interval': 1.0,
        'enable_checkpointing': True,
        'gradient_clip_val': 1.0,
        'logger': False,  # exp_manager will create its own logger
    }
    
    trainer = pl.Trainer(**trainer_cfg)
    
    # Setup experiment manager for checkpointing and logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    exp_manager(trainer, {
        "exp_dir": str(LOG_DIR),
        "name": "finetune",
        "explicit_log_dir": str(LOG_DIR / "finetune"),
        "create_checkpoint_callback": False,  # Don't create - trainer already has one
        "create_tensorboard_logger": True,
    })
    
    model.set_trainer(trainer)
    print("✓ Trainer configured")

    # [6] Start training
    print("\n[6/6] Starting training...")
    print(f"Training for {MAX_EPOCHS} epochs")
    print("-" * 80)
    
    trainer.fit(model)
    
    print("-" * 80)
    print("\n✓ Training completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
    print(f"✓ Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
