import argparse
from pathlib import Path

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune NeMo ASR model on Quran dataset.")
    parser.add_argument("--config", default="configs/quran_finetune.yaml", help="Path to config file")
    parser.add_argument("--data_dir", default=None, help="Override data_dir in config")
    return parser.parse_args()


def load_config(config_path: str, data_dir: str | None):
    cfg = OmegaConf.load(config_path)
    if data_dir:
        cfg.data_dir = data_dir
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return OmegaConf.create(cfg)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.data_dir)

    from pytorch_lightning import Trainer, seed_everything
    from nemo.collections.asr.models import EncDecHybridRNNTCTCModel
    from nemo.utils.exp_manager import exp_manager

    seed_everything(cfg.seed)

    trainer = Trainer(**cfg.trainer)
    exp_manager(
        trainer,
        {
            "exp_dir": cfg.exp_dir,
            "name": cfg.experiment_name,
            "create_tensorboard_logger": True,
            "create_checkpoint_callback": True,
        },
    )

    model = EncDecHybridRNNTCTCModel.from_pretrained(model_name=cfg.model_name)

    if cfg.vocab_path and Path(cfg.vocab_path).exists():
        vocab = [line.strip() for line in Path(cfg.vocab_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if hasattr(model, "change_vocabulary"):
            model.change_vocabulary(vocab)

    train_ds_cfg = {
        "manifest_filepath": cfg.train_manifest,
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": True,
        "max_duration": cfg.data.max_duration,
    }
    val_ds_cfg = {
        "manifest_filepath": cfg.val_manifest,
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": True,
        "max_duration": cfg.data.max_duration,
    }
    test_ds_cfg = {
        "manifest_filepath": cfg.test_manifest,
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": True,
        "max_duration": cfg.data.max_duration,
    }

    model.setup_training_data(train_data_config=train_ds_cfg)
    model.setup_validation_data(val_data_config=val_ds_cfg)
    if Path(cfg.test_manifest).exists():
        model.setup_test_data(test_data_config=test_ds_cfg)

    model.setup_optimization(cfg.optim)

    trainer.fit(model)


if __name__ == "__main__":
    main()
