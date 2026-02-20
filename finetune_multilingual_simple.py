"""
Refactored finetune entrypoint that reuses upstream helper functions in
`nemo_scripts/speech_to_text_finetune.py` while still loading the local
Hydra-like YAML config `nemo_scripts/finetune_config_streaming_custom_tokenizer.yaml`.

This keeps behavior consistent with the upstream patterns (tokenizer handling,
trainer resolution, exp_manager) without forcing a config-directory refactor.
"""

from omegaconf import OmegaConf
import lightning.pytorch as pl

from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from nemo_scripts.speech_to_text_finetune import (
    get_base_model,
    check_vocabulary,
    setup_dataloaders,
)


def main():
    cfg_path = "nemo_scripts/finetune_config_streaming_custom_tokenizer.yaml"
    cfg = OmegaConf.load(cfg_path)

    # Resolve trainer and experiment manager using upstream utilities
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Build model, update tokenizer/vocab if requested, setup data and optimization
    asr_model = get_base_model(trainer, cfg)
    asr_model = check_vocabulary(asr_model, cfg)
    asr_model = setup_dataloaders(asr_model, cfg)
    asr_model.setup_optimization(cfg.model.optim)

    if hasattr(cfg.model, 'spec_augment') and cfg.model.spec_augment is not None:
        from nemo.collections.asr.models import ASRModel

        asr_model.spec_augment = ASRModel.from_config_dict(cfg.model.spec_augment)

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()
