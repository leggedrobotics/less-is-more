"""Callback to save model weights as SafeTensors after checkpoint save."""

from pathlib import Path
from typing import Any
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from safetensors.torch import save_file
from limo.src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SaveSafetensorsCallback(Callback):
    """Save model weights as SafeTensors whenever a checkpoint is saved."""

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        """Called when saving a checkpoint."""
        if not isinstance(trainer.checkpoint_callback, ModelCheckpoint):
            return

        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback.last_model_path:
            ckpt_path = Path(ckpt_callback.last_model_path)

            # Save to weights directory
            weights_dir = ckpt_path.parent.parent / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            safetensors_path = weights_dir / f"{ckpt_path.stem}.safetensors"

            state_dict = pl_module.net.state_dict()
            save_file(state_dict, str(safetensors_path))

            trainer.strategy.barrier()
            if trainer.is_global_zero:
                log.info(f"Saved SafeTensors weights to: {safetensors_path}")
