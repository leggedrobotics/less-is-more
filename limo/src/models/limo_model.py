from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import wandb
import numpy as np
import yaml
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.regression import MeanAbsoluteError

from limo.src.utils.visualization import create_combined_visualization


class LimoModel(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        camera_info: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.loss = loss

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.val_loss_best = MinMetric()
        self.camera_info = self._load_camera_info(camera_info)

    def _load_camera_info(
        self, camera_info_path: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if camera_info_path is None:
            return None

        path = Path(camera_info_path)
        if not path.exists():
            self.print(
                f"camera_info file not found: {camera_info_path}. Falling back to image-only visualization."
            )
            return None

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_mae.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = batch["path"]
        preds = self(batch)
        loss = self.loss(preds, path)
        return loss, preds, path

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_mae(preds, targets)

        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss, "preds": preds}

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_mae(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds}

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        preds = self(batch)
        uuids = batch["uuid"]
        paths = preds.detach().cpu().numpy()
        return {"uuid": uuids, "path_pred": paths}

    def on_validation_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0 and wandb.run is not None:
            self.log_images_wandb(outputs, batch, split="val")

    def on_train_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if batch_idx == 0 and wandb.run is not None:
            self.log_images_wandb(outputs, batch, split="train")

    def on_validation_epoch_end(self) -> None:
        cur_val_loss = self.val_loss.compute()
        self.val_loss_best(cur_val_loss)
        self.log(
            "val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_mae(preds, targets)

        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        self.net.setup()
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = self.hparams.scheduler(
                optimizer=optimizer, total_steps=total_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def log_images_wandb(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        split: str,
        num_imgs: int = 6,
    ):
        """Log predicted paths as combined visualizations to wandb."""
        assert wandb.run is not None, "This can only be used with wandb active"

        # Get predictions, ground truth, and goals
        predicted_paths = outputs["preds"][:num_imgs].detach().cpu().numpy()
        ground_truth_paths = batch["path"][:num_imgs].cpu().numpy()

        # Get images
        images = batch["image_front"][:num_imgs]
        goals = batch["goal"][:num_imgs]

        def _to_np(t: "torch.Tensor") -> np.ndarray:
            return (t.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

        images = _to_np(images)
        images_left = (
            _to_np(batch["image_left"][:num_imgs]) if "image_left" in batch else None
        )
        images_right = (
            _to_np(batch["image_right"][:num_imgs]) if "image_right" in batch else None
        )

        goals = goals.cpu().numpy()

        visualizations = []
        for i in range(len(predicted_paths)):
            combined_img = create_combined_visualization(
                images[i],
                [ground_truth_paths[i], predicted_paths[i]],
                goals[i : i + 1],
                camera_info=self.camera_info,
                image_left=images_left[i] if images_left is not None else None,
                image_right=images_right[i] if images_right is not None else None,
            )
            visualizations.append(wandb.Image(combined_img))

        wandb.log({f"{split}/predictions": visualizations})
