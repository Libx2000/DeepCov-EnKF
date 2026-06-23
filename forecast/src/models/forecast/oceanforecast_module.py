from typing import Any, Dict, Tuple
import copy
import torch
import pickle
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, MaxMetric
from src.utils.weighted_acc_rmse import weighted_rmse_torch, weighted_acc_torch, weighted_mae_torch, activity_torch

class ForecastLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        after_scheduler: torch.optim.lr_scheduler,
        mean_path: str,
        std_path: str,
        ckpt_path: str,
        dict_vars: str,
        loss: object,
    ) -> None:
        """Initialize a `FourCastNetLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        if self.hparams.ckpt_path is not None:
            weights_dict = torch.load(self.hparams.ckpt_path)['state_dict']
            load_weights_dict = {k[4:]: v for k, v in weights_dict.items()
                                 if self.net.state_dict()[k[4:]].numel() == v.numel()}
            self.net.load_state_dict(load_weights_dict, strict=True)
        
        # loss function
        self.criterion = self.hparams.loss

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.dict_vars = dict_vars
        mult = np.ones(len(self.dict_vars))
        for i in range(len(self.dict_vars)):
            mult[i] = self.std[self.dict_vars[i]] * mult[i]
        self.mult = torch.tensor(mult, dtype=torch.float32, requires_grad=False)

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.val_rmse_sst_best = MinMetric()
        self.val_acc_sst_best = MaxMetric()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, lead_times: torch.Tensor, vars, out_vars) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x, mask, lead_times, vars, out_vars)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.val_rmse_sst_best.reset()
        self.val_acc_sst_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], phase: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y, clim, out_mask, lead_times, vars, out_vars, iter_num = batch
        loss = 0
        for iter in range(iter_num):
            if iter == 0:
                if iter_num > 1:
                    preds = self.forward(x.to(self.device), out_mask[:,iter], lead_times, vars, out_vars)
                else:
                    preds = self.forward(x.to(self.device), out_mask, lead_times, vars, out_vars)
                if (phase == 'val') or (phase == 'test'):
                    preds = preds.detach()
            else:
                atmos = y[:,iter-1,:len(vars)-len(out_vars)]
                preds = self.forward(torch.concat([atmos,preds], dim=1).to(self.device), out_mask[:,iter], lead_times, vars, out_vars)
                if (phase == 'val') or (phase == 'test'):
                    preds = preds.detach()
            if (iter_num > 1):
                loss += self.criterion(out_mask[:,iter] * preds, out_mask[:,iter] * y[:,iter,len(vars)-len(out_vars):], out_mask[:,iter])
            else:
                loss += self.criterion(out_mask * preds, out_mask * y, out_mask)
                
        torch.cuda.empty_cache()
        if iter_num > 1:
            return loss / iter_num, preds.detach(), y[:,-1,len(vars)-len(out_vars):], clim[:, -1], out_mask[:,-1]
        else:
            return loss, preds.detach(), y, clim, out_mask

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, clims, out_mask = self.model_step(batch, "train")

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, clims, out_mask = self.model_step(batch, "val")
        val_rmse = self.mult.to(self.device, dtype=preds.dtype) * weighted_rmse_torch(preds, targets, out_mask)
        val_rmse = val_rmse.detach()
        val_acc = weighted_acc_torch(out_mask * (preds - clims), out_mask * (targets - clims))
        val_acc = val_acc.detach()

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse_thetao_0.5", val_rmse[self.dict_vars.index('thetao_0.5')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_thetao_0.5", val_acc[self.dict_vars.index('thetao_0.5')], on_step=False, on_epoch=True, prog_bar=True)

        return {'rmse': val_rmse, 'acc': val_acc, 'preds': preds, 'targets': targets}

    def validation_epoch_end(self, validation_step_outputs) -> None:
        "Lightning hook that is called when a validation epoch ends."
        val_rmse, val_acc = 0, 0
        for out in validation_step_outputs:
            val_rmse += out['rmse'] / len(validation_step_outputs)
            val_acc += out['acc'] / len(validation_step_outputs)

        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss
        val_rmse_sst = val_rmse[self.dict_vars.index('thetao_0.5')]  # get current val rmse of v10
        self.val_rmse_sst_best(val_rmse_sst)  # update best so far val rmse of u10
        
        val_acc_sst = val_acc[self.dict_vars.index('thetao_0.5')]  # get current val acc of v10
        self.val_acc_sst_best(val_acc_sst)  # update best so far val acc of v10

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)
        self.log("val/rmse_sst_best", self.val_rmse_sst_best.compute(), prog_bar=True)
        self.log("val/acc_sst_best", self.val_acc_sst_best.compute(), prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        with torch.inference_mode(False):
            loss, preds, targets, clims, out_mask = self.model_step(batch, "test")
            test_rmse = self.mult.to(self.device, dtype=preds.dtype) * weighted_rmse_torch(preds, targets, out_mask)
            test_rmse = test_rmse.detach()
            test_acc = weighted_acc_torch(out_mask * (preds - clims), out_mask * (targets - clims))
            test_acc = test_acc.detach()

            # update and log metrics
            self.test_loss(loss)
            self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/rmse_thetao_0.5", test_rmse[self.dict_vars.index('thetao_0.5')], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test/acc_thetao_0.5", test_acc[self.dict_vars.index('thetao_0.5')], on_step=False, on_epoch=True, prog_bar=True)
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)
        optimizer = self.hparams.optimizer(params=self.parameters([
            {
                "params": decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0,
            }
        ]))
        if self.hparams.scheduler is not None:
            if self.hparams.after_scheduler is not None:
                after_scheduler = self.hparams.after_scheduler(optimizer=optimizer)
                scheduler = self.hparams.scheduler(optimizer=optimizer, after_scheduler=after_scheduler)
            else:
                scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = ForecastLitModule(None, None, None, None)
