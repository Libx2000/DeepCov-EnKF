import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from src.datamodules.h5dataset_oceanforecast import H5Dataset

def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    clim = torch.stack([batch[i][2] for i in range(len(batch))])
    out_mask = torch.stack([batch[i][3] for i in range(len(batch))])
    lead_times = torch.stack([batch[i][4] for i in range(len(batch))])
    variables = batch[0][5]
    out_variables = batch[0][6]
    iter = batch[0][7]
    return (
        inp,
        out,
        clim,
        out_mask,
        lead_times,
        [v for v in variables],
        [v for v in out_variables],
        iter,
    )

class ForecastDataModule(LightningDataModule):
    def __init__(
            self,
            root_dir: str,
            scale_dir: str,
            start_idx: float,
            end_idx: float,
            variables: list,
            out_variables: list,
            iter_num: int = 4,
            seed : int= 1024,
            batch_size: int = 64,
            num_workers: int = 0,
            shuffle: bool = True,
            pin_memory: bool = True,
            prefetch_factor: int = 2,
            noise: float = 0,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        if out_variables is None:
            self.hparams.out_variables = variables
        self.listers_clim = [os.path.join(self.hparams.root_dir, "climatology", f) 
                             for f in os.listdir(os.path.join(self.hparams.root_dir, "climatology")) 
                             if os.path.isfile(os.path.join(self.hparams.root_dir, "climatology", f))]
        self.listers_train = [os.path.join(self.hparams.root_dir, "train", f) 
                              for f in os.listdir(os.path.join(self.hparams.root_dir, "train")) 
                              if os.path.isfile(os.path.join(self.hparams.root_dir, "train", f))]
        self.listers_val = [os.path.join(self.hparams.root_dir, "val", f) 
                            for f in os.listdir(os.path.join(self.hparams.root_dir, "val")) 
                            if os.path.isfile(os.path.join(self.hparams.root_dir, "val", f))]
        self.listers_test = [os.path.join(self.hparams.root_dir, "test", f) 
                             for f in os.listdir(os.path.join(self.hparams.root_dir, "test")) 
                             if os.path.isfile(os.path.join(self.hparams.root_dir, "test", f))]

        self.train_data, self.val_data, self.test_data = None, None, None

        self.transforms = self.get_normalize(self.hparams.variables)
        self.output_transforms = self.get_normalize(self.hparams.out_variables)

    def get_normalize(self, variables: Optional[Dict] = None):
        if variables is None:
            variables = self.hparams.variables
        scale_dir = self.hparams.scale_dir
        normalize_mean = dict(np.load(os.path.join(scale_dir, "normalize_mean.npz")))
        mean = []
        for var in variables:
            if var != "total_precipitation":
                mean.append(normalize_mean[var])
            else:
                mean.append(np.array([0.0]))
        normalize_mean = np.concatenate(mean)
        normalize_std = dict(np.load(os.path.join(scale_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[var] for var in variables])
        data_transforms = transforms.Normalize(normalize_mean, normalize_std)
        return data_transforms

    def get_lat_lon(self):
        # assume different data sources have the same lat and lon coverage
        lat = np.load(os.path.join(self.hparams.scale_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.scale_dir, "lon.npy"))
        return lat, lon

    def _init_fn(self, worker_id):
        # 固定随机数
        np.random.seed(self.hparams.seed + worker_id)

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.train_data and not self.val_data and not self.test_data:
            self.train_data = H5Dataset(
                                root_dir=self.hparams.root_dir,
                                mode="train",
                                file_list=self.listers_train,
                                clim_list=self.listers_clim,
                                start_idx=self.hparams.start_idx,
                                end_idx=self.hparams.end_idx,
                                variables=self.hparams.variables,
                                out_variables=self.hparams.out_variables,
                                iter_num=self.hparams.iter_num,
                                transforms=self.transforms,
                                output_transforms=self.output_transforms,
                                noise=self.hparams.noise,
                            )
                        
            self.val_data = H5Dataset(
                                root_dir=self.hparams.root_dir,
                                mode="val",
                                file_list=self.listers_val,
                                clim_list=self.listers_clim,
                                start_idx=self.hparams.start_idx,
                                end_idx=self.hparams.end_idx,
                                variables=self.hparams.variables,
                                out_variables=self.hparams.out_variables,
                                iter_num=self.hparams.iter_num,
                                transforms=self.transforms,
                                output_transforms=self.output_transforms,
                                noise=False,
                            )               

            self.test_data = H5Dataset(
                                root_dir=self.hparams.root_dir,
                                mode="test",
                                file_list=self.listers_test,
                                clim_list=self.listers_clim,
                                start_idx=self.hparams.start_idx,
                                end_idx=self.hparams.end_idx,
                                variables=self.hparams.variables,
                                out_variables=self.hparams.out_variables,
                                iter_num=self.hparams.iter_num,
                                transforms=self.transforms,
                                output_transforms=self.output_transforms,
                                noise=False,
                            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=self._init_fn,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn = collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=self._init_fn,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn = collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=self._init_fn,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn = collate_fn
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass