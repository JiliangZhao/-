import torch
from torch.utils.data import DataLoader
from layout_data.models.unet import UNetV2
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule
from layout_data.data.layout import LayoutDataset
import layout_data.utils.np_transforms as transforms
import torch.nn.functional as F
import scipy.io as sio
from layout_data.utils.visualize import visualize_heatmap
from layout_data.models.fcn import FCNAlex8s
import os

class UnetSL(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._build_model()
        xy_data = sio.loadmat('../example/xy.mat')
        self.xs, self.ys = xy_data['xs'], xy_data['ys']

    def _build_model(self):
        self.model = UNetV2(in_channels=1, num_classes=1, bn=False, multi_scale=False)

    def forward(self, x):
        y = self.model(x)
        return y

    def __dataloader(self, dataset, batch_size, shuffle=True):
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [scheduler]

    def prepare_data(self):
        """Prepare dataset
        """
        size: int = self.hparams.input_size
        transform_layout = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                torch.tensor([self.hparams.mean_layout]),
                torch.tensor([self.hparams.std_layout]),
            ),
        ])
        transform_heat = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.train_dir, list_path=self.hparams.train_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )
        val_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.val_dir, list_path=self.hparams.val_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )
        test_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.test_dir, list_path=self.hparams.test_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )

        print(
            f"Prepared dataset, train:{len(train_dataset)},\
                val:{len(val_dataset)}, test:{len(test_dataset)}"
        )

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.__dataloader(self.val_dataset, batch_size=16, shuffle=False)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, batch_size=1, shuffle=False)

    def training_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pre = self(layout)

        loss = F.l1_loss(heat_pre, heat - 298.0)

        self.log('loss', loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pre = self(layout)
        heat_pred_k = heat_pre + 298

        val_mae = F.l1_loss(heat_pred_k, heat)

        if batch_idx == 0:
            N, _, _, _ = heat.shape
            heat_list, heat_pre_list, heat_err_list = [], [], []
            for heat_idx in range(5):
                heat_list.append(heat[heat_idx, :, :, :].squeeze().cpu().numpy())
                heat_pre_list.append(heat_pred_k[heat_idx, :, :, :].squeeze().cpu().numpy())
            visualize_heatmap(self.xs, self.ys, heat_list, heat_pre_list, self.current_epoch)

        return {"val_mae": val_mae}

    def validation_epoch_end(self, outputs):
        val_mae_mean = torch.stack([x["val_mae"] for x in outputs]).mean()

        self.log('val_mae_mean', val_mae_mean)

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass


class FCNSL(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._build_model()
        xy_data = sio.loadmat('../example/xy.mat')
        self.xs, self.ys = xy_data['xs'], xy_data['ys']

    def _build_model(self):
        self.model = FCNAlex8s(num_classes=128, in_channels=1, bn=False)

    def forward(self, x):
        y = self.model(x)
        return y

    def __dataloader(self, dataset, batch_size, shuffle=True):
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.85)
        return [optimizer], [scheduler]

    def prepare_data(self):
        """Prepare dataset
        """
        size: int = self.hparams.input_size
        transform_layout = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                torch.tensor([self.hparams.mean_layout]),
                torch.tensor([self.hparams.std_layout]),
            ),
        ])
        transform_heat = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.train_dir, list_path=self.hparams.train_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )
        val_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.val_dir, list_path=self.hparams.val_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )
        test_dataset = LayoutDataset(
            self.hparams.data_root, subdir=self.hparams.test_dir, list_path=self.hparams.test_list,
            transform=transform_layout, target_transform=transform_heat,
            load_name=self.hparams.load_name, nx=self.hparams.nx,
        )

        print(
            f"Prepared dataset, train:{len(train_dataset)},\
                val:{len(val_dataset)}, test:{len(test_dataset)}"
        )

        # assign to use in dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return self.__dataloader(self.val_dataset, batch_size=16, shuffle=False)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, batch_size=1, shuffle=False)

    def training_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pre = self(layout)

        loss = F.l1_loss(heat_pre, heat - 298.0)

        self.log('loss', loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        layout, heat = batch
        heat_pre = self(layout)
        heat_pred_k = heat_pre + 298

        val_mae = F.l1_loss(heat_pred_k, heat)

        if batch_idx == 0:
            N, _, _, _ = heat.shape
            heat_list, heat_pre_list, heat_err_list = [], [], []
            for heat_idx in range(5):
                heat_list.append(heat[heat_idx, :, :, :].squeeze().cpu().numpy())
                heat_pre_list.append(heat_pred_k[heat_idx, :, :, :].squeeze().cpu().numpy())
            visualize_heatmap(self.xs, self.ys, heat_list, heat_pre_list, self.current_epoch)

        return {"val_mae": val_mae}

    def validation_epoch_end(self, outputs):
        val_mae_mean = torch.stack([x["val_mae"] for x in outputs]).mean()

        self.log('val_mae_mean', val_mae_mean)

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass

