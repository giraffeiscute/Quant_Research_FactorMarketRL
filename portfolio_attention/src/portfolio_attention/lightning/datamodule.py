"""Lightning DataModule wrapper for scenario training datasets."""

from __future__ import annotations

from typing import Callable

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig
from ..data.dataset import PortfolioPanelDataset


class LightningTrainDataModule(pl.LightningDataModule):
    """Thin DataModule wrapper around the existing scenario dataset stack."""

    def __init__(
        self,
        *,
        data_config: DataConfig,
        num_workers: int = 0,
        interrupt_checker: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.num_workers = int(num_workers)
        self.interrupt_checker = interrupt_checker

        self.dataset: PortfolioPanelDataset | None = None
        self.train_dataset: Dataset | None = None
        self.validation_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def _raise_if_interrupted(self) -> None:
        if self.interrupt_checker is None:
            return
        self.interrupt_checker()

    def build_datasets(self) -> None:
        self._raise_if_interrupted()
        if self.dataset is not None:
            return
        dataset = PortfolioPanelDataset(
            self.data_config,
            interrupt_checker=self.interrupt_checker,
        )
        train_dataset, validation_dataset, test_dataset = dataset.build_train_validation_test_datasets()
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self._raise_if_interrupted()

    def validate_validation_divisibility(self, world_size: int) -> None:
        self.build_datasets()
        if self.validation_dataset is None:
            raise RuntimeError("validation_dataset is unavailable before divisibility validation.")
        resolved_world_size = int(world_size)
        if resolved_world_size <= 0 or (len(self.validation_dataset) % resolved_world_size) != 0:
            raise ValueError(
                "validation dataset size must be divisible by world size to avoid duplicated validation scenarios under DistributedSampler"
            )

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "validate", "test"):
            return
        self.build_datasets()

    def _build_dataloader(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is unavailable before building the train DataLoader.")
        return self._build_dataloader(
            self.train_dataset,
            batch_size=self.data_config.train_batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.validation_dataset is None:
            raise RuntimeError(
                "validation_dataset is unavailable before building the validation DataLoader."
            )
        return self._build_dataloader(
            self.validation_dataset,
            batch_size=1,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is unavailable before building the test DataLoader.")
        return self._build_dataloader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
        )
