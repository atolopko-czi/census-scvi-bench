import logging

import cellxgene_census
import click
import lightning.pytorch as pl
import scvi
import somacore
import torch
import torch.distributed as dist
import torchdata
from cellxgene_census.experimental.ml import ExperimentDataPipe
from lightning.pytorch.callbacks import DeviceStatsMonitor
from scvi import REGISTRY_KEYS
from scvi.model import SCVI
from torch.utils.data import DataLoader


from cellxgene_census.experimental.ml.pytorch import pytorch_logger

scvi.settings.seed = 0
N_GENES = 60664

logger = logging.getLogger("census_scvi")
logger.setLevel(logging.INFO)


class CensusDataLoader(DataLoader):
    
    def __init__(self, datapipe: ExperimentDataPipe, *args, **kwargs):
        super().__init__(datapipe, *args, **kwargs)
        pytorch_logger.info(f"pytorch dist rank={dist.get_rank()}")

    def __iter__(self):
        for tensors in super().__iter__():
            x, _ = tensors
            x = x.float()  # avoid "RuntimeError: mat1 and mat2 must have the same dtype", due to 32-bit vs 64-bit floats
            # print(x.shape)
            yield {
                REGISTRY_KEYS.X_KEY: x,
                REGISTRY_KEYS.BATCH_KEY: torch.zeros((x.shape[0], 1)),
                REGISTRY_KEYS.LABELS_KEY: None,
                REGISTRY_KEYS.CONT_COVS_KEY: None,
                REGISTRY_KEYS.CAT_COVS_KEY: None,
            }


class CensusSCVI(SCVI):

    def __init__(
        self,
        datapipe: torchdata.datapipes.iter.IterDataPipe,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        gene_likelihood: str = "zinb",
        latent_distribution: str = "normal",
        **model_kwargs,
    ):
        self.module = self._module_cls(
            N_GENES,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )
        self.datapipe = datapipe

        self.is_trained_ = False
        self._model_summary_string = ""
        self.train_indices_ = None
        self.test_indices_ = None
        self.validation_indices_ = None
        self.history_ = None

    def train(
        self,
        max_epochs: int = None,
        use_gpu: bool = None,
        accelerator: str = "auto",
        devices: int = "auto",
        plan_kwargs: dict = None,
        **trainer_kwargs,
    ):
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)
        datamodule = CensusDataModule(self.datapipe)

        print(trainer_kwargs)
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=datamodule,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs
        )
        return runner()
    

class CensusDataModule(pl.LightningDataModule):

    def __init__(self, datapipe):
        self.datapipe = datapipe
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        super().__init__()

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return CensusDataLoader(self.datapipe)
    
    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


@click.option("--census-uri", default=None, help="URI to census tiledb")
@click.option("--organism", default="homo_sapiens", help="Organism to use")
@click.option("--measurement-name", default="RNA")
@click.option("--layer-name", default="raw", help="Layer name to use")
@click.option("--obs-value-filter", default=None, type=str, help="Obs value filter to use")
@click.option("--torch-batch-size", default=128)
@click.option("--soma-buffer-bytes", type=int)
@click.option("--torch-devices", type=str, default=None)
@click.option("--max-epochs", default=1)
@click.command()
def main(census_uri,
         organism,
         measurement_name,
         layer_name,
         obs_value_filter,
         torch_batch_size,
         soma_buffer_bytes,
         torch_devices,
         max_epochs
         ) -> None:
    pytorch_logger.setLevel(logging.DEBUG)

    census = cellxgene_census.open_soma(uri=census_uri) if census_uri else cellxgene_census.open_soma()

    dp = ExperimentDataPipe(
        census["census_data"][organism],
        measurement_name=measurement_name,
        X_name=layer_name,
        obs_query=somacore.AxisQuery(value_filter=obs_value_filter),
        batch_size=int(torch_batch_size),
        soma_buffer_bytes=soma_buffer_bytes,
    )
    print(f"training data shape={dp.shape}")

    # for b, batch in enumerate(dp):
    #     if b % 1000 == 0:
    #         print(f"processed {b} batches")
    # sys.exit(0)
    
    shuffle_dp = dp # .shuffle()
    model = CensusSCVI(shuffle_dp)

    model.train(max_epochs=int(max_epochs), accelerator="gpu" if torch_devices else "cpu",
                devices=torch_devices if torch_devices else 1, strategy="ddp_find_unused_parameters_true",
                profiler="simple", callbacks=[DeviceStatsMonitor()],
                # for iterable datasets
                # see https://pytorch-lightning.readthedocs.io/en/1.7.7/guides/data.html#iterable-datasets and
                # https://lightning.ai/docs/pytorch/stable/common/trainer.html#val-check-interval
                # val_check_interval=100, check_val_every_n_epoch=None,
                )


if __name__ == "__main__":
    main()
