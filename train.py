import logging
import pytorch_lightning as pl
import torch
import hydra
import omegaconf
from src.models import TuniziDialectClassifier
from src import utils

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def train_model(cfg: omegaconf.DictConfig) -> None:
    logger.info(f"Config: {omegaconf.OmegaConf.to_yaml(cfg)}")
    utils.setup_environment(random_seed=cfg.general.seed, gpu_list=cfg.general.gpu_list)
    tensorboard_logger = hydra.utils.instantiate(cfg.callbacks.tensorboard)
    model = TuniziDialectClassifier(hydra_config=cfg)
  
    trainer = pl.Trainer(
      max_epochs=cfg.training.max_epochs,
      min_epochs=cfg.training.min_epochs,
      logger=[tensorboard_logger],
      gpus=cfg.general.gpu_list,
      fast_dev_run=False,
      precision=32,
      progress_bar_refresh_rate=1,
      deterministic=True
    )
    logger.info(".. Shake your Hands Training Will Begin .. ")
    trainer.fit(model)



if __name__ == "__main__":
    train_model()

