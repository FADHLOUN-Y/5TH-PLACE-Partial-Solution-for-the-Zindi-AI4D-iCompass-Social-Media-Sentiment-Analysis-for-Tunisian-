import hydra
import omegaconf
import torch
import glob
import os
from src import utils
import numpy as np
import pandas as pd
from src.models import TuniziDialectClassifier
from barbar import Bar
import gc
import sys
import logging
logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def run_inference(cfg: omegaconf.DictConfig) -> None:
  
  logger.info(" .. Testing Will Be Starting in few seconds .. ")

  test_df = pd.read_csv(cfg.testing.test_csv)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  checkpoints = glob.glob(
    os.path.join(
        cfg.general.logs_dir, "checkpoints",f"{cfg.model.architecture_name}{cfg.classifiermode.num_classes}","*.ckpt"
    )
  )
  num_models = len(checkpoints)

  if num_models == 0:
    sys.exit()
  if cfg.classifiermode.num_classes == 1:
    test_preds = np.zeros(len(test_df))
  else:
    test_preds = np.zeros((len(test_df),3))

  for checkpoint_id, checkpoint_path in enumerate(checkpoints):

    output_name = checkpoint_path.split("/")[2]
    seed = int(checkpoint_path.split("/")[3].split(".")[0].split("_")[1])
    utils.setup_environment(random_seed=seed, gpu_list=cfg.general.gpu_list)
    model = TuniziDialectClassifier.load_from_checkpoint(
      checkpoint_path, hydra_config=cfg
    )
    model.eval().to(device)
    test_predictions = []
    with torch.no_grad():
      for batch_idx,batch in enumerate(Bar(model.test_dataloader())):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        input_ids = input_ids.to(device,dtype=torch.long)
        attention_mask = attention_mask.to(device,dtype=torch.long)
        outputs = model.forward(input_ids,attention_mask=attention_mask)

        if cfg.classifiermode.num_classes==1:
          outputs = torch.sigmoid(outputs).detach().cpu().numpy()
          test_predictions.append(outputs)
        else:
          outputs = torch.softmax(outputs,1).detach().cpu().numpy()
          test_predictions.append(outputs)

    test_predictions = np.concatenate(test_predictions,axis=0)
    if cfg.classifiermode.num_classes == 1:
      test_predictions = test_predictions.reshape(test_predictions.shape[0])
    gc.collect()
    torch.cuda.empty_cache()
    utils.create_submission(test_df,output_name+str(seed),test_predictions,cfg.classifiermode.num_classes)

if __name__ == "__main__":
    run_inference()


  
        