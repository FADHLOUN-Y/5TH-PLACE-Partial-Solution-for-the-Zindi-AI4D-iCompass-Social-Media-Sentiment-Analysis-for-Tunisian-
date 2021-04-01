import hydra
import omegaconf
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def run_blending(cfg: omegaconf.DictConfig) -> None:


  logger.info(" .. Blending Will Be Starting in few seconds .. ")
  test_df = pd.read_csv(cfg.testing.test_csv)

  if cfg.classifiermode.num_classes == 1:
    preds_cols = ["predictions"]
    test_preds = np.zeros((len(test_df),1))
    num_classes = 2
  else:
    preds_cols = ["Negative","Neutral","Positive"]
    num_classes = 3
    test_preds = np.zeros((len(test_df),3))
  submissionsPath = f"./submissions/{num_classes}Labels/"
  for filename in os.listdir(submissionsPath):
    if filename.endswith(".csv"):
      submission = pd.read_csv(os.path.join(submissionsPath,filename))
      test_preds += submission[preds_cols].values

  
  test_preds /= len(os.listdir(submissionsPath))

  
  if test_preds.shape[1] == 3:
    test_df["label"] = np.argmax(test_preds,1)
    test_df["label"] = test_df["label"].apply(lambda x:x-1)
  else:
    test_df["label"] = (test_preds>=cfg.classifiermode.threshhold).astype(int)
    test_df["label"] = test_df["label"].apply(lambda x:-1 if x==0 else x)

  test_df[["ID","label"]].to_csv("./best_submission/FADHLOUNKLAI.csv",index=False)




if __name__ == "__main__":
  run_blending()


