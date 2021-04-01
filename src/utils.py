import torch
import numpy as np
import random
import os
import pytorch_lightning as pl
from typing import List
import pandas as pd
from typing import Tuple


def split_data(
  df:pd.DataFrame,
  seed:int,
  num_classes:int,
  valid_positive_text_samples:float,
  valid_negative_text_samples:float,
  valid_neutral_text_samples:float=0) -> Tuple[pd.DataFrame,pd.DataFrame]:

  """ Function That Splits The data into train and validation According to Given Ratios For each Class
    Args:
      df : Pandas DataFrame Containing 2 or 3 Labels 
      seed : a Random State
      num_classes : Number of Classes Available (2 or 3)
      valid_positive_text_samples: Ratio of Positive Samples that will be Multiplied by 5000 == validation shape
      valid_negative_text_samples: Ratio of Negative Samples that will be Multiplied by 5000 == validation shape
      valid_neutral_text_samples: Ratio of Neutral Samples that will be Multiplied by 5000 == validation shape
    Returns :
      - Training and Valid DataFrames
    
    """
  if num_classes==1:
    valid = pd.DataFrame()
    valid = pd.concat([valid,df[df.label==1].sample(n=valid_positive_text_samples,random_state=seed)])
    valid = pd.concat([valid,df[df.label==0].sample(n=valid_negative_text_samples,random_state=seed)])
    df = df[np.logical_not(df.index.isin(valid.index))]

        
    df.reset_index(drop=True,inplace=True)
    valid.reset_index(drop=True,inplace=True)
    return df,valid
  else:
    valid = pd.DataFrame()
    valid = pd.concat([valid,df[df.label==2].sample(n=valid_positive_text_samples,random_state=seed)])
    valid = pd.concat([valid,df[df.label==1].sample(n=valid_neutral_text_samples,random_state=seed)])
    valid = pd.concat([valid,df[df.label==0].sample(n=valid_negative_text_samples,random_state=seed)])
    df = df[np.logical_not(df.index.isin(valid.index))]

        
    df.reset_index(drop=True,inplace=True)
    valid.reset_index(drop=True,inplace=True)
    return df,valid



def create_submission(
  test_file:pd.DataFrame,
  output_name:str,
  test_predictions:np.array,
  num_classes:int) -> None:

  """ 
  Function that Creates Predictions Submission 
    Args:
      test_file : test_file with samples ID's
      output_name : output file name
      test_predictions : Array of Predictions sorted with ID's
      num_classes : number of classes Available
  """


  test_file["ID"] = test_file["ID"].apply(lambda x:str(0)+x if len(x)==6 else x)
  if num_classes>2:
    test_preds = pd.DataFrame(test_predictions,columns=["Negative","Neutral","Positive"])
    test_file = pd.concat([test_file,test_preds],axis=1)
    test_file[["ID","Negative","Neutral","Positive"]].to_csv(f"./submissions/3Labels/{output_name}.csv",index=False)
  else:
    test_preds = pd.DataFrame(test_predictions,columns=["predictions"])
    test_file = pd.concat([test_file,test_preds],axis=1)
    test_file[["ID","predictions"]].to_csv(f"./submissions/2Labels/{output_name}.csv",index=False)


def setup_environment(random_seed: int, gpu_list: List) -> None:
    """
    Setup Environment Variables .
    Args:
        random_seed: random seed
        gpu_list: list of GPUs available for the experiment
    """

    os.environ["HYDRA_FULL_ERROR"] = "1"
    pl.seed_everything(random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpu_list])
    
