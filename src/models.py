import torch
from typing import Dict,List,Optional,Tuple,Union
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torch.nn as nn
from src.dataset import IcompassTuniziDialectClassificationDataset
import hydra
import transformers 
import pandas as pd
from src import utils
import os
import numpy as np

class TuniziDialectClassifier(pl.LightningModule):
  """ Custom Architecture for Zindi & Icompass Sentiment Analysis Competition """

  def __init__(
      self,
      hydra_config:Optional[omegaconf.DictConfig] = None
      ) -> None :
    """
      Args:
        hydra_config: Hydra Configuration with All needed Parameters
    """
    super(TuniziDialectClassifier,self).__init__()
    self.config = hydra_config
    self.__build_model()
    self.__build_loss()
    self.__build_metric()
    self.best_validation_accuracy = 0



  def __build_model(self) -> None:
    """ Instantiate Bert/Roberta + LSTM and The Classification Head """
    self.model = hydra.utils.instantiate(
        self.config.models
    )
    self.lstm = nn.LSTM(self.model.config.hidden_size,self.config.model.lstm_hidden_size,bidirectional = True,batch_first= True)
    self.output = nn.Linear(self.config.model.lstm_hidden_size*2,self.config.classifiermode.num_classes)

  def __build_loss(self) -> None:
    """ Instantiate Loss Function """
    self.criterion = hydra.utils.instantiate(
        self.config.loss_fn
    )

  def forward(self,input_ids,attention_mask) -> torch.tensor:
    sequence_out = self.model(input_ids,attention_mask=attention_mask).last_hidden_state
    out,(final_hidden_state, final_cell_state) = self.lstm(sequence_out)
    out = self.output(out[:,-1].view(-1,self.config.model.lstm_hidden_size*2))
    return out
  
  def configure_optimizers(self) -> Tuple[torch.optim.Optimizer,transformers.SchedulerType]:

    """ Look At Base Class ."""
    num_training_steps = len(self.train_dataloader()) * self.config.training.max_epochs
    optimizer = hydra.utils.instantiate(
        self.config.optimizer, params=self.parameters()
    )

    scheduler = hydra.utils.instantiate(
      self.config.scheduler,
      optimizer = optimizer,
      num_training_steps = num_training_steps,
      num_warmup_steps = 0
    )
    return [optimizer],[scheduler]

  def configure_callbacks(self) -> Tuple[Callback,Callback]:
    """ Look At Base Class ."""
    earlystopping_callback = hydra.utils.instantiate(
        self.config.callbacks.early_stopping
    )
    checkpoint_callback = hydra.utils.instantiate(
      self.config.callbacks.model_checkpoint
    )
      
    return earlystopping_callback,checkpoint_callback

  def __build_metric(self) -> None:
    """ Instantiate Metric """
    self.metric = pl.metrics.Accuracy()

  def setup(self, stage: str = "fit") -> None:
    self.train_df = pd.read_csv(self.config.training.train_csv)
    if self.config.classifiermode.num_classes==1:
      self.train_df = self.train_df[self.train_df.label!=0]
      self.train_df["label"] = self.train_df["label"].apply(lambda x:0 if x==-1 else x)
    else:
      self.train_df["label"] +=1

    self.train_df["sequence_length"] = self.train_df["text"].apply(lambda x:len(x.split(' ')))
    self.train_df = self.train_df[(self.train_df["sequence_length"]<=1000)]

    self.train_df,self.valid_df = utils.split_data(
      self.train_df,self.config.general.seed,
      num_classes = self.config.classifiermode.num_classes,
      valid_positive_text_samples=int(self.config.classifiermode.valid_samples * float(self.config.classifiermode.valid_positive_text_samples)),
      valid_negative_text_samples=int(self.config.classifiermode.valid_samples * float(self.config.classifiermode.valid_negative_text_samples)),
      valid_neutral_text_samples=int(self.config.classifiermode.valid_samples * float(self.config.classifiermode.valid_neutral_text_samples))
    )

  def training_step(self,batch,batch_idx) -> torch.tensor:
    """ 
    Runs one training step. it consists in the forward function followed by the loss function.
      Args:
        batch: The output of your dataloader. 
        batch_idx: Integer displaying which batch this is
    Returns:
      - Training loss to be added to the lightning logger.
    """

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    targets = batch["targets"]

    outputs = self(input_ids=input_ids,
                    attention_mask=attention_mask)


    if self.config.classifiermode.num_classes==1:
      targets = targets.type_as(outputs)
      loss = self.criterion(outputs,targets.view(-1,1))
    else:
      targets = targets.long()
      loss = self.criterion(outputs,targets)
    self.log(
        'train_loss', loss, on_step=True,
        on_epoch=True, prog_bar=True , logger=True
    )
    return loss

  def validation_step(self,batch, batch_idx) -> Dict:
    """ Similair to Training Step but Model in eval mode """

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    targets = batch["targets"]


    outputs = self(input_ids=input_ids,
                    attention_mask=attention_mask)
    if self.config.classifiermode.num_classes==1:
      targets = targets.type_as(outputs)
      val_loss = self.criterion(outputs,targets.view(-1,1))
      outputs = torch.sigmoid(outputs)
    else:
      targets = targets.long()
      val_loss = self.criterion(outputs,targets)
      outputs = torch.softmax(outputs,1)

    self.log(
        'val_loss', val_loss, on_step=True,
        on_epoch=False, prog_bar=True , logger=True 
    )
    targets = targets.int()
    val_accuracy = self.metric(outputs,targets)

    self.log(
        'val_accuracy', val_accuracy, on_step=True,
        on_epoch=False, prog_bar=True , logger=True
    )
    return outputs
    
  def validation_epoch_end(self,outputs) -> None:
    """ Measures Model Performance Across Validation Dataset """
    outputs = np.vstack([x.cpu().detach().numpy() for x in outputs])
    if outputs.shape[0]==self.valid_df.shape[0]:
      if self.config.classifiermode.num_classes > 2:
        outputs = np.argmax(outputs,1)
      else:
        outputs = (outputs>=self.config.classifiermode.threshhold).astype(int)
        val_accuracy = accuracy_score(outputs,self.valid_df["label"])
        if val_accuracy > self.best_validation_accuracy:
          self.best_validation_accuracy = val_accuracy
      
    else:
      val_accuracy = 0
    self.log(
        'val_accuracy', val_accuracy,prog_bar=True , logger=True
    )

  def train_dataloader(self) -> DataLoader:
    """ Function that Loads Train DataLoader"""
    self.train_ds = IcompassTuniziDialectClassificationDataset(
        text = self.train_df["text"],
        target = self.train_df["label"],
        max_length = self.config.model.sequence_max_length,
        hydra_config = self.config,
    )

    self.train_dl = DataLoader(
        self.train_ds,
        batch_size = self.config.training.batch_size,
        num_workers = self.config.general.num_workers,
        shuffle = True
    )

    return self.train_dl

  def val_dataloader(self) -> DataLoader:
    """ Function that Loads Validation DataLoader"""
    self.valid_ds = IcompassTuniziDialectClassificationDataset(
        text = self.valid_df["text"],
        target = self.valid_df["label"],
        max_length = self.config.model.sequence_max_length,
        hydra_config = self.config,
    )

    self.valid_dl = DataLoader(
        self.valid_ds,
        batch_size = self.config.training.batch_size,
        num_workers = self.config.general.num_workers,
        shuffle = False
    )

    return self.valid_dl

  def test_dataloader(self) -> DataLoader:
    """ Function that Loads Test DataLoader"""
    self.test_df  = pd.read_csv(self.config.testing.test_csv)
    self.test_df["label"] = 0

    self.test_ds = IcompassTuniziDialectClassificationDataset(
        text = self.test_df["text"],
        target = self.test_df["label"],
        max_length = self.config.model.sequence_max_length,
        hydra_config = self.config,
    )

    self.test_dl = DataLoader(
        self.test_ds,
        batch_size = self.config.training.batch_size,
        num_workers = self.config.general.num_workers,
        shuffle = False
    )

    return self.test_dl

    





  


