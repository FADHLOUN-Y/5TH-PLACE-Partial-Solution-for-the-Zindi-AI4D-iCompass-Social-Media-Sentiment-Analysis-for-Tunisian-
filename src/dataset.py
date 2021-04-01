from torch.utils.data import Dataset
import torch
from typing import Dict,List,Optional
import omegaconf
import hydra




class IcompassTuniziDialectClassificationDataset(Dataset):
    """ Custom Dataset for Zindi & Icompass Sentiment Analysis Competition """ 
    def __init__(
        self,
        text:List[str],
        target:List[int],
        max_length:int,
        hydra_config:Optional[omegaconf.DictConfig] = None,
        ) -> None:

      """ 
        Args:
          hydra_config: Hydra Configuration with All needed Parameters
          text: 
          target : 
        Returns:
          dictionary with the expected model inputs and targets .
      """
      self.config = hydra_config
      self.text = text
      self.target = target
      self.__init_tokenizer()
        
    def __len__(self) -> int:
      return len(self.text)
    
    def __init_tokenizer(self) -> None:
      self.tokenizer = hydra.utils.instantiate(
          self.config.tokenizer
      )

    def __getitem__(self,idx) -> Dict:
      sentence = str(self.text[idx])
      inputs = self.tokenizer.encode_plus(
          sentence,
          None,
          add_special_tokens=True,
          padding="max_length",
          truncation=True,
          max_length=self.config.model.sequence_max_length,
          return_attention_mask=True,
          return_token_type_ids=False
      )
      target = self.target[idx]
      input_ids = inputs["input_ids"]
      attention_mask = inputs["attention_mask"]
      return {"input_ids":torch.tensor(input_ids,dtype=torch.long),
              "attention_mask":torch.tensor(attention_mask,dtype=torch.long),
              "targets":torch.tensor(target,dtype=torch.long)}
