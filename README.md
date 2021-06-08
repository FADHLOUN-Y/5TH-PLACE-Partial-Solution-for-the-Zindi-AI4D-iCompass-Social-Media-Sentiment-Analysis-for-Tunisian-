# 5TH PLACE Partial Solution for the Zindi AI4D iCompass Social Media Sentiment Analysis for Tunisian Arabizi

[Redirect to Challenge Website](https://zindi.africa/competitions/ai4d-icompass-social-media-sentiment-analysis-for-tunisian-arabizi)

## Objective Of the Challenge

The objective of this challenge is to, given a sentence, classify whether the sentence is of positive, negative, or neutral sentiment. For messages conveying both a positive and negative sentiment, whichever is the stronger sentiment should be chosen. Predict if the text would be considered positive, negative, or neutral (for an average user).

## Quick Introduction
The dataset has three sets of labels - Negative,Positive,Neutral and was highly Imbalanced.
**54 % Positive Samples**
**42 % Negative Samples**
**04  % Neutral Samples**
## Our Approach
We encountered two main problems with the dataset which are annotation and imbalacing of text classes .. 
We started to believe even if we had the chance to add external data and train our language modeling model we won't achieve great results so as a result of that we proceeded with two different approaches text-augmentation (which is not included in this solution )  and turning the problem into a binary task keeping only positive and negative texts . 
Our models were a combination between transformers and reccurent neural networks ( **RobertaXLM - LSTM** ) and (**Bert Multilingual Cased - LSTM**) inspired from Icompass Paper [Learning Word Representations for Tunisian Sentiment Analysis](https://arxiv.org/pdf/2010.06857.pdf) .

## Instructions to run the code

Code could be run using google colab .
#### Environment Setup

You'll find a requirement file that you could install in your own virtual environment.

#### Data Setup
Download data from the competition website and save it to the ./data/ directory.

#### Training Phase
To start training the models run ./train.sh it will take about 4 hours to run the two architectures .

#### Testing Phase

To start testing the models run ./test.sh it will create test files with ID of text sample and predictions if you are using the twolabels classifier mode or Negative,Neutral and Positive predictions if threelabels classifiermode is used .


#### Blending Phase

./inference.sh could be run after training and test phases it will create the best submission with IDs and labels (-1 Negative , 1 Positive ) if classifiermode used is twolabels or (-1 , 0 Neutral , 1 ) if classifiermode used is threelabels .



## Single Best Model

Best single model was BertMultilingualCased combined with LSTM averaged on 5 folds of the test set which achieved **83.52** Accuracy .

### Hyperparameters

ArchitectureName : Bert-LSTM
LossFunction : BCEWithLogistLoss
SequenceMaxLength : 64
BatchSize : 32
LearningRate : 3e-5
NumClasses : 2
Threshhold : 0.5


## Blending

Final model was an average of all models checkpoints which achieved **0.8382** on the private LB.

Architectures : BertMultilingualCased-LSTM , RobertaXLM-LSTM and BertMultilingualUncased trained on augmented data.

SequenceMaxLength : 64,128

## What didn't Work 

-- We had around 300k tokens for only 70000 samples for training set , there were a lot of similair words but with similair meaning but different spelling so the idea was to cluster those words grouping each words that can be written with same words and in the same order and then map each word in training samples with same chars and same order for them into one token .

-- Pseudo Labeling .






