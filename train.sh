python train.py \
general.seed=1 \
classifiermode=twolabels \
models=bertMCased \
model.architecture_name=bert_lstm \
training.lr=3e-5

python train.py \
general.seed=2020 \
classifiermode=twolabels \
models=robertaXLM \
model.architecture_name=roberta_lstm \
training.lr=1e-5

python train.py --multirun \
general.seed=2020,42,2000,123456 \
classifiermode=twolabels \
models=bertMCased \
model.architecture_name=bert_lstm \
training.lr=1e-5






# python train.py \
# general.seed=1 \
# classifiermode=threelabels \
# models=bertMCased \
# model.architecture_name=bert_lstm \
# training.lr=3e-5 \
# loss_fn=crossEntropyLoss \


# python train.py \
# general.seed=2020 \
# classifiermode=threelabels \
# models=robertaXLM \
# model.architecture_name=roberta_lstm \
# training.lr=1e-5 \
# loss_fn=crossEntropyLoss \

# python train.py --multirun \
# general.seed=2020,42,2000,123456 \
# classifiermode=threelabels \
# models=bertMCased \
# model.architecture_name=bert_lstm \
# training.lr=3e-5 \
# loss_fn=crossEntropyLoss \


