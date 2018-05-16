# T2T-for-English to Hind
Install tensor2tensor using 
pip install tensor2tensor

## Prepare the dataset
Take two parallel dataset file for each language(English and Hind)
The train files will be train.en and traain.hi
Put the training files under a folder named training
compress the training folder to train_enhi.tar.gz
Rename the train_enhi.tar.gz to train_enhi
Again compress train_enhi to train_enhi.gz

The dev files will be dev.en and dev.hi
Put the dev.en and dev.hi files under a folder named dev
compress the  dev folder dev.tar.gz
Rename the dev.tar.gz to dev
Again compress dev to dev.gz

Now move train_enhi.gz and dev.gz to ~/t2t

## Create new problem for tensor2tensor
Go to 
~/anaconda3/lib/python3.6/site-packages/tensor2tensor/data_generators
and open any exsisting model such as translate_enzh.py in text editor

## Replace (line 48)

_NC_TRAIN_DATASETS = [[
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12"
    ".tgz", [
        "training/news-commentary-v12.zh-en.en",
        "training/news-commentary-v12.zh-en.zh"
    ]
]]


_NC_TEST_DATASETS = [[
    "http://data.statmt.org/wmt17/translation-task/dev.tgz",
    ("dev/newsdev2017-enzh-src.en.sgm", "dev/newsdev2017-enzh-ref.zh.sgm")
]]

## to

_enhi_TRAIN_DATASETS = [
    [
        "~/t2t/train_enhi.gz",  # pylint: disable=line-too-long
        ("training/train.en",
         "training/train.hi")
    ],
]

_enhi_TEST_DATASETS = [
    [
        "~/t2t//dev.gz",
        ("dev/dev.en", "dev/dev.hi")
    ],
]

## To Generate training data
In terminal run

PROBLEM=translateenhi_main
MODEL=transformer
HPARAMS=transformer_base
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
  
## To train the model
PROBLEM=translateenhi_main
MODEL=transformer
HPARAMS=transformer_base
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR
t2t-trainer \
  --generate_data \
  --data_dir=~/t2t_data \
  --output_dir=~/t2t_train \
  --problem=translateenhi_main \
  --model=transformer \
  --hparams_set=transformer_base \
  --train_steps=100000 \
  --eval_steps=10000
  
  
  

