# T2T-for-English to Hindi
Install tensor2tensor using 
```
pip install tensor2tensor
```
## Prepare the dataset
Take two parallel dataset file for each language(English and Hindi)  <br/>
The train files will be train.en and traain.hi <br/>
Put these two files under a folder named training inside $HOME/t2t_data directory <br/>

<br/>
<br/>
The dev files will be dev.en and dev.hi <br/>
Put the dev.en and dev.hi files under a folder named dev <br/>
Put this dev folder inside $HOME/t2t_data directory  <br/>
<br/>

## Create new problem for tensor2tensor

Go to 
```
~/anaconda3/lib/python3.6/site-packages/tensor2tensor/data_generators
```
download the translate_enhi.py and paste it on data_generator directory
<br/>
open all_problems.py under data_generaor folder using any text editor and add this line at the beginning 
```
from tensor2tensor.data_generators import translate_enhi
```



## To Generate training data
In terminal run
```
PROBLEM=translateenhi
MODEL=transformer
HPARAMS=transformer_base
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
```  
## To train the model
```
PROBLEM=translateenhi
MODEL=transformer
HPARAMS=transformer_base
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --train_steps=1000 \
  --eval_steps=100
```
  
  

