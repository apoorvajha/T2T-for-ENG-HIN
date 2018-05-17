# T2T-for-English to Hind
Install tensor2tensor using 
```
pip install tensor2tensor
```
## Prepare the dataset
Take two parallel dataset file for each language(English and Hind)  <br/>
The train files will be train.en and traain.hi <br/>
Put the training files under a folder named training <br/>
compress the training folder to train_enhi.tar.gz <br/>
Rename the train_enhi.tar.gz to train_enhi <br/>
Again compress train_enhi to train_enhi.gz <br/>
<br/>
<br/>
The dev files will be dev.en and dev.hi <br/>
Put the dev.en and dev.hi files under a folder named dev <br/>
compress the  dev folder dev.tar.gz <br/>
Rename the dev.tar.gz to dev <br/>
Again compress dev to dev.gz <br/>
<br/>
Now move train_enhi.gz and dev.gz to ~/t2t <br/>
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
PROBLEM=translateenhi_main
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
PROBLEM=Translateenhi
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
```
  
  

