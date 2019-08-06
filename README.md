# Multitask learning 

Multitask learning framework for sequence prediction and sequence labelling 

## Environment setup 

This project works under two versions of pytorch : torch 0.4 and torch 1.0  

Therefore two possible conda env : 
- `conda env create -f lm.yml`  (1.0) (migh need to add missing, `pip install --upgrade oauth2client` !)

In `./env/project_variables.py` define BERT_MODELS_DIRECTORY as the location of the bert tar.gz and vocabulary file 

`mkdir BERT_MODELS_DIRECTORY`

`mkdir checkpoints`
`mkdir checkpoints/bert`

## Downloading Bert Models 


`cd BERT_MODELS_DIRECTORY`

### Multilingual base cased model 

`curl -O "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt"` <br>
`curl -O "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz"`

### English base cased model 

`curl -O  "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz"` <br>
`curl -O "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt"`


## Data

Data should be in CoNLLU format https://universaldependencies.org/format.html

To download the all annotaded UD data 

`curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2899{/conll-2017-2018-blind-and-preprocessed-test-data.zip}`


## Training and evaluating 



### POS (master branch)

```
python ./train_evaluate_bert_normalizer.py 
## Data
--train_path ./data/en-ud-train-demo.conllu  # 
--dev_path ./data/en-ud-dev-demo.conllu # 
--test_path ./data/en-ud-test-demo.conllu  # 
## tasks 
--tasks pos ## as a space separated list of task among  ['normalize', 'pos', 'edit', 'norm_not_norm']
## tokenization 
--tokenize_and_bpe 0 #
## architecture
--bert_module mlm # 
--initialize_bpe_layer 1 #
--bert_model bert_base_multilingual_cased  # bert models cf. env/model_dir to see available models 
--aggregating_bert_layer_mode last ## 
--layer_wise_attention 0 ##
--append_n_mask 0
## optimization
--epochs 1 
--batch_size 2 #
--lr 5e-05 #
--freeze_parameters 0 #
--fine_tuning_strategy standart # 
## regularization 
--dropout_classifier 0.1 ## 
--dropout_input_bpe 0.0 ##
## dump path 
--overall_report_dir ./checkpoints/28d8d-B-summary #
--model_id_pref 28d8d-B-model_1 #
``` 

### Parsing (!! dev branch still unstable)

 python ./train_evaluate_bert_normalizer.py --train_path /Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../../parsing/normpar/data/en_lines+ewt-ud-train.conllu --dev_path /Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../../parsing/normpar/data/en-ud-dev.integrated-po_as_norm --test_path /Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../../parsing/normpar/data/en_lines+ewt-ud-train.conllu /Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../../parsing/normpar/data/en-ud-test.conllu /Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../../parsing/normpar/data/en-ud-dev.integrated-po_as_norm --tasks parsing --batch_size 4 --lr 1e-05 --initialize_bpe_layer 1 --freeze_parameters 0 --bert_model cased --dropout_classifier 0.0 --fine_tuning_strategy standart --dropout_input_bpe 0.0 --bert_module mlm --append_n_mask 0 --dropout_bert 0.1 --aggregating_bert_layer_mode last --layer_wise_attention 0 --tokenize_and_bpe 0 --multitask 1 --overall_label a241d-B --model_id_pref a241d-B-model_1 --epochs 1 --overall_report_dir /Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../checkpoints/a241d-B-summary


### Reporting details 

In ./checkpoint/bert/model_full_name folder :

Model dictionary, predictions, checkpoints, argument and performance report will be written in it 


### doc TODO 
- handle report_template 
- add list of all possible arguments with comments on their default and their use 
- test the full git pull, conda create , and curl 
- test a train and evaluation 



