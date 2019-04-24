# Multitask learning 

Multitask learning framework for sequence prediction and sequence labelling 

## Environment setup 

### setup 

This project works under two versions of pytorch : torch 0.4 and torch 1.0  

Therefore two possible conda env : 
- Install conda environment with mt_norm_parse.yml (0.4)
- Install conda environment with lm.yml  (1.0) (ALLENNLP missing, pip install --upgrade oauth2client !)


NB : in rioc loading conda and gcc might be required 

### test environment 

sh test_/run_py_env_test.sh


## Highlights

- the main classes are in ./model . They allow architecture definition
- the main command scripts are ./train_evaluate_run.py and ./grid_run.py to train and evaluate a single model or a several of them.

### Vocabularies  

2 modes :
- one model specific : we create the dictionary inside the model folder
- other is standard
- TODO : should make the dict_path in argument.json 

## Test guidelines

When any new features is implemented the impact the training process : 

- train  on normalization == False and see if it fits demo2 dataset
- train on normalization == True and see if it fits demo2 dataset
- evaluate the model and get the metrics  

## Evaluation pipeline : highlights

- We tried to defined a task agnostic and score agnostic pipeline for evaluation  
- Now the idea is : 
   - at the lower level 
        - you define : a scoring formula dictionary that gives a score name and how to get it (what fraction to do)
        - the numerators, denominator
   - This is used toward upper level (after iterating over all batch) for summing all relevant stats and computing the score using the formula   
- It's meant to be flexible


## Conventions 

- id 0 in all dictionary is default one : UNK tokens


## Verbosity typology

### Model + batching

- 0 is only starting , end with final loss and must read warnings/info
- 1 includes 0 + epoch-wise information : loss, + info about the epochs 
- 2 includes 0 + 1 + batch wise information like loss per batch + summary info for each batch 
- 3 includes 0 + 1 + 2 + dimensions information of each tensors of the input, output the model, the loss 
- 4 : add masking info + packed_sequence info + type info 
- 5 : printing data 


### Sequence

verbose == 4 to see decoding


## ABOUT BERT 

### About BPE tokenization 

Some statistics on the impact of BPE tokenization on the word noisy-clean alignement 
- dataset : /normpar/data/lexnorm.integrated  
	- Out of 328 batch of 1 sentences each : 46 batch aligned ; 277 with at least 1 sentence noisy MORE SPLITTED ; 4 with  LESS SPLITTED  
- dataset : owoputi : Out of 306 batch of 1 sentences each : 58 batch aligned ; 226 with at least 1 sentence noisy MORE SPLITTED ; 21 with  LESS SPLITTED  
- dataset : LIU TRAIN
	- Out of 2009 batch of 1 sentences each : 444 batch aligned ; 1243 with at least 1 sentence noisy MORE SPLITTED ; 321 with  LESS SPLITTED  
- dataset : LIU DEV 
	- Out of 569 batch of 1 sentences each : 135 batch aligned ; 342 with at least 1 sentence noisy MORE SPLITTED ; 91 with  LESS SPLITTED  
- LEXNORM TRAIN 
	- Out of 1000 batch of 1 sentences each : 483 batch aligned ; 249 with at least 1 sentence noisy MORE SPLITTED ; 268 with  LESS SPLITTED  
- LEXNORM DEV
	- Out of 1000 batch of 1 sentences each : 476 batch aligned ; 260 with at least 1 sentence noisy MORE SPLITTED ; 264 with  LESS SPLITTED  

FOR liu, lexnorm.integrated  and owoputi not have to handle 1 to n 
For lexnorm_train, dev have to 

### TODO 

#### To document

- assumptions : 
    - should have word to word mapping between source and target sequence
 - About word encoder : 
    - in the mode it's coded properly
    - in the data iterator its not really 
        - normalization means only char decoder
        - if you set word_decoding True it adds word in the output iterator   

