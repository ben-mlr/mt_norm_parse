# mt_norm_parse

multitask learning model for normalization and dependency parsing


### Dictionary 

2 modes :
- one model specific : we create the dictionary inside the model folder
- other is stanart
- TODO : should make the dict_path in argument.json 

## Test guidelines

When any new features is implemented the impact the training process : 

- train  on normalization == False and see if it fits demo2 dataset
- train on normalization == True and see if it fits demo2 dataset
- evaluate the model and get the metrics  



## Verbosity typology

### Model + batching

- 0 is only starting , end with final loss 
- 1 includes 0 + epoch-wise information : loss, + info about the epochs 
- 2 includes 0 + 1 + batch wise information like loss êr batch + summary info on each batch 
- 3 includes 0 + 1 + 2 + dimensions information of each tensors of the input, output the model, the loss 
- 4 : add masking info + packed_sequence info 
- 5 : printing data 


### Sequence

verbose == 4 to see decoding


### To document

- assumptions : 
    - should have word to word mapping between source and target sequence

