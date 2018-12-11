# mt_norm_parse

multitask learning model for normalization and dependency parsing


## Verbosity typology

### Model + batching

0 is only starting , end with final loss
1 includes 0 + epoch-wise information : loss, + info about the epochs
2 includes 0 + 1 + batch wise information like loss êr batch + summary info on each batch
3 includes 0 + 1 + 2 + dimensions information of each tensors of the input, output the model, the loss
4 : add masking info + packed_sequence info
5 : printing data

### Sequence

verbose == 4 to see decoding


