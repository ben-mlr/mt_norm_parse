# TODO

## Description

1- Code
- push to rioc and make it gpu compatible
- experiments and measure on reproduction
- add smart checkpointing
- add validation loss
- switch to sentence level and add context


2- Bug to solve
- end symbol to add
- I think there is a bug in your score metric if you have unknown characters
- edit distance : what to do if empty string ?
2- Experiments

## Bag of TODOs

### Dev todo

1 evaluate on reproduction
2
3 quick experiments on lexnorm on predicting NormPar + LexNorm + Owuputi + EWT: Norm=
TODO : when you do packed sequence you do teacher force : add inference-like training
TODO : add stop symbol in the data + at decoding time
TODO : add quick training report

TODO :
then build code to play with the model (write a noisy code --> gives you the prediction)
plug tensorboard



ressources : https://bastings.github.io/annotated_encoder_decoder/
             https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

NB : batch_size is in data_gen_conllu relates to the collu format setence level !!
NB you cant have batch_size == 1 WHY ??


### iterator todo

RECALL :
the data_gen_conllu originally generates batch of sentences :
            #  the batch_size provided corresponds of batch of sentences
if the batch_size of sentences if greater than the number of sentences available in the data
            #  --> it will fall on the length og the current bucket : for instance :
            #  2 if only two sentences in the bucket (5 word in the sentnce)

DEAL WITH data_iterator :
- size problem
then get it as correct matrix array in numpy
then use it in the batch iterator
then fit model !!

ABOUT SYMBOLIC ROOT + END
- For character level : if you dont encode character sequence
at the sentence level but only at the word level :
then you dont need the symbolic ROOT/END for the sentence within characters
Pb : with symbolic END if normalization

ABOUT SWITCHING TO SEQUENCE AT THE SENTENCE LEVEL
you have to make input character sequence and target fit in the packed seqence...