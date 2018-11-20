from model.seq2seq import LexNormalizer, Generator
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import pdb
import time

#pdb.set_trace = lambda: 1

data_in = torch.LongTensor([[1., 2., 4., 0],[2., 5., 1., 0]])
data_out = torch.LongTensor([[1, 0, 0, 0],[5, 1, 2, 0]])

# DEFINE MODEL
model = LexNormalizer(generator=Generator, char_embedding_dim=5, hidden_size_encoder=11, hidden_size_decoder=11, verbose=0)
#forward = model.forward(input_seq=data_in, output_seq=data_out, input_mask=None, output_mask=None)
#pdb.set_trace()
#print(forward)
#pdb.set_trace()
#print(loss.size(), loss)

# ---------------------------------------------------------------------- #


#  THEN run_epoch()
# generate x, y sequences
# do forward pass using model.forward() to get prediction
# LossCompute takes the model.generator, the CE model --> output the CE
# do loss_compute(prediction, y sequences, LossCompute)

# TODO :
# add real_data io : link to characters
# then build code to play with the model (write a noisy code --> gives you the prediction)
# plug tensorboard



# TODO
##confirm dimensions output
##TODO 2 Design test masking :
##-- : for test : you can try with padding sequences (should I pad with zeros : yes) : print the mask along forward pass to see (with various batch , seq len ...)
##-- : both input and output
## come up with a test to make sure the seq output never see the target !
## + a test on the softmax score !

