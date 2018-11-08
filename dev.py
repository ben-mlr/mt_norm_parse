from model.seq2seq import LexNormalizer
import torch.nn as nn
import torch
import numpy as np
import pdb

pdb.set_trace = lambda: 1

data_in = torch.LongTensor([[1., 2., 4., 0]])
data_out = torch.LongTensor([[1, 0, 0, 0]])

model = LexNormalizer()
forward = model.forward(input_seq=data_in, output_seq=data_out)
pdb.set_trace()

print(forward)
loss = model.loss(input_seq=data_in, output_seq=data_out)
pdb.set_trace()
print(loss.size(), loss)
if False:
    data_in = torch.LongTensor([1., 2., 4., 5.])
    char_embedding = nn.Embedding(num_embeddings=6, embedding_dim=5)
    print(char_embedding(data_in))