import torch.nn as nn


class PosPredictor(nn.Module):
    def __init__(self,  input_dim,
                 voc_pos_size, dense_dim, dense_dim_2=None,
                 verbose=2):
        super(PosPredictor, self).__init__()

        self.dense_output_1 = nn.Linear(input_dim, dense_dim)
        self.dense_output_2 = nn.Linear(dense_dim, voc_pos_size)

    def forward(self, context):

        prediction_state = nn.ReLU()(self.dense_output_2(nn.ReLU()(self.dense_output_1(context))))
        return prediction_state
