
import torch.nn as nn


class BinaryPredictor(nn.Module):

    def __init__(self, input_dim):
        super(BinaryPredictor, self).__init__()
        self.predictor = nn.Linear(input_dim, out_features=2)

    def forward(self, encoder_state_projected):
        # encoder_state_projected size : [batch, ??, dim decoder (encoder state projected)]
        hidden_state_normalize_not = self.predictor(encoder_state_projected)
        # hidden_state_normalize_not  size [batch, ???, 2]
        return hidden_state_normalize_not