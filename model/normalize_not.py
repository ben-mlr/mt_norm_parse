
import torch.nn as nn
from io_.info_print import printing

class BinaryPredictor(nn.Module):

    def __init__(self, input_dim, dense_dim, verbose=1):
        super(BinaryPredictor, self).__init__()
        self.verbose = verbose
        if dense_dim is not None:
            self.dense = nn.Linear(input_dim, dense_dim)
            printing("WARNING : BinaryPredictor dense_dim is set to {} in norm_not_norm predictor",var=dense_dim,
                     verbose=self.verbose, verbose_level=1)
        else:
            self.dense = None
            printing("WARNING : BinaryPredictor as dense_dim is None no dense layer added to norm_not_norm predictor",
                     verbose=self.verbose, verbose_level=1)
            dense_dim = input_dim
        self.predictor = nn.Linear(dense_dim, out_features=2)

    def forward(self, encoder_state_projected):
        # encoder_state_projected size : [batch, ??, dim decoder (encoder state projected)]

        if self.dense is not None:
            intermediary = nn.ReLU()(self.dense(encoder_state_projected))
        else:
            intermediary = encoder_state_projected
            # hidden_state_normalize_not  size [batch, ???, 2]
        hidden_state_normalize_not = self.predictor(intermediary)
        return hidden_state_normalize_not