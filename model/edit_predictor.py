import torch.nn as nn
import torch



class EditPredictor(nn.Module):
    def __init__(self,  input_dim, dense_dim=0, dense_dim_2=0, verbose=2):
        super(EditPredictor, self).__init__()

        if dense_dim>0:
            self.dense_output_1 = nn.Linear(input_dim, dense_dim)
        else:
            self.dense_output_1 = None
        if dense_dim_2 >0:
            assert dense_dim>0
            self.dense_output_2 = nn.Linear(dense_dim, dense_dim_2)
        else:
            self.dense_output_2 = None
            if dense_dim>0:
                dense_dim_2=dense_dim
            else:
                dense_dim_2=input_dim
        self.dense_output_3 = nn.Linear(dense_dim_2, 1)

    def forward(self, context):
        if self.dense_output_1 is not None:
            context = torch.sigmoid(self.dense_output_1(context))
        if self.dense_output_2 is not None:
            context = torch.sigmoid(self.dense_output_2(context))
        prediction_state = torch.sigmoid(self.dense_output_3(context))
        return prediction_state
