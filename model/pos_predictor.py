from env.importing import *


class PosPredictor(nn.Module):
    def __init__(self,  input_dim,
                 voc_pos_size, dense_dim=0, dense_dim_2=0,
                 dropout_dense=0.0,
                 verbose=2):
        super(PosPredictor, self).__init__()
        if dense_dim > 0:
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
                dense_dim_2 = input_dim
        if dropout_dense > 0:
            drop_dense = nn.Dropout(dropout_dense)
        self.dense_output_3 = nn.Linear(dense_dim_2, voc_pos_size)

    def forward(self, context):
        if self.dense_output_1 is not None:
            context = nn.LeakyReLU()(self.dense_output_1(context))
        if self.dense_output_2 is not None:
            context = nn.LeakyReLU()(self.dense_output_2(context))
        prediction_state = self.dense_output_3(context)
        return prediction_state
