import numpy as np


def output_text(one_code_prediction, char_dic):
    decoding = []
    for batch in range(one_code_prediction.size(0)):
        word = []
        for char in range(one_code_prediction.size(1)):
            word.append(char_dic.get_instance(one_code_prediction[batch, char]))
        decoding.append(word)
    return np.array(decoding)
