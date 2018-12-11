import numpy as np
from io_.dat.constants import END_CHAR, PAD_CHAR, CHAR_START


def output_text(one_code_prediction, char_dic, start_symbol=CHAR_START ,
                stop_symbol=PAD_CHAR, single_sequence=True):
    decoding = []
    str_decoded = []
    for batch in range(one_code_prediction.size(0)):
        word = []
        word_to_print = ""
        for char in range(one_code_prediction.size(1)):
            char_decoded = char_dic.get_instance(one_code_prediction[batch, char])
            word.append(char_decoded)
            if not char_decoded == stop_symbol and not char_decoded == start_symbol:
                word_to_print += char_decoded
        decoding.append(word)
        if single_sequence:
            str_decoded = word_to_print
        else:
            str_decoded.append(word_to_print)
    return np.array(decoding), str_decoded

