import numpy as np
from io_.dat.constants import END_CHAR, PAD_CHAR, CHAR_START
import pdb


def output_text(one_code_prediction, char_dic, start_symbol=CHAR_START ,
                stop_symbol=END_CHAR, single_sequence=True):
    decoding = []
    str_decoded = []
    for batch in range(one_code_prediction.size(0)):
        word = []
        word_to_print = ""
        for char in range(one_code_prediction.size(1)):
            char_decoded = char_dic.get_instance(one_code_prediction[batch, char])
            word.append(char_decoded)
            #if not char_decoded == stop_symbol and not char_decoded == start_symbol:

            if char_decoded == stop_symbol:
                break
            if not char_decoded == start_symbol:
                word_to_print += char_decoded

        decoding.append(word)
        if single_sequence:
            str_decoded = word_to_print
        else:
            str_decoded.append(word_to_print)
    return np.array(decoding), str_decoded


def output_text_(one_code_prediction, char_dic, start_symbol=CHAR_START ,
                 stop_symbol=END_CHAR, single_sequence=True):

    decoding = []
    str_decoded = []

    for batch in range(one_code_prediction.size(0)):
        sent = []
        word_str_decoded = []
        for word_i in range(one_code_prediction.size(1)):
            word = []
            word_to_print = ""
            for i_char, char in enumerate(range(one_code_prediction.size(2))):
                char_decoded = char_dic.get_instance(one_code_prediction[batch, word_i, char])
                # if not char_decoded == stop_symbol and not char_decoded == start_symbol:
                empty_decoded_word = False
                # We break decoding when we reach padding symbol or stop symnol
                if (char_decoded == stop_symbol) or (char_decoded == PAD_CHAR):
                    # WARNING : we assume always add_start = 1 ! we also :
                    if i_char == 1 and (char_decoded == stop_symbol or char_decoded == PAD_CHAR):
                        empty_decoded_word = True
                    # we break if only one padded symbol witout adding anything
                    # to word to print : only one PADDED symbol to the array
                    break
                # we append word_to_print only starting the second decoding (we assume _START is here)
                if not (char_decoded == start_symbol and i_char == 0):
                    word_to_print += char_decoded
                word.append(char_decoded)

            sent.append(word)
            if single_sequence:
                word_str_decoded = word_to_print
            else:
                # we want to remove gold empty words (coming from the sentence level padding)
                if len(word_to_print) > 0 or empty_decoded_word:
                    word_str_decoded.append(word_to_print)
                #else:     printing("Word to print empty ")
        if not single_sequence:
            str_decoded.append(word_str_decoded)
        else:
            str_decoded = word_to_print
        decoding.append(sent)
    return np.array(decoding), str_decoded

