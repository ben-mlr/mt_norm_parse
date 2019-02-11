import numpy as np
from io_.dat.constants import END_CHAR, PAD_CHAR, CHAR_START, PAD_ID_WORD
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


def output_text_(one_code_prediction, char_dic=None, start_symbol=CHAR_START,
                 output_str=False, word_dic=None, word_decode=False, char_decode=True,
                 stop_symbol=END_CHAR, single_sequence=True, last=False, debug=False):

    decoding = []
    str_decoded = []
    words_count = 0
    decoding_all_sequences = []
    for batch in range(one_code_prediction.size(0)):
        sent = []
        word_str_decoded = []
        sent_all_sequence = []
        for word_i in range(one_code_prediction.size(1)):
            word = []
            word_to_print = ""
            word_all_sequence = []
            break_word_to_print = False
            if char_decode:
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
                        break_word_to_print = True
                        #break
                    # we append word_to_print only starting the second decoding (we assume _START is here)
                    if not (char_decoded == start_symbol and i_char == 0) and not break_word_to_print:
                        word_to_print += char_decoded

                    if not break_word_to_print:
                        word.append(char_decoded)
                    word_all_sequence.append(char_decoded)
            empty_decoded_word = False

            if word_decode:
                word_to_print = word_dic.get_instance(one_code_prediction[batch, word_i]) if one_code_prediction[batch, word_i] != PAD_ID_WORD or word_i == 0 else "" #shoule make that more general
                if len(word_to_print)>0:
                    word_all_sequence.append(word_to_print)
                word = word_to_print
            if len(word) > 0:
                #print("WARNING : from_array_to_text.py --> adding filter !! ")
                sent.append(word)
                sent_all_sequence.append(word_all_sequence)
                words_count += 1
            # we want to remove gold empty words (coming from the sentence level padding)
            #print("Word to print empty ", len(word_to_print), word_to_print, empty_decoded_word)
            if len(word_to_print) > 0 or empty_decoded_word:
                word_str_decoded.append(word_to_print)
        str_decoded.append(word_str_decoded)
        decoding.append(sent)
        decoding_all_sequences.append(sent_all_sequence)
        #print("FINAL", sent, word_i)
    # NB : former single_sequence have no impact on output
    if single_sequence:
        # for interactive mode : as batch_size == 2 not supported we have to decode with batch_size 2 and then only keeping first
        decoding = decoding[0]
        str_decoded = str_decoded[0]
        decoding_all_sequences = decoding_all_sequences[0]
    if output_str:
        _out = str_decoded
    else:
        _out = decoding
    if last:
        if debug:
            pdb.set_trace()
    return words_count, _out, decoding_all_sequences

