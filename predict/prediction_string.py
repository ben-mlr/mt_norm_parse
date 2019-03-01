import torch
from torch.autograd import Variable
from io_.info_print import printing
import numpy as np
import pdb
from model.sequence_prediction import decode_word, decode_sequence
from evaluate.visualize_attention import show_attention
from toolbox.norm_not_norm import get_label_norm
from io_.dat.constants import PAD, ROOT, END, ROOT_CHAR, END_CHAR
# EPSILON for the test of edit distance
EPSILON = 0.000001
TEST_SCORING_IN_CODE = False


def decode_interacively(model, char_dictionary,  max_len, pad=1, sent_mode=False, save_attention=False,
                        show_attention=False, beam_decode=False,beam_size=None,
                        dir_attention=None, verbose=0):
    if char_dictionary is None:
        printing("INFO : dictionary is None so setting char_dictionary to model.char_dictionary",
                 verbose=verbose, verbose_level=0)
        char_dictionary = model.char_dictionary
    if model.arguments["hyperparameters"]["encoder_arch"].get("word_embed",False):
        word_dictionary = model.word_dictionary
    else:
        word_dictionary = None
    sentence = []
    while True:
        seq_string = input("Please type what you want to normalize word by word and "
                           "finishes by 'stop' ? to end type : 'END'    ")
        if seq_string == "":
            continue
        if seq_string == "stop":
            if not sent_mode:

                break
            else:
                decode_seq_str(seq_string=sentence, model=model, char_dictionary=char_dictionary, pad=pad, max_len= max_len,
                               show_att=show_attention, beam_decode=beam_decode,beam_size=beam_size,
                               word_dictionary=word_dictionary,
                               verbose=verbose, sent_mode=True, dir_attention=dir_attention, save_attention=save_attention)
                sentence = []
        elif seq_string == "END":
            printing("ENDING INTERACTION", verbose=verbose, verbose_level=0)
            break
        else:
            sentence.append(seq_string)
            if not sent_mode:
                decode_seq_str(seq_string, model, char_dictionary, pad, max_len, verbose)


def decode_seq_str(seq_string, model, char_dictionary, pad=1,
                   dir_attention=None, save_attention=False,
                   show_att=False, beam_decode=False,beam_size=None,
                   word_dictionary=None,
                   max_len=20, verbose=2, sent_mode=False):
    assert sent_mode
    sent = seq_string.copy()
    # we add empty words at the end otherwie poblem !! # TODO : understand why ? is it because we need word padded at the end of the sentence ?
    #sent.append("")
    with torch.no_grad():
        sent_character = []
        sent_words_mask = []
        sent_words_lens = []
        word_ls = [] if word_dictionary is not None else None # word_norm_dictionary is the signal for using word at input
        print("sent", sent)
        sent = [ROOT]+sent+[END]
        for seq_string in sent:
            # should be padded in the same way as done in the training data conll_reader
            pdb.set_trace()
            if len(seq_string) > 0:
                word_string = seq_string[:]
                if seq_string == ROOT:
                    seq_string = ROOT_CHAR
                _seq_string = ["_START"]
                printing("WARNING : we added _START symbol and _END_CHAR ! ", verbose=verbose, verbose_level=2)
                if seq_string not in [ROOT, END, ROOT_CHAR]:
                    _seq_string.extend(list(seq_string))
                else:
                    _seq_string.append(seq_string)
                seq_string = _seq_string + ["_END_CHAR"] #["_END_CHAR"]#["_PAD_CHAR"]
            if len(seq_string) > max_len:
                # cutting to respect dim requirements
                seq_string = seq_string[:max_len-1]+["_PAD_CHAR"]
            if len(seq_string) > 0:
                printing("INPUT SEQ is {} ", var=[seq_string], verbose=verbose, verbose_level=2)
            word_id = word_dictionary.get_index(word_string) if word_dictionary is not None else None
            sequence_characters = [char_dictionary.get_index(letter) for letter in seq_string]+[pad for _ in range(max_len-len(seq_string))]
            sent_character.append(sequence_characters)
            pdb.set_trace()
            if word_ls is not None:
                word_ls.append(word_id)
            masks = [1 for _ in seq_string]+[0 for _ in range(max_len-len(seq_string))]
            sent_words_mask.append(masks)
            words_lens = min(max_len, len(seq_string))
            sent_words_lens.append(words_lens)
            # we have to create batch_size == 2 because of bug
        pdb.set_trace()
        batch = Variable(torch.from_numpy(np.array([sent_character, sent_character])), requires_grad=False)
        input_word = Variable(torch.from_numpy(np.array([word_ls, word_ls])), requires_grad=False) if word_ls is not None else None
        batch_masks = Variable(torch.from_numpy(np.array([sent_words_mask, sent_words_mask])), requires_grad=False)
        batch_lens = Variable(torch.from_numpy(np.array([sent_words_lens, sent_words_lens])), requires_grad=False)
        batch_lens = batch_lens.unsqueeze(dim=2)
        if beam_decode:
            decode_sequence_beam(model=model, char_dictionary=char_dictionary,
                                 max_len=max_len, src_seq=batch, src_len=batch_lens,beam_size=beam_size,
                                 src_mask=batch_masks, pad=pad,
                                 verbose=verbose)
        else:
            pdb.set_trace()
            if model.arguments["hyperparameters"]["decoder_arch"].get("char_decoding", True):
                assert not model.arguments["hyperparameters"]["decoder_arch"].get("word_decoding", False), "ERROR : only on type of decoding should be set (for now)"
                (text_decoded, src_text, target, src_words_from_embed), _, (attention, src_seq), (pred_norm,_, _, _)  \
                    = decode_sequence(model=model, char_dictionary=char_dictionary,
                                      max_len=max_len, src_seq=batch, src_len=batch_lens, input_word=input_word,
                                      src_mask=batch_masks, single_sequence=True, pad=pad, verbose=verbose)
            elif model.arguments["hyperparameters"]["decoder_arch"].get("word_decoding", False):
                pdb.set_trace()
                (text_decoded, src_text, target, src_words_from_embed), counts, (attention, src_seq), \
                (pred_norm, output_seq_n_hot, src_seq, target_seq_gold) = decode_word(model,
                                                                                      src_seq=batch,
                                                                                      src_len=batch_lens,
                                                                                      input_word=input_word,
                                                                                      single_sequence=True,
                                                                                      target_word_gold=None)
        if attention is not None:
            print("Attention", attention, src_seq, text_decoded)
            for pred_word, src_word, attention_word in zip(text_decoded, src_seq, attention):

                show_attention(list(pred_word), src_word[:attention_word.size(1)],
                               attention_word.transpose(1, 0), save=save_attention, dir_save=dir_attention,show=show_att,
                               model_full_name=model.model_full_name)
            #show_attention("[lekfezlfkh efj ", ["se", "mjfsemkfj"], torch.tensor([[0, .4], [1, 0.6]]))
        if pred_norm is not None:
            pdb.set_trace()
            norm_not_norm_seq = [(get_label_norm(norm), word) for norm, word in zip(pred_norm, src_text)]
            printing("NORMALIZING : {} ", var=[norm_not_norm_seq], verbose_level=0, verbose=0)
        printing("DECODED text is : {} original is {} and {} seen as word embed ",var=(text_decoded, src_text, src_words_from_embed), verbose_level=0, verbose=0)
