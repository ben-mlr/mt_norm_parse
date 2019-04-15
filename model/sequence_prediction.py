from env.importing import *

from io_.info_print import printing
from io_.from_array_to_text import output_text, output_text_
from io_.dat.constants import PAD_ID_CHAR, CHAR_END_ID
from toolbox.sanity_check import get_timing, sanity_check_model_pred
# EPSILON for the test of edit distance

EPSILON = 0.000001
TEST_SCORING_IN_CODE = False


def decode_word(model, src_seq, src_len,
                pad=1, target_word_gold=None,
                use_gpu=False, target_pos_gold=None,
                mode="word", input_word=None,
                single_sequence=False, verbose=2):
    """
    NB : could be more factorized (its exactly the same prediction the only difference is the dictionary
    """
    _, word_pred, pos_pred, norm_not_norm, edit_pred, _, _ = model.forward(input_seq=src_seq,
                                                                input_word_len=src_len,
                                                                word_embed_input=input_word,
                                                                word_level_predict=True)
    sanity_check_model_pred(mode=mode, pos_pred=pos_pred, word_pred=word_pred, norm_not_norm=norm_not_norm)

    pred_norm_not_norm = None
    src_word_input = None
    target_word_gold_text = None
    words_count_pred = 0
    words_count_gold = 0
    text_decoded = None
    print("DECODING WITH ", mode)
    if mode in ["word", "norm_not_norm"]:
        if target_word_gold is None:
            print("INFO decoding with no gold reference")
        # PREDICT
        if word_pred is not None:
            prediction = word_pred.argmax(dim=-1)
            # we trust the predictor to do the padding !
            src_seq = src_seq[:, :prediction.size(1)]
        if norm_not_norm is not None:
            pred_norm_not_norm = norm_not_norm.argmax(dim=-1)

        # HANDLE PADDING
        # should not be mandatory right ?
        if target_word_gold is not None and word_pred is not None:
            target_word_gold = target_word_gold[:, :prediction.size(1)]
        if pred_norm_not_norm is not None:
            pred_norm_not_norm = pred_norm_not_norm[:, :src_seq.size(1)]  # followign what's done above

        # GET SRC CHARACTERS AS STRING
        src_word_count, src_text, src_all_ls = output_text_(src_seq, model.char_dictionary,
                                                            single_sequence=single_sequence, debug=False,
                                                            output_str=True)
        # GET SRC WORD AS STRING
        if input_word is not None:
            words_embed_count_src, src_word_input, _ = output_text_(input_word,
                                                                    word_decode=True,
                                                                    word_dic=model.word_dictionary,
                                                                    # model.word_nom_dictionary,
                                                                    debug=False, single_sequence=single_sequence,
                                                                    char_decode=False, output_str=True)
        if word_pred is not None:
            words_count_pred, text_decoded, _ = output_text_(prediction, word_decode=True, word_dic=model.word_dictionary,#word_nom_dictionary,
                                                         single_sequence=single_sequence, char_decode=False,
                                                         debug=False,
                                                         output_str=True)
        # GET GOLD TARGET AS STRING
        if target_word_gold is not None:
            assert model.word_nom_dictionary is not None, "ERROR : word_nom_dictionary is required"
            words_count_gold, target_word_gold_text, _ = output_text_(target_word_gold,#input_word,#target_word_gold,
                                                                      word_decode=True,
                                                                      word_dic=model.word_nom_dictionary,
                                                                      debug=False,
                                                                      showing_attention=False,
                                                                      single_sequence=single_sequence, char_decode=False,
                                                                      output_str=True)

        # fix by hand  # TODO : check if it is correct
        # its based on src_text sequence length because
        # we assumed word to word mapping and we want to predict without gold
        if word_pred is not None:
            text_decoded = [sent[:len(ls_gold)] for sent, ls_gold in zip(text_decoded, src_text)]
            words_count_pred = sum([len(sent) for sent in text_decoded])
            printing("PRED : array text {} ", var=[text_decoded], verbose=verbose, verbose_level=5)

        printing("GOLD : array text {} ", var=[target_word_gold_text], verbose=verbose, verbose_level=5)
        printing("SRC : array text {} ", var=[src_text], verbose=verbose, verbose_level=5)

    #TODO : should be factorized with above
    elif mode == "pos":
        assert target_word_gold is None, "Only target_pos_gold should be provided"
        #assert target_pos_gold is not None
        prediction_pos = pos_pred.argmax(dim=-1)
        src_seq = src_seq[:, :prediction_pos.size(1)]
        if target_pos_gold is not None:

            target_pos_gold = target_pos_gold[:, :prediction_pos.size(1)]

        src_word_count, src_text, src_all_ls = output_text_(src_seq, model.char_dictionary,
                                                            single_sequence=single_sequence,debug=False,
                                                            output_str=True)
        words_count_pred, text_decoded, _ = output_text_(prediction_pos, word_decode=True,
                                                         word_dic=model.pos_dictionary,
                                                         single_sequence=single_sequence, char_decode=False,
                                                         debug=False,
                                                         output_str=True)
        if target_pos_gold is not None:

            words_count_gold, target_word_gold_text, _ = output_text_(target_pos_gold, word_decode=True,
                                                                      word_dic=model.pos_dictionary,
                                                                      debug=False, single_sequence=single_sequence,
                                                                      char_decode=False, output_str=True)
            text_decoded = [sent[:len(ls_gold)] for sent, ls_gold in zip(text_decoded, src_text)]
            words_count_pred = sum([len(sent) for sent in text_decoded])

    if single_sequence:
        if pred_norm_not_norm is not None:
            pred_norm_not_norm = pred_norm_not_norm[0]

    retu = OrderedDict([("text", OrderedDict([("pred", text_decoded), ("src_as_chars",src_text), ("src_as_words",src_word_input),("gold",target_word_gold)])),
                        ("count", OrderedDict(
                            [("src_word_count", src_word_count), ("pred_word_count", words_count_pred),    ("target_word_count", words_count_gold)])),
                        ("1hot", OrderedDict([("pred_norm_not_norm", pred_norm_not_norm), ("src_seq", src_seq), ("target_word_gold",target_word_gold)]))
                        ])

    return (text_decoded, src_text, target_word_gold_text, src_word_input), \
            {"src_word_count": src_word_count, "target_word_count": words_count_gold,
            "pred_word_count": words_count_pred}, \
            (None, None,), \
           (pred_norm_not_norm, None, src_seq, target_word_gold)


def decode_sequence(model, char_dictionary, max_len, src_seq, src_mask, src_len,
                    pad=PAD_ID_CHAR, target_seq_gold=None, input_word=None,
                    use_gpu=False, showing_attention=False,
                    single_sequence=False, eval_time=True, verbose=2,
                    timing=False):

    #eval_time alays True for now
    printing("EVAL TIME is {}", var=eval_time, verbose=verbose, verbose_level=2)
    output_seq = pad*np.ones(src_seq.size(), dtype=np.int64)
    # we start with the _START symbol
    output_seq[:, :, 0] = src_seq[:, :, 0] #CHAR_START_ID
    src_text_ls = []
    target_seq_gold_ls = [] if target_seq_gold is not None else None
    output_mask = np.ones(src_mask.size(), dtype=np.int64)
    output_mask[:, :, 1:] = 0
    output_len = Variable(torch.from_numpy(np.ones((src_seq.size(0), src_seq.size(1), 1), dtype=np.int64)),
                          requires_grad=False)
    output_mask = Variable(torch.from_numpy(output_mask), requires_grad=False)
    output_seq = Variable(torch.from_numpy(output_seq), requires_grad=False)
    printing("Data Start source {} {} ", var=(src_seq, src_seq.size()), verbose=verbose, verbose_level=5)
    output_str = True
    printing("WARNING : output_str = True hardcoded (decode_sequence)", verbose=verbose, verbose_level=2)
    printing("Data output sizes ", var=(output_seq.size(), output_len.size(), output_mask.size()), verbose=verbose, verbose_level=6)
    start_decode_sequence = time.time() if timing else None

    for step, char_decode in enumerate(range(2,  max_len)):
        if use_gpu:
            src_seq = src_seq.cuda()
            output_seq = output_seq.cuda()
            src_len = src_len.cuda()
            output_len = output_len.cuda()
        start = time.time() if timing else None

        pdb.set_trace()
        decoding_states, word_pred, pos_pred, norm_not_norm, edit_pred, attention, _ \
            = model.forward(input_seq=src_seq,
                            output_seq=output_seq,
                            input_word_len=src_len,
                            output_word_len=output_len,
                            word_embed_input=input_word)

        output_len = (src_len[:, :, 0] != 0).unsqueeze(dim=2) * char_decode
        printing("DECODER step {} output len {} ", var=(step, output_len), verbose=verbose, verbose_level=3)

        time_forward, start = get_timing(start)
        # [batch, seq_len, V]

        pred_norm_not_norm = norm_not_norm.argmax(dim=-1) if norm_not_norm is not None else None
        scores = model.generator.forward(x=decoding_states)

        time_generate, start = get_timing(start)
        # each time step predict the most likely
        # len
        # output_len defined based on src_len to remove empty words
        #output_len[:] = char_decode # before debugging
        # mask
        output_mask = np.ones(src_seq.size(), dtype=np.int64)
        output_mask[:, char_decode:] = 0
        # new seq
        predictions = scores.argmax(dim=-1)

        time_argmax_printing, start = get_timing(start)
        if verbose >= 0:
            # .size() takes some time
            printing("Prediction size {} ", var=(predictions.size()), verbose=verbose, verbose_level=0)
            printing("SCORES {} ", var=[str(scores)], verbose=verbose, verbose_level=0)
            printing("Prediction {} ", var=[predictions], verbose=verbose, verbose_level=0)

            printing("scores: (1st batch)  {} scores sized  {} \n predicion size {} prediction {} ",
                     var=[scores[0, :, :, :], scores.size(),
                          predictions.size(), predictions[0,:, -1],],
                     verbose=verbose, verbose_level=0)
        time_printing, start = get_timing(start)

        output_seq = output_seq[:, :scores.size(1), :]
        time_output_seq, start = get_timing(start)
        if pred_norm_not_norm is not None:
            pred_norm_not_norm = pred_norm_not_norm[:, :scores.size(1)]  # followign what's done above

        output_seq[:, :, char_decode - 1] = predictions[:, :, -1]

        if verbose >= 0:
            sequence = [" ".join([char_dictionary.get_instance(output_seq[sent, word_ind, char_i]) for char_i in range(max_len)])
                        + "|sent-{}|".format(sent) for sent in range(output_seq.size(0)) for word_ind in range(output_seq.size(1))]
        else:
            sequence = []

        printing("Decoding step {} decoded target {} ", var=(step, sequence), verbose=verbose, verbose_level=0)
        time_sequence_text, start = get_timing(start)
        printing("DECODING scores {}", var=[scores[0]], verbose=verbose, verbose_level=0)
        printing("DECODING decoding_states {}", var=[decoding_states[0]], verbose=verbose, verbose_level=0)

        if eval_time:
            # at test time : if all prediction in the batch are whether PAD symbol or END symbol : we break
            if ((predictions[:, :, -1] == PAD_ID_CHAR) + (predictions[:, :, -1] == CHAR_END_ID)).all():
                printing("PREDICTION IS ONLY PAD or END SYMBOL SO BREAKING DECODING", verbose=verbose, verbose_level=1)
                break
    # no need to do that in the loop
    print("WARNING : shfited output sequence of one character not to output START token")
    pred_word_count, text_decoded, decoded_ls = output_text_(output_seq[:, :, 1:],
                                                             char_dictionary, single_sequence=single_sequence,
                                                             output_str=output_str, output_len=output_len,
                                                             last=(char_decode == (max_len-1)),
                                                             showing_attention=showing_attention,
                                                             debug=False)

    time_output_text, start = get_timing(start)

    time_decoding_all_seq, start  = get_timing(start_decode_sequence)
    printing("PREDICTION : array text {} ", var=[text_decoded], verbose=verbose, verbose_level=5)
    src_word_count, src_text, src_all_ls = output_text_(src_seq, char_dictionary, single_sequence=single_sequence,
                                                        showing_attention=showing_attention,output_str=output_str)
    printing("SOURCE  : array text {} ", var=[src_text], verbose=verbose, verbose_level=5)
    src_text_ls.extend(src_text)
    if target_seq_gold is not None:
        target_word_count, target_text, _ = output_text_(target_seq_gold, char_dictionary,
                                                         showing_attention=showing_attention,
                                                         single_sequence=single_sequence, output_str=output_str)
        target_seq_gold_ls.extend(target_text)
        printing("GOLD : array text {} ", var=[target_text], verbose=verbose, verbose_level=5)
    else:
        target_word_count = None
    if single_sequence:
        if model.decoder.attn_layer is not None:
            attention = attention[0]
        if pred_norm_not_norm is not None:
            pred_norm_not_norm = pred_norm_not_norm[0]
    if timing:
        print("DECODING TIME : {}".format(
            OrderedDict([("time_decoding_all_seq", time_decoding_all_seq),
                         ("time_forward", time_forward),
                         ("time_generate", time_generate),
                         ("time_argmax_printing", time_argmax_printing),
                         ("time_printing", time_printing),
                         ("time_output_seq", time_output_seq),
                         ("time_sequence_text", time_sequence_text),
                         ("time_output_text",time_output_text),
                         ("time_decoding_all_seq",time_decoding_all_seq)
                         ])))

    return (text_decoded, src_text_ls, target_seq_gold_ls, None), \
           {
           "src_word_count": src_word_count,
           "target_word_count": target_word_count,
           "pred_word_count": pred_word_count},\
           (attention, src_all_ls,), \
           (pred_norm_not_norm, output_seq, src_seq, target_seq_gold)


