import torch
from torch.autograd import Variable
from io_.info_print import printing
from io_.from_array_to_text import output_text, output_text_
import numpy as np
from evaluate.normalization_errors import score_ls, score_ls_
from io_.dat.constants import CHAR_START_ID
import pdb
# EPSILON for the test of edit distance 
EPSILON = 0.000001
TEST_SCORING_IN_CODE = False


def _init_metric_report(score_to_compute_ls, mode_norm_score_ls):
    if score_to_compute_ls is not None:
        dic = {score+"-"+norm_mode: 0 for score in score_to_compute_ls for norm_mode in mode_norm_score_ls}
        dic.update({score+"-"+norm_mode+"-"+"total_tokens": 0 for score in score_to_compute_ls for norm_mode in mode_norm_score_ls})
        dic.update({score+"-"+norm_mode+"-"+"mean_per_sent": 0 for score in score_to_compute_ls for norm_mode in mode_norm_score_ls})
        dic.update({score + "-" + norm_mode + "-" + "n_word_per_sent": 0 for score in score_to_compute_ls for norm_mode in mode_norm_score_ls})
        return dic
    return None


def greedy_decode_batch(batchIter, model,char_dictionary, batch_size, pad=1,
                        gold_output=False, score_to_compute_ls=None, stat=None,
                        use_gpu=False,
                        compute_mean_score_per_sent=False,
                        mode_norm_score_ls=None,
                        verbose=0):

        score_dic = _init_metric_report(score_to_compute_ls, mode_norm_score_ls)
        total_count = {"src_word_count": 0,
                       "target_word_count": 0,
                       "pred_word_count": 0}
        if mode_norm_score_ls is None:
            mode_norm_score_ls = ["all"]
        assert len(set(mode_norm_score_ls) & set(["all", "NEED_NORM", "NORMED"])) >0
        with torch.no_grad():
            for step, batch in enumerate(batchIter):
                # read src sequence
                src_seq = batch.input_seq
                src_len = batch.input_seq_len
                src_mask = batch.input_seq_mask
                target_gold = batch.output_seq if gold_output else None
                # do something with it : When do you stop decoding ?
                max_len = src_seq.size(-1)
                printing("WARNING : word max_len set to src_seq.size(-1) {} ", var=(max_len), verbose=verbose,
                         verbose_level=3)
                # decoding one batch
                text_decoded_ls, src_text_ls, gold_text_seq_ls, counts = decode_sequence(model=model,
                                                                                 char_dictionary=char_dictionary,
                                                                                 single_sequence=False,
                                                                                 target_seq_gold=target_gold,
                                                                                 use_gpu=use_gpu,
                                                                                 max_len=max_len, src_seq=src_seq,
                                                                                 src_mask=src_mask,src_len=src_len,
                                                                                 batch_size=batch_size, pad=pad,
                                                                                 verbose=verbose)
                total_count["src_word_count"] += counts["src_word_count"]
                total_count["target_word_count"] += counts["target_word_count"]
                total_count["pred_word_count"] += counts["pred_word_count"]
                printing("Source text {} ", var=[(src_text_ls)], verbose=verbose, verbose_level=5)
                printing("Prediction {} ", var=[(text_decoded_ls)], verbose=verbose, verbose_level=5)
                #scores_ls_func = "score_ls_"
                if gold_output:
                    printing("Gold {} ", var=[(gold_text_seq_ls)], verbose=verbose, verbose_level=5)
                    if score_to_compute_ls is not None:
                        for mode_norm_score in mode_norm_score_ls:
                            #print("Scoring with mode {}".format(mode_norm_score))
                            for metric in score_to_compute_ls:
                                #we score the batch
                                _score, _n_tokens = score_ls_(text_decoded_ls, gold_text_seq_ls, ls_original=src_text_ls,
                                                              score=metric, stat=stat,
                                                              compute_mean_score_per_sent=compute_mean_score_per_sent,
                                                              normalized_mode=mode_norm_score,
                                                              verbose=verbose)
                                #_score = _score["sum"]
                                score_dic[metric + "-" + mode_norm_score + "-n_word_per_sent"] += _score["n_word_per_sent"] if compute_mean_score_per_sent else 0
                                score_dic[metric + "-" + mode_norm_score+"-mean_per_sent"] += _score["mean_per_sent"] if compute_mean_score_per_sent else 0
                                score_dic[metric+"-"+mode_norm_score] += _score["sum"]
                                score_dic[metric+"-"+mode_norm_score+"-"+"total_tokens"] += _n_tokens
                    test_scoring = TEST_SCORING_IN_CODE
                    if test_scoring:
                        assert len(list((set(mode_norm_score_ls)&set(["NEED_NORM", "NORMED","all"])))) == 3, "ERROR : to perform test need all normalization mode "
                            #print("Scoring with mode {}".format(mode_norm_score))
                        for metric in score_to_compute_ls:
                            assert score_dic[metric + "-NEED_NORM-total_tokens"]+score_dic[metric + "-NORMED-total_tokens"] == score_dic[metric + "-all-total_tokens"], \
                                'ERROR all-total_tokens is {}  not equal to NEED NORMED {} +  NORMED {} '.format(score_dic[metric + "-all-total_tokens"], score_dic[metric + "-NEED_NORM-total_tokens"], score_dic[metric + "-NORMED-total_tokens"])
                            assert np.abs(score_dic[metric + "-NEED_NORM"]+score_dic[metric + "-NORMED"] - score_dic[metric + "-all"]) < EPSILON, \
                            "ERROR : correct NEED_NORM {} , NORMED {} and all {} ".format(score_dic[metric + "-NEED_NORM"], score_dic[metric + "-NORMED"], score_dic[metric + "-all"])
                            print("TEST PASSED")
            assert total_count["src_word_count"] == total_count["target_word_count"], \
                "ERROR src_word_count {} vs target_word_count {}".format(total_count["src_word_count"], total_count["target_word_count"])
            assert total_count["src_word_count"] == total_count["pred_word_count"], \
                "ERROR src_word_count {} vs pred_word_count {}".format(total_count["src_word_count"], total_count["pred_word_count"])
            printing("Assertion passed : there are as many words in the source side,"
                     "the target side and"
                     "the predicted side : {} ".format(total_count["src_word_count"]), verbose_level=0, verbose=verbose)
            return score_dic


def decode_sequence(model, char_dictionary, max_len, src_seq, src_mask, src_len,
                    batch_size, pad=1, target_seq_gold=None,
                    use_gpu=False,
                    single_sequence=False, verbose=2):

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
    printing("WARNING : output_str = True hardcoded", verbose=verbose, verbose_level=0)
    printing("Data output sizes ", var=(output_seq.size(), output_len.size(), output_mask.size()), verbose=verbose, verbose_level=6)
    for step, char_decode in enumerate(range(2,  max_len)):
        if use_gpu:
            src_seq = src_seq.cuda()
            output_seq = output_seq.cuda()
            src_len = src_len.cuda()
            output_len = output_len.cuda()
        decoding_states = model.forward(input_seq=src_seq,
                                        output_seq=output_seq,
                                        input_word_len=src_len,
                                        output_word_len=output_len)
        # we remove in src_seq the empty words
        #src_seq = src_seq[:,:decoding_states.size(1),:]
        # [batch, seq_len, V]
        scores = model.generator.forward(x=decoding_states)
        # each time step predict the most likely
        # len
        # output_len defined based on src_len to remove empty words
        output_len = (src_len[:, :, 0] != 0).unsqueeze(dim=2)*char_decode
        printing("DECODER step {} output len {} ", var=(step, output_len), verbose=verbose, verbose_level=3)
        #output_len[:] = char_decode # before debugging
        # mask
        output_mask = np.ones(src_seq.size(), dtype=np.int64)
        output_mask[:, char_decode:] = 0
        #output_mask = Variable(torch.from_numpy(output_mask), requires_grad=False)
        # new seq

        predictions = scores.argmax(dim=-1)

        printing("Prediction size {} ", var=(predictions.size()), verbose=verbose, verbose_level=4)
        printing("Prediction {} ", var=(predictions), verbose=verbose, verbose_level=5)
        printing("scores: {} scores {} scores sized  {} predicion size {} prediction {} outputseq ", var=(scores,
                 scores.size(),
                 predictions.size(),
                 predictions[:, -1],
                 output_seq.size()),
                 verbose=verbose, verbose_level=5)

        output_seq = output_seq[:, :scores.size(1), :]
        output_seq[:, :, char_decode - 1] = predictions[:, :, -1]
        if verbose >= 5:
            sequence = [" ".join([char_dictionary.get_instance(output_seq[sent, word_ind, char_i]) for char_i in range(max_len)])
                        + "|sent-{}|".format(sent) for sent in range(output_seq.size(0)) for word_ind in range(output_seq.size(1))]
        else:
            sequence = []
        printing("Decoding step {} decoded target {} ", var=(step, sequence), verbose=verbose, verbose_level=5)
        pred_word_count, text_decoded = output_text_(output_seq,#predictions,
                                                    char_dictionary, single_sequence=single_sequence, output_str=output_str)
        printing("PREDICTION : array text {} ", var=[text_decoded],
                 verbose=verbose,
                 verbose_level=5)

    #text_ = "output_text_"
    src_word_count, src_text = output_text_(src_seq, char_dictionary, single_sequence=single_sequence,output_str=output_str)
    src_text_ls.extend(src_text)
    if target_seq_gold is not None:
        target_word_count, target_text = output_text_(target_seq_gold, char_dictionary, single_sequence=single_sequence,output_str=output_str)
        target_seq_gold_ls.extend(target_text)
    else:
        target_word_count = None

    return text_decoded, src_text_ls, target_seq_gold_ls, {"src_word_count": src_word_count,
                                                           "target_word_count": target_word_count,
                                                           "pred_word_count": pred_word_count}


def decode_seq_str(seq_string, model, char_dictionary, pad=1,
                   max_len=20, verbose=2, sent_mode=False):
    sent = seq_string.copy()
    # we add empty words at the end otherwie poblem !! # TODO : understand why ? is it because we need word padded at the end of the sentence ?
    sent.append("")
    with torch.no_grad():
        sent_character = []
        sent_words_mask = []
        sent_words_lens = []
        for seq_string in sent:
            if len(seq_string)>0:
                _seq_string = ["_START"]
                printing("WARNING : we added _START symbol and _END_CHAR ! ", verbose=verbose, verbose_level=0)
                _seq_string.extend(list(seq_string))
                seq_string = _seq_string + ["_END_CHAR"] #["_END_CHAR"]#["_PAD_CHAR"]
            if len(seq_string) > max_len:
                # cutting to respect dim requirements
                seq_string = seq_string[:max_len-1]+["_PAD_CHAR"]
            printing("INPUT SEQ is {} ", var=[seq_string], verbose=verbose, verbose_level=2)
            sequence_characters = [char_dictionary.get_index(letter) for letter in seq_string]+[pad for _ in range(max_len-len(seq_string))]
            sent_character.append(sequence_characters)
            masks = [1 for _ in seq_string]+[0 for _ in range(max_len-len(seq_string))]
            sent_words_mask.append(masks)
            words_lens = min(max_len, len(seq_string))
            sent_words_lens.append(words_lens)
            # we have to create batch_size == 2 because of bug
            if False:
                sequence_characters = Variable(torch.from_numpy(np.array([sequence_characters, sequence_characters])), requires_grad=False)

                char_seq = sequence_characters.unsqueeze(dim=1)
                #char_seq = Variable(torch.from_numpy(np.array([sequence_characters, sequence_characters])), requires_grad=False)
                char_mask = Variable(torch.from_numpy(np.array([masks, masks])), requires_grad=False)
                char_mask = char_mask.unsqueeze(dim=1)
                char_len = Variable(torch.from_numpy(np.array([[min(max_len, len(seq_string)), 0],
                                                               [min(max_len, len(seq_string)), 0]])))
                char_len = char_len.unsqueeze(dim=2)
        batch = Variable(torch.from_numpy(np.array([sent_character, sent_character])),
                                       requires_grad=False)
        batch_masks = Variable(torch.from_numpy(np.array([sent_words_mask, sent_words_mask])), requires_grad=False)
        batch_lens = Variable(torch.from_numpy(np.array([sent_words_lens, sent_words_lens])), requires_grad=False)
        batch_lens = batch_lens.unsqueeze(dim=2)
        batch_size = 2
        text_decoded, src_text, target, _ = decode_sequence(model=model, char_dictionary=char_dictionary,
                                                            max_len=max_len,batch_size=batch_size,
                                                            src_seq=batch, src_len=batch_lens,
                                                            src_mask=batch_masks, single_sequence=True,
                                                            pad=pad, verbose=verbose)

        printing("DECODED text is : {} original is {}".format(text_decoded, src_text), verbose_level=0, verbose=verbose)


def decode_interacively(model , char_dictionary,  max_len, pad=1, sent_mode=False,verbose=0):
    if char_dictionary is None:
        printing("INFO : dictionary is None so setting char_dictionary to model.char_dictionary",
                 verbose=verbose, verbose_level=0)
        char_dictionary = model.char_dictionary
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
                decode_seq_str(seq_string=sentence, model=model, char_dictionary=char_dictionary, pad=pad,max_len= max_len,
                               verbose=verbose, sent_mode=True)
                sentence = []

        elif seq_string == "END":
            printing("ENDING INTERACTION", verbose=verbose, verbose_level=0)
            break
        else:
            sentence.append(seq_string)
            if not sent_mode:
                decode_seq_str(seq_string, model, char_dictionary, pad, max_len, verbose)