import torch
from torch.autograd import Variable
from io_.info_print import printing
from io_.dat.normalized_writer import write_normalization
from io_.from_array_to_text import output_text, output_text_
import numpy as np
from evaluate.normalization_errors import score_norm_not_norm
from evaluate.normalization_errors import score_ls_, score_ls_2
from env.project_variables import WRITING_DIR
from io_.dat.constants import CHAR_START_ID
import pdb
import os
from evaluate.visualize_attention import show_attention
from toolbox.norm_not_norm import get_label_norm
# EPSILON for the test of edit distance 
EPSILON = 0.000001
TEST_SCORING_IN_CODE = False


def _init_metric_report(score_to_compute_ls, mode_norm_score_ls):
    if score_to_compute_ls is not None:
        dic = {score+"-"+norm_mode: 0 for score in score_to_compute_ls for norm_mode in mode_norm_score_ls}
        dic.update({score + "-" + norm_mode + "-" + "n_sents": 0 for score in score_to_compute_ls for norm_mode in
                    mode_norm_score_ls})
        dic.update({score+"-"+norm_mode+"-"+"total_tokens": 0 for score in score_to_compute_ls for norm_mode in mode_norm_score_ls})
        dic.update({score+"-"+norm_mode+"-"+"mean_per_sent": 0 for score in score_to_compute_ls for norm_mode in mode_norm_score_ls})
        dic.update({score + "-" + norm_mode + "-" + "n_word_per_sent": 0 for score in score_to_compute_ls for norm_mode in mode_norm_score_ls})
        dic.update({token + "-norm_not_norm-" + type_ + "-count":0 for token in ["all","need_norm"] for type_ in ["pred", "gold"]})
        dic["all-norm_not_norm-pred_correct-count"] = 0
        dic["need_norm-norm_not_norm-pred_correct-count"] = 0
        return dic
    return None


def greedy_decode_batch(batchIter, model,char_dictionary, batch_size, pad=1,
                        gold_output=False, score_to_compute_ls=None, stat=None,
                        use_gpu=False,
                        compute_mean_score_per_sent=False,
                        mode_norm_score_ls=None,
                        label_data=None,
                        write_output=False, write_to="conll", dir_normalized=None, dir_original=None,
                        verbose=0):

        score_dic = _init_metric_report(score_to_compute_ls, mode_norm_score_ls)
        total_count = {"src_word_count": 0,
                       "target_word_count": 0,
                       "pred_word_count": 0}
        if mode_norm_score_ls is None:
            mode_norm_score_ls = ["all"]
        assert len(set(mode_norm_score_ls) & set(["all", "NEED_NORM", "NORMED"])) >0
        with torch.no_grad():
            for step, (batch, _) in enumerate(batchIter):
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
                (text_decoded_ls, src_text_ls, gold_text_seq_ls), counts, _, (pred_norm, output_seq_n_hot, src_seq, target_seq_gold) \
                    = decode_sequence(model=model,
                                      char_dictionary=char_dictionary,
                                      single_sequence=False,
                                      target_seq_gold=target_gold,
                                      use_gpu=use_gpu,
                                      max_len=max_len,
                                      src_seq=src_seq,
                                      src_mask=src_mask,
                                      src_len=src_len,
                                      pad=pad,
                                      verbose=verbose)
                if write_output:
                    if dir_normalized is None:
                        dir_normalized = os.path.join(WRITING_DIR, model.model_full_name+"-{}-normalized.conll".format(label_data))
                        dir_original = os.path.join(WRITING_DIR, model.model_full_name+"-{}-original.conll".format(label_data))

                    write_normalization(format=write_to, dir_normalized=dir_normalized, dir_original=dir_original,
                                        text_decoded_ls=text_decoded_ls, src_text_ls=src_text_ls, verbose=verbose)

                total_count["src_word_count"] += counts["src_word_count"]
                total_count["pred_word_count"] += counts["pred_word_count"]
                printing("Source text {} ", var=[(src_text_ls)], verbose=verbose, verbose_level=5)
                printing("Prediction {} ", var=[(text_decoded_ls)], verbose=verbose, verbose_level=5)
                if gold_output:
                    total_count["target_word_count"] += counts["target_word_count"]
                    # we can score
                    printing("Gold {} ", var=[(gold_text_seq_ls)], verbose=verbose, verbose_level=5)
                    if score_to_compute_ls is not None:
                        try:
                            _score_2 = score_ls_2(ls_pred=text_decoded_ls, ls_gold=gold_text_seq_ls,
                                                  output_seq_n_hot=output_seq_n_hot, src_seq=src_seq, target_seq_gold=target_seq_gold,
                                                  pred_norm_not_norm=pred_norm, gold_norm_not_norm=batch.output_norm_not_norm,
                                                  ls_original=src_text_ls)
                            if batch.output_norm_not_norm is not None and False :
                                for token in ["all", "need_norm"]:
                                    for type_ in ["pred", "gold", "pred_correct"]:
                                        if token == "all" and type_ == "pred":
                                            continue
                                        #score_dic[token + "-norm_not_norm-" + type_ + "-count"] += \
                                        ok = _score[token + "-norm_not_norm-" + type_ + "-count"]
                        except :
                            print("TEST failed score_3", _score_2)

                        for metric in score_to_compute_ls:
                            # TODO : DEPRECIATED : should be remove til # ---- no more need t set the norm/need_norm : all is done by default
                            for mode_norm_score in mode_norm_score_ls:
                                if metric not in ["norm_not_norm-F1", "norm_not_norm-Precision", "norm_not_norm-Recall", "norm_not_norm-accuracy"]:
                                    try:
                                        _score, _n_tokens = score_ls_(text_decoded_ls, gold_text_seq_ls, ls_original=src_text_ls,
                                                                      score=metric, stat=stat,
                                                                      compute_mean_score_per_sent=compute_mean_score_per_sent,
                                                                      normalized_mode=mode_norm_score,
                                                                      verbose=verbose)
                                        if compute_mean_score_per_sent:
                                            score_dic[metric + "-" + mode_norm_score + "-n_sents"] += _score["n_sents"]
                                            score_dic[metric + "-" + mode_norm_score + "-n_word_per_sent"] += _score["n_word_per_sent"]
                                            score_dic[metric + "-" + mode_norm_score+"-mean_per_sent"] += _score["mean_per_sent"]
                                        score_dic[metric + "-" + mode_norm_score] += _score["sum"]
                                        score_dic[metric + "-" + mode_norm_score + "-" + "total_tokens"] += _n_tokens
                                    except Exception as e:
                                        print("Exception {}".format(e))
                                        score_dic[metric + "-" + mode_norm_score] += 0
                                        score_dic[metric + "-" + mode_norm_score + "-" + "total_tokens"] += 0
                                        if compute_mean_score_per_sent:
                                            score_dic[metric + "-" + mode_norm_score + "-n_sents"] += 0
                                            score_dic[metric + "-" + mode_norm_score + "-n_word_per_sent"] += 0
                                            score_dic[metric + "-" + mode_norm_score+"-mean_per_sent"] += 0

                            if batch.output_norm_not_norm is not None:
                                batch.output_norm_not_norm = batch.output_norm_not_norm[:, :pred_norm.size(1)]  # not that clean : we cut gold norm_not_norm sequence
                                _score = score_norm_not_norm(pred_norm, batch.output_norm_not_norm)
                                # means : aux task is on
                                for token in ["all", "need_norm"]:
                                    for type_ in ["pred", "gold","pred_correct"]:
                                        if token == "all" and type_ == "pred":
                                            continue
                                        score_dic[token+"-norm_not_norm-"+type_+"-count"] \
                                                += _score[token+"-norm_not_norm-"+type_+"-count"]

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
            if gold_output:
                assert total_count["src_word_count"] == total_count["target_word_count"], \
                    "ERROR src_word_count {} vs target_word_count {}".format(total_count["src_word_count"], total_count["target_word_count"])
                assert total_count["src_word_count"] == total_count["pred_word_count"], \
                    "ERROR src_word_count {} vs pred_word_count {}".format(total_count["src_word_count"], total_count["pred_word_count"])
                printing("Assertion passed : there are as many words in the source side,"
                         "the target side and"
                         "the predicted side : {} ".format(total_count["src_word_count"]), verbose_level=0, verbose=verbose)
            return score_dic


def decode_sequence(model, char_dictionary, max_len, src_seq, src_mask, src_len,
                    pad=1, target_seq_gold=None,
                    use_gpu=False,
                    single_sequence=False, verbose=2):

    output_seq = pad*np.ones(src_seq.size(), dtype=np.int64)
    # we start with the _START symbol
    output_seq[:, :, 0] = src_seq[:, :, 0] #CHAR_START_ID
    src_text_ls = []
    target_seq_gold_ls = [] if target_seq_gold is not None else None
    output_mask = np.ones(src_mask.size(), dtype=np.int64)
    output_mask[:, :, 1:] = 0
    output_len = Variable(torch.from_numpy(np.ones((src_seq.size(0), src_seq.size(1), 1), dtype=np.int64)), requires_grad=False)
    output_mask = Variable(torch.from_numpy(output_mask), requires_grad=False)
    output_seq = Variable(torch.from_numpy(output_seq), requires_grad=False)
    printing("Data Start source {} {} ", var=(src_seq, src_seq.size()), verbose=verbose, verbose_level=5)
    output_str = True
    printing("WARNING : output_str = True hardcoded (decode_sequence)", verbose=verbose, verbose_level=2)
    printing("Data output sizes ", var=(output_seq.size(), output_len.size(), output_mask.size()), verbose=verbose, verbose_level=6)
    for step, char_decode in enumerate(range(2,  max_len)):
        if use_gpu:
            src_seq = src_seq.cuda()
            output_seq = output_seq.cuda()
            src_len = src_len.cuda()
            output_len = output_len.cuda()
        decoding_states, norm_not_norm, attention = model.forward(input_seq=src_seq,
                                                                  output_seq=output_seq,
                                                                  input_word_len=src_len,
                                                                  output_word_len=output_len)
        # [batch, seq_len, V]
        pred_norm_not_norm = norm_not_norm.argmax(dim=-1) if norm_not_norm is not None else None
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
        if pred_norm_not_norm is not None:
            pred_norm_not_norm = pred_norm_not_norm[:, :scores.size(1)]  # followign what's done above
        output_seq[:, :, char_decode - 1] = predictions[:, :, -1]
        if verbose >= 5:
            sequence = [" ".join([char_dictionary.get_instance(output_seq[sent, word_ind, char_i]) for char_i in range(max_len)])
                        + "|sent-{}|".format(sent) for sent in range(output_seq.size(0)) for word_ind in range(output_seq.size(1))]
        else:
            sequence = []
        printing("Decoding step {} decoded target {} ", var=(step, sequence), verbose=verbose, verbose_level=5)
        pred_word_count, text_decoded, decoded_ls = output_text_(output_seq,#predictions,
                                                                 char_dictionary, single_sequence=single_sequence,
                                                                 output_str=output_str, last=char_decode==(max_len-1),
                                                                 debug=False)
        printing("PREDICTION : array text {} ", var=[text_decoded],
                 verbose=verbose,
                 verbose_level=5)
    src_word_count, src_text, src_all_ls = output_text_(src_seq, char_dictionary, single_sequence=single_sequence,
                                                        output_str=output_str)
    src_text_ls.extend(src_text)
    if target_seq_gold is not None:
        target_word_count, target_text, _ = output_text_(target_seq_gold, char_dictionary, single_sequence=single_sequence,output_str=output_str)
        target_seq_gold_ls.extend(target_text)
    else:
        target_word_count = None
    if single_sequence:
        if model.decoder.attn_layer is not None:
            attention = attention[0]
        if pred_norm_not_norm is not None:
            pred_norm_not_norm = pred_norm_not_norm[0]

    return (text_decoded, src_text_ls, target_seq_gold_ls), \
           {
           "src_word_count": src_word_count,
           "target_word_count": target_word_count,
           "pred_word_count": pred_word_count
           },\
           (attention, src_all_ls,), \
           (pred_norm_not_norm,output_seq,src_seq,target_seq_gold)


def decode_seq_str(seq_string, model, char_dictionary, pad=1,
                   dir_attention=None, save_attention=False,
                   show_att=False,
                   max_len=20, verbose=2, sent_mode=False):
    assert sent_mode
    sent = seq_string.copy()
    # we add empty words at the end otherwie poblem !! # TODO : understand why ? is it because we need word padded at the end of the sentence ?
    sent.append("")
    with torch.no_grad():
        sent_character = []
        sent_words_mask = []
        sent_words_lens = []
        print("sent",sent)
        for seq_string in sent:
            if len(seq_string) > 0:
                _seq_string = ["_START"]
                printing("WARNING : we added _START symbol and _END_CHAR ! ", verbose=verbose, verbose_level=2)
                _seq_string.extend(list(seq_string))
                seq_string = _seq_string + ["_END_CHAR"] #["_END_CHAR"]#["_PAD_CHAR"]
            if len(seq_string) > max_len:
                # cutting to respect dim requirements
                seq_string = seq_string[:max_len-1]+["_PAD_CHAR"]
            if len(seq_string)>0:
                printing("INPUT SEQ is {} ", var=[seq_string], verbose=verbose, verbose_level=2)
            sequence_characters = [char_dictionary.get_index(letter) for letter in seq_string]+[pad for _ in range(max_len-len(seq_string))]
            sent_character.append(sequence_characters)
            masks = [1 for _ in seq_string]+[0 for _ in range(max_len-len(seq_string))]
            sent_words_mask.append(masks)
            words_lens = min(max_len, len(seq_string))
            sent_words_lens.append(words_lens)
            # we have to create batch_size == 2 because of bug

        batch = Variable(torch.from_numpy(np.array([sent_character, sent_character])),
                                       requires_grad=False)
        batch_masks = Variable(torch.from_numpy(np.array([sent_words_mask, sent_words_mask])), requires_grad=False)
        batch_lens = Variable(torch.from_numpy(np.array([sent_words_lens, sent_words_lens])), requires_grad=False)
        batch_lens = batch_lens.unsqueeze(dim=2)
        batch_size = 2
        (text_decoded, src_text, target), _, (attention, src_seq), (pred_norm,)\
            = decode_sequence(model=model, char_dictionary=char_dictionary,
                              max_len=max_len, batch_size=batch_size, src_seq=batch, src_len=batch_lens,
                              src_mask=batch_masks, single_sequence=True, pad=pad, verbose=verbose)
        if attention is not None:
            print("Attention", attention, src_seq, text_decoded)
            for pred_word, src_word, attention_word in zip(text_decoded, src_seq, attention):

                show_attention(list(pred_word), src_word[:attention_word.size(1)],
                               attention_word.transpose(1, 0), save=save_attention, dir_save=dir_attention,show=show_att,
                               model_full_name=model.model_full_name)
            #show_attention("[lekfezlfkh efj ", ["se", "mjfsemkfj"], torch.tensor([[0, .4], [1, 0.6]]))

        if pred_norm is not None:
            norm_not_norm_seq = [(get_label_norm(norm), word) for norm, word in zip(pred_norm, src_text)]
            printing("NORMALIZING : {} ", var=[norm_not_norm_seq], verbose_level=0, verbose=0)
        printing("DECODED text is : {} original is {}",var=(text_decoded, src_text), verbose_level=0, verbose=0)


def decode_interacively(model , char_dictionary,  max_len, pad=1, sent_mode=False, save_attention=False,
                        show_attention=False,
                        dir_attention=None,verbose=0):
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
                               show_att=True,
                               verbose=verbose, sent_mode=True, dir_attention=dir_attention, save_attention=save_attention)
                sentence = []

        elif seq_string == "END":
            printing("ENDING INTERACTION", verbose=verbose, verbose_level=0)
            break
        else:
            sentence.append(seq_string)
            if not sent_mode:
                decode_seq_str(seq_string, model, char_dictionary, pad, max_len, verbose)