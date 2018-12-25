import torch
from torch.autograd import Variable
from io_.info_print import printing
from io_.from_array_to_text import output_text, output_text_
import numpy as np
from evaluate.normalization_errors import score_ls, score_ls_
from io_.dat.constants import CHAR_START_ID
import pdb
from model.seq2seq import DEV_4, DEV_5


def _init_metric_report(score_to_compute_ls):
    if score_to_compute_ls is not None:
        dic = {score: 0 for score in score_to_compute_ls}
        dic.update({score+"total_tokens": 0 for score in score_to_compute_ls})
        return dic
    return None


def greedy_decode_batch(batchIter, model,char_dictionary, batch_size, pad=1,
                        gold_output=False, score_to_compute_ls=None, stat=None,
                        verbose=0):

        score_dic = _init_metric_report(score_to_compute_ls)
        with torch.no_grad():

            for batch in batchIter:
                # read src sequence
                src_seq = batch.input_seq
                src_len = batch.input_seq_len
                src_mask = batch.input_seq_mask
                target_gold = batch.output_seq if gold_output else None
                # do something with it : When do you stop decoding ?
                #if not DEV_4:
                max_len = src_seq.size(-1)
                pdb.set_trace()
                text_decoded_ls, src_text_ls, gold_text_seq_ls = decode_sequence(model=model,
                                                                                 char_dictionary=char_dictionary,
                                                                                 single_sequence=False,
                                                                                 target_seq_gold=target_gold,
                                                                                 max_len=max_len, src_seq=src_seq,
                                                                                 src_mask=src_mask,src_len=src_len,
                                                                                 batch_size=batch_size, pad=pad,
                                                                                 verbose=verbose)

                printing("Source text {} ".format(src_text_ls), verbose=verbose, verbose_level=2)
                printing("Prediction {} ".format(text_decoded_ls), verbose=verbose, verbose_level=2)
                scores_ls_func = "scores_ls" if not DEV_4 else "score_ls_"
                if gold_output:
                    printing("Gold {} ".format(gold_text_seq_ls), verbose=verbose, verbose_level=2)
                    if score_to_compute_ls is not None:
                        for metric in score_to_compute_ls:
                            _score, _n_tokens = eval(scores_ls_func)(text_decoded_ls, gold_text_seq_ls,
                                                                     score=metric, stat=stat,
                                                                     verbose=verbose)
                            score_dic[metric] += _score
                            score_dic[metric+"total_tokens"] += _n_tokens
            return score_dic


def decode_sequence(model, char_dictionary, max_len, src_seq, src_mask, src_len,
                    batch_size, pad=1, target_seq_gold=None,
                    single_sequence=False,verbose=2):

    output_seq = pad*np.ones(src_seq.size(), dtype=np.int64)
    # we start with the _START symbol
    if not DEV_4:
        output_seq[:,  0] = CHAR_START_ID
    else:
        pdb.set_trace()
        output_seq[:, :, 0] = src_seq[:,:,0] #CHAR_START_ID

    src_text_ls = []
    target_seq_gold_ls = [] if target_seq_gold is not None else None

    if not DEV_4:
        output_mask = np.ones(src_seq.size(), dtype=np.int64)
        output_mask[:, 1:] = 0

    else:
        output_mask = np.ones(src_mask.size(), dtype=np.int64)
        output_mask[:, :, 1:] = 0
        if not DEV_5:
            output_len = Variable(torch.from_numpy(np.ones((src_seq.size(0) * src_seq.size(1)), dtype=np.int64)),
                                  requires_grad=False)
        else:
            pdb.set_trace()
            output_len = Variable(torch.from_numpy(np.ones((src_seq.size(0), src_seq.size(1), 1), dtype=np.int64)), requires_grad=False)

    output_mask = Variable(torch.from_numpy(output_mask), requires_grad=False)
    output_seq = Variable(torch.from_numpy(output_seq), requires_grad=False)
    printing("Data Start source {} {} ".format(src_seq, src_seq.size()), verbose=verbose, verbose_level=6)

    printing("Data Start ".format(output_seq, output_len, output_mask), verbose=verbose, verbose_level=6)

    for step, char_decode in enumerate(range(2,  max_len)):
        pdb.set_trace()
        decoding_states = model.forward(input_seq=src_seq,
                                        output_seq=output_seq,
                                        input_mask=src_mask,
                                        input_word_len=src_len, output_mask=output_mask,
                                        output_word_len=output_len)
        # pdb.set_trace()
        # decoding_states = model.forward(input_seq=src_seq, output_seq=None, input_mask=src_mask,
        # input_word_len=src_len, output_mask=None, output_word_len=None)
        # [batch, seq_len, V]
        #print("scores", scores, scores.size())
        scores = model.generator.forward(x=decoding_states)
        # each time step predict the most likely
        # len
        if DEV_4:
            if not DEV_5:
                output_len = Variable(torch.from_numpy(np.ones(src_seq.size(0)*src_seq.size(1), dtype=np.int64)), requires_grad=False)
            else:
                output_len = Variable(torch.from_numpy(np.ones((src_seq.size(0), scores.size(1), 1), dtype=np.int64)),
                                      requires_grad=False)
        else:
            output_len = Variable(torch.from_numpy(np.ones(src_seq.size(0), dtype=np.int64)),
                                  requires_grad=False)
        if not DEV_5:

            output_len[:] = char_decode
        else:
            pdb.set_trace()
            output_len[:] = char_decode
        # mask
        output_mask = np.ones(src_seq.size(), dtype=np.int64)
        output_mask[:, char_decode:] = 0
        output_mask = Variable(torch.from_numpy(output_mask), requires_grad=False)
        # new seq
        pdb.set_trace()
        predictions = scores.argmax(dim=-1)

        #predictions = predictions[perm[0], :]
        #print("WARNIN reodrered ", perm)
        printing("Prediction size {} ".format(predictions.size()), verbose=verbose, verbose_level=4)
        printing("Prediction {} ".format(predictions), verbose=verbose, verbose_level=5)
        printing("scores: {} scores {} scores sized  {} predicion size {} prediction {} outputseq ".format(scores,
                 scores.size(),
                 predictions.size(),
                 predictions[:, -1],
                 output_seq.size()),
                 verbose=verbose, verbose_level=5)
        if not DEV_4:
            output_seq[:, char_decode-1] = predictions[:, -1]
        else:
            pdb.set_trace()
            #predictions[:, : , -1] = torch.cat((predictions[:, :, -1],
            #                                torch.ones((output_seq.size(0), output_seq.size(1) - predictions.size(1)),
            #                                           dtype=torch.long)), dim=1)
            pdb.set_trace()
            #predictions.size(1)
            output_seq = output_seq[:, :scores.size(1), :]
            output_seq[:, :, char_decode - 1] = predictions[:, :, -1]

        if DEV_4:
            sequence = [" ".join([char_dictionary.get_instance(output_seq[sent, word_ind, char_i]) for char_i in range(max_len)])
                        + "/" for sent in range(batch_size) for word_ind in range(output_seq.size(1))]

        else:
            sequence = [" ".join(
                [char_dictionary.get_instance(output_seq[sent, char_i]) for char_i in range(max_len)])
                        + "/" for sent in range(batch_size)]
        printing("Decoding step {} decoded target {} ".format(step, sequence), verbose=verbose, verbose_level=3)
        #pdb.set_trace()
        if DEV_4:
            pdb.set_trace()
            text_decoded_array, text_decoded = output_text_(predictions,#predictions,
                                                            char_dictionary, single_sequence=single_sequence)
        else:
            text_decoded_array, text_decoded = output_text(predictions, char_dictionary,single_sequence=single_sequence)
        printing("PREDICTION : {} array text {} ".format(text_decoded_array, text_decoded),
                                                         verbose=verbose, verbose_level=6)

        #pdb.set_trace()
    text_ = "output_text_" if DEV_4 else "output_text"
    #pdb.set_trace()
    _, src_text = eval(text_)(src_seq, char_dictionary, single_sequence=single_sequence)
    src_text_ls.extend(src_text)
    if target_seq_gold is not None:
        _, target_text = eval(text_)(target_seq_gold, char_dictionary, single_sequence=single_sequence)
        target_seq_gold_ls.extend(target_text)

    return text_decoded, src_text_ls, target_seq_gold_ls


def decode_seq_str(seq_string, dictionary, model, char_dictionary, pad=1, max_len=20, verbose=2):
    with torch.no_grad():
        _seq_string = ["_START"]
        printing("WARNING : we added _START symbol and _END_CHAR ! ", verbose=verbose, verbose_level=0)
        _seq_string.extend(list(seq_string))
        seq_string = _seq_string + ["_END_CHAR"] #["_END_CHAR"]#["_PAD_CHAR"]

        if len(seq_string) > max_len:
            # cutting to respect dim requirements
            seq_string = seq_string[:max_len-1]+["_PAD_CHAR"]
        printing("INPUT SEQ is {} ".format(seq_string), verbose=verbose, verbose_level=2)
        sequence_characters = [dictionary.get_index(letter) for letter in seq_string]+[pad for _ in range(max_len-len(seq_string))]
        masks = [1 for _ in seq_string]+[0 for _ in range(max_len-len(seq_string))]
        # we have to create batch_size == 2 because of bug
        char_seq = Variable(torch.from_numpy(np.array([sequence_characters, sequence_characters])), requires_grad=False)
        char_mask = Variable(torch.from_numpy(np.array([masks, masks])), requires_grad=False)
        char_len = Variable(torch.from_numpy(np.array([[min(max_len, len(seq_string))],[min(max_len, len(seq_string))]])))
        batch_size = 2

        text_decoded, src_text, target = decode_sequence(model=model, char_dictionary=char_dictionary,
                                                         max_len=max_len,batch_size=batch_size,
                                                         src_seq=char_seq, src_len=char_len,
                                                         src_mask=char_mask,single_sequence=True,
                                                         pad=pad, verbose=verbose)
        print("DECODED text is : {} ".format(text_decoded))


def decode_interacively(dictionary, model , char_dictionary,  max_len, verbose,pad=1):

    while True:
        seq_string = input("Please type what you want to normalize ? to stop type : 'stop'    ")
        if seq_string == "":
            continue
        if seq_string == "stop":
            break
        else:
            decode_seq_str(seq_string, dictionary, model, char_dictionary, pad, max_len, verbose)