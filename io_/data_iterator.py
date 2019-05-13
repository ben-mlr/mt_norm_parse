from env.importing import *
from env.models_dir import BERT_MODEL_DIC
from env.project_variables import DEV, LIU_DEV

from io_.dat.constants import *
from io_.batch_generator import MaskBatch
from env.project_variables import EN_LINES_EWT_TRAIN, LIU_DEV, TRAINING, DEMO, TASKS_PARAMETER
from io_.dat import conllu_data
from io_.info_print import printing, print_char_seq, disable_tqdm_level
from io_.printout_iterator_as_raw import outputing_raw_data_from_iterator
from toolbox.sanity_check import get_timing
from io_.bert_iterators_tools.alignement import aligned_output, realigne

from io_.bert_iterators_tools.string_processing import preprocess_batch_string_for_bert, from_bpe_token_to_str, get_indexes
NORM2NOISY = False


def data_gen_conllu(data, word_dictionary, char_dictionary,
                    word_dictionary_norm,
                    batch_size, task_info="",
                    get_batch_mode=True,
                    padding=PAD_ID_CHAR, print_raw=False, normalization=False,
                    pos_dictionary=None,
                    extend_n_batch=1, dropout_input=0,
                    timing=False,
                    verbose=0):

    n_sents = data[-1]
    if extend_n_batch != 1:
        assert get_batch_mode, "ERROR extending nbatch only makes sense in get_batch True (random iteration) "
    nbatch = n_sents//batch_size*extend_n_batch  # approximated lower approximation 1.9//2 == 0

    if nbatch == 0:
        printing("INFO : n_sents < batch_size so nbatch set to 1 ", verbose=verbose, verbose_level=0)
    printing("TRAINING : Task {} Running {} batches of {} dim (nsent : {} extended {} time(s)) (if 0 will be set to 1) "
             .format(task_info,
                     nbatch,
                     batch_size,
                     n_sents, extend_n_batch), verbose=verbose, verbose_level=1)
    printing("ITERATOR INFO : 1 epoch is {} iteration/step/batch (extension is {}) ", var=[nbatch,
                                                                                           extend_n_batch],
             verbose=verbose, verbose_level=1)
    nbatch = 1 if nbatch == 0 else nbatch
    # deterministic run over all the dataset (for evaluation)
    if not normalization:
        printing("WARNING : Normalisation is False : model is a autoencoder (BOTH iteration and get cases)  "
                 "(get_batch_mode:{}) ",
                 var=[get_batch_mode], verbose=verbose, verbose_level=0)
    if not get_batch_mode:
        for batch in tqdm(conllu_data.iterate_batch_variable(data, batch_size=batch_size,
                                                             normalization=normalization),
                          disable=disable_tqdm_level(verbose, verbose_level=2)):

            words, word_norm, chars, chars_norm, word_norm_not_norm, edit, pos, xpos, heads, types, \
                masks, lengths, order_ids, raw_word_inputs, normalized_str, raw_lines = batch

            if not normalization:
                chars_norm = chars.clone()

            outputing_raw_data_from_iterator(words, word_norm, chars, chars_norm, word_norm_not_norm, pos,
                                             word_dictionary=word_dictionary, pos_dictionary=pos_dictionary,
                                             word_norm_dictionary=word_dictionary_norm,
                                             char_dictionary=char_dictionary,
                                             verbose=verbose, print_raw=print_raw, normalization=normalization)
            if not NORM2NOISY:
                yield MaskBatch(chars, chars_norm,  output_norm_not_norm=word_norm_not_norm, pad=padding, timing=timing,
                                edit=edit,
                                output_word=word_norm, pos=pos, input_word=words, dropout_input=dropout_input,
                                raw_input=raw_word_inputs, raw_output=normalized_str,
                                verbose=verbose), order_ids
            else:
                yield MaskBatch(chars_norm, chars,  output_norm_not_norm=word_norm_not_norm, pad=padding, timing=timing,
                                output_word=word_norm, pos=pos, input_word=words, dropout_input=dropout_input,
                                edit=edit,
                                raw_input=normalized_str, raw_output=raw_word_inputs,
                                verbose=verbose), order_ids

    # get_batch randomly (for training purpose)
    elif get_batch_mode:
        for ibatch in tqdm(range(1, nbatch+1), disable=disable_tqdm_level(verbose, verbose_level=2)):
            # word, char, pos, xpos, heads, types, masks, lengths, morph
            printing("Data : getting {} out of {} batches", var=(ibatch, nbatch+1), verbose= verbose, verbose_level=2)

            word, word_norm, char, chars_norm, word_norm_not_norm, edit, pos, _, _, _, \
            _, lenght, order_ids, raw_word_inputs, normalized_str, _ = conllu_data.get_batch_variable(data,
                                                                                                      batch_size=batch_size,
                                                                                                      normalization=normalization,
                                                                                                      unk_replace=0)
            if char.size(0) <= 1:
                print("WARNING : NOT Skip character ")
                #continue
            printing("TYPE {} word, char {} , chars_norm {} length {} ", var=(word.is_cuda, char.is_cuda,
                                                                              #chars_norm.is_cuda, lenght.is_cuda
                                                                              ),
                     verbose=verbose, verbose_level=5)
            assert min(lenght.data) > 0, "ERROR : min(lenght.data) is {} ".format(min(lenght.data))
            # TODO : you want to correct that : you're missing word !!

            if not normalization:
                chars_norm = char.clone()

            __word_ind = 0
            if normalization:
                if word_norm_not_norm is not None:
                    printing("norm not norm {} ", var=(word_norm_not_norm[:, __word_ind]), verbose=verbose,
                             verbose_level=5)
                printing("Normalized sequence {} ", var=(chars_norm[:, __word_ind, :]), verbose=verbose, verbose_level=5)
            printing("Char {} word ind : word : {}  ", var=(__word_ind, char[:, __word_ind, :]), verbose=verbose,
                     verbose_level=5)

            outputing_raw_data_from_iterator(word, word_norm, char, chars_norm, word_norm_not_norm, pos,
                                             word_dictionary=word_dictionary, pos_dictionary=pos_dictionary,
                                             char_dictionary=char_dictionary,
                                             word_norm_dictionary=word_dictionary_norm,
                                             verbose=verbose, print_raw=True, normalization=normalization)

            if NORM2NOISY:
                print("WARNING !! NORM2NOISY ON ")
                yield MaskBatch(chars_norm, char, output_word=word_norm, edit=edit,
                                output_norm_not_norm=word_norm_not_norm, dropout_input=dropout_input,
                                pos=pos, pad=padding, timing=timing, input_word=word, verbose=verbose), order_ids
            else:
                yield MaskBatch(char, chars_norm, output_word=word_norm, edit=edit,
                                output_norm_not_norm=word_norm_not_norm, dropout_input=dropout_input,
                                pos=pos, pad=padding, timing=timing, input_word=word, verbose=verbose,
                                raw_input=raw_word_inputs, raw_output=normalized_str), order_ids


def data_gen_dummy(V, batch, nbatches, sent_len=9, word_len=5, verbose=0, seed=None):
    "Generate random data for a src-tgt copy task."
    if seed is not None:
        np.random.seed(seed)
    for i in tqdm(range(nbatches), disable=disable_tqdm_level(verbose, verbose_level=2)):
        data = torch.from_numpy(np.random.randint(low=2, high=V, size=(batch, sent_len, word_len)))
        data[:, :,0] = 2
        # we force padding in the dummy model
        data[:, :, -1] = 1
        data[:, :, -2] = 1
        printing("DATA dummy {} ", var=(data), verbose=verbose, verbose_level=5)
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield MaskBatch(src, tgt, pad=1)


def data_gen(V, batch, nbatches,seq_len=10):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(2, V, size=(batch, seq_len)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield MaskBatch(src, tgt, pad=1)


import numpy as np


MODE_BATCH_SAMPLING_AVAILABLE = ["proportional", "uniform"]


def sampling_proportion(task_n_sent, total_n_sents):
    return task_n_sent/total_n_sents*100


def readers_load(datasets, tasks, word_dictionary, word_dictionary_norm , char_dictionary,
                 pos_dictionary, xpos_dictionary, type_dictionary,
                 use_gpu,
                 norm_not_norm=False,
                 word_decoder=False, must_get_norm=True,
                 simultanuous_training=False, bucket=True,max_char_len=None,
                 add_start_char=1, add_end_char=1, symbolic_end=True, symbolic_root=True,
                 verbose=1):

    readers = {}
    #assert not simultanuous_training, "ERROR : so far : "
    assert "all" not in tasks, "ERROR not supported yet (pb for simultanuous training..) "
    if not "all" in tasks and not simultanuous_training:
        assert len(tasks) == len(datasets), "ERROR : as simultanuous_training is {} : " \
                                            "we need 1 dataset per task but have only {} for task {} ".format(simultanuous_training, datasets, tasks)
    elif not simultanuous_training:
        assert len(tasks) == 1, "ERROR : if all should have only all nothing else"
        printing("TRAINING : MultiTask Iterator wit task 'all' ", verbose_level=1, verbose=verbose)
    elif simultanuous_training:
        printing("TRAINING : Training simulatnuously tasks provided in {} (should have all required labels in datasets)",
                 verbose_level=1, verbose=verbose)
        raise(Exception("Not supported yet --> should handle the loop "))

    for task, data in zip(tasks, datasets):
        if task == "normalize":
            tasks = ["normalize", "norm_not_norm"]
        else:
            tasks = [task]
        readers[task] = conllu_data.read_data_to_variable(data, word_dictionary, char_dictionary,
                                                          pos_dictionary,
                                                          xpos_dictionary, type_dictionary,
                                                          use_gpu=use_gpu,
                                                          word_decoder=word_decoder,
                                                          symbolic_end=symbolic_end, symbolic_root=symbolic_root,
                                                          dry_run=0, lattice=False,
                                                          normalization=TASKS_PARAMETER[task]["normalization"],
                                                          bucket=bucket,
                                                          add_start_char=add_start_char,
                                                          add_end_char=add_end_char, tasks=tasks,
                                                          max_char_len=max_char_len,
                                                          must_get_norm=must_get_norm,
                                                          word_norm_dictionary=word_dictionary_norm, verbose=verbose)

    return readers


def data_gen_multi_task_sampling_batch(tasks, readers, word_dictionary, char_dictionary, pos_dictionary,
                                       word_dictionary_norm,
                                       extend_n_batch,
                                       batch_size,  get_batch_mode, mode_batch_sampling="proportional",
                                       padding=PAD_ID_CHAR,
                                       dropout_input=0, print_raw=False,
                                       verbose=1):
    "multitask learning iterator"
    assert len(tasks) == len(readers)
    assert mode_batch_sampling in MODE_BATCH_SAMPLING_AVAILABLE
    iterator = {}
    end_task_flag = {}
    n_sents_per_task_dataset_cumul = {}
    cumul_n_sent = 0
    for task in tasks:
        iterator[task] = data_gen_conllu(data=readers[task], word_dictionary=word_dictionary, task_info=task,
                                         char_dictionary=char_dictionary, pos_dictionary=pos_dictionary,
                                         word_dictionary_norm=word_dictionary_norm,
                                         batch_size=batch_size, extend_n_batch=extend_n_batch,
                                         get_batch_mode=get_batch_mode, dropout_input=dropout_input,
                                         padding=padding,
                                         print_raw=print_raw, normalization=TASKS_PARAMETER[task]["normalization"],
                                         verbose=verbose)
        end_task_flag[task] = False
        cumul_n_sent += readers[task][-1]
        n_sents_per_task_dataset_cumul[task] = cumul_n_sent
    n_sents_per_task_dataset_cumul["all"] = n_sents_per_task_dataset_cumul[tasks[-1]]
    printing("TRAINING : MultiTask batch sampling iterator {} cumulated n_sent   ", var=[n_sents_per_task_dataset_cumul], verbose_level=1, verbose=verbose)
    batch_iter = 0
    while True:
        n_sent_start = 0
        random_sample_id = np.random.randint(0, 100)
        for ind, task in enumerate(tasks):
            if sampling_proportion(n_sent_start, n_sents_per_task_dataset_cumul["all"]) < random_sample_id < sampling_proportion(n_sents_per_task_dataset_cumul[task], n_sents_per_task_dataset_cumul["all"]) and not end_task_flag[task]:
                try:
                    batch, order = iterator[task].__next__()
                    pdb.set_trace()
                    sanity_check_batch_label(task, batch, verbose=verbose)
                    batch_iter += 1
                    yield batch
                except StopIteration:
                    end_task_flag[task] = True
                    printing("ITERATOR END {} ", var=[task], verbose_level=1, verbose=verbose)
                    break
            else:
                n_sent_start = n_sents_per_task_dataset_cumul[task]
        if sum(end_task_flag.values()) == len(tasks):
            break


def sanity_check_batch_label(task, batch, verbose=1):
    # NB : we can do this if elif only because we don't do simulatnuous stuff
    if task in ["all", "normalize"]:
        assert batch.output_seq is not None, "ERROR checking normalization output seq"
    elif task in ["all", "pos"]:
        assert batch.pos is not None, "ERROR checking pos "
    elif task in ["all", "norm_not_norm"]:
        assert batch.output_norm_not_norm is not None, "ERROR checking norm_not_norm"
    elif task in ["all", "edit_prediction"]:
        assert batch.edit is not None, "ERROR edit batch was found None "
    else:
        raise(Exception("task provided {} could not be checked".format(task)))
    #printing("BATCH CHECKED ", verbose=verbose, verbose_level=1)





# TODO :
# - integrate to train.py for both train and validation
# - checl if it works the same when iterate is used and not get_batch
# - check if expand works properly : expands vocab for word and word norm also on POS dataset based on word embedding matrix
# - check if there is not repetition
# is there a test to do so

if __name__=="__main__":
    dummy, conll = False, True
    if dummy:
        iter = data_gen_dummy(V=5, batch=2, nbatches=1)

        for ind, batch in enumerate(iter):
            print("BATCH NUMBER {} ".format(ind))
            print("SRC : ", batch.input_seq)
            print("SRC MASK : ", batch.input_seq_mask)
            print("TARGET : ", batch.output_seq)
            #print("TARGET MASK : ", batch.output_mask)
    elif conll:
        dict_path = "../dictionaries/"
        test_path = "/Users/bemuller/Documents/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated.demo2"
        verbose = 2
        batch_size = 1
        add_start_char = 1
        add_end_char = 1
        extend_n_batch = 1
        word_decoder = True
        word_dictionary, word_dictionary_norm , char_dictionary, pos_dictionary,\
        xpos_dictionary, type_dictionary = conllu_data.create_dict(dict_path=dict_path,
                                                                   train_path=LIU_DEV,
                                                                   dev_path=LIU_DEV,
                                                                   test_path=None,
                                                                   word_embed_dict={},
                                                                   word_normalization=word_decoder,
                                                                   tasks=["normalize"],
                                                                   dry_run=False,
                                                                   pos_specific_data_set=EN_LINES_EWT_TRAIN,
                                                                   add_start_char=add_start_char)

        data_set = [EN_LINES_EWT_TRAIN]
        tasks = ["normalize"]
        print(data_set)
        readers = readers_load(datasets=data_set, tasks=tasks, word_dictionary= word_dictionary,
                               word_dictionary_norm=word_dictionary_norm, char_dictionary=char_dictionary,
                               pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                               type_dictionary=type_dictionary, use_gpu=None,
                               norm_not_norm=True, word_decoder=word_decoder, bucket=False,
                               add_start_char=1, add_end_char=1, symbolic_end=True, symbolic_root=True,
                               verbose=1)
        iterator_multi = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers, batch_size=1,
                                                            word_dictionary=word_dictionary,
                                                            char_dictionary=char_dictionary,
                                                            pos_dictionary=pos_dictionary,
                                                            word_dictionary_norm=word_dictionary_norm,
                                                            extend_n_batch=1, print_raw=True,
                                                            get_batch_mode=False,
                                                            verbose=1)

        while True:
            try:
                batch = iterator_multi.__next__()
                pdb.set_trace()
            except StopIteration as e:
                print(Exception(e))
                break
