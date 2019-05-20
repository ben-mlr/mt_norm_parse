#sys.path.insert(0, "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/ELMoLex_sosweet/")
from io_.dat import conllu_data
from io_.data_iterator import data_gen_conllu, data_gen_multi_task_sampling_batch, readers_load
import pdb
import numpy as np
import torch
from env.project_variables import LIU, DEV, DEMO, LIU_DEV, TEST


def _test_iterator_get_batch_mode_False(batch_size, bucket, get_batch_mode, extend_n_batch=1,
                                        verbose = 3):
    path = "/Users/bemuller/Documents/Work/INRIA/dev/parsing/normpar/data/en-ud-dev.integrated"
    path = "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/data/LiLiu/2577_tweets-li.conll"
    path = LIU
    print("test on {}".format(path))
    #pdb.set_trace = lambda: 1

    add_start_char = 1
    add_end_char = 1
    dict_path = "../dictionaries/"
    normalization = True
    word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = conllu_data.create_dict(dict_path=dict_path,
                                                               train_path=path,
                                                               dev_path=path,
                                                               test_path=None,
                                                               word_embed_dict={},
                                                               dry_run=False, word_normalization=True,
                                                               add_start_char=add_start_char)

    data = conllu_data.read_data_to_variable(path, word_dictionary, char_dictionary,
                                             pos_dictionary,
                                             xpos_dictionary,type_dictionary,
                                             word_norm_dictionary=word_norm_dictionary,
                                             use_gpu=None,
                                             bucket=bucket,
                                             symbolic_end=True, symbolic_root=True,
                                             dry_run=0, lattice=False, verbose=verbose,
                                             norm_not_norm=True,
                                             normalization=normalization, word_decoder=True,
                                             add_start_char=add_start_char, add_end_char=add_end_char)
    batchIter = data_gen_conllu(data, word_dictionary, char_dictionary,
                                batch_size=batch_size, get_batch_mode=get_batch_mode, extend_n_batch=extend_n_batch,
                                print_raw=True, normalization=normalization,
                                verbose=verbose)
    n_tokens = 0
    n_sents_outputed = 0
    orders = []
    for i, (batch , order)in enumerate(batchIter):
        #print("order", order)
        orders.extend(order)
        n_tokens += batch.ntokens.data
        n_sents_outputed += batch.input_seq.size(0)
        # we check that each batch is composed of non empty sentence
        sent_i = 0
        see_details = True if i == 0 else False
        if see_details:
            print("sent {}  word 0 input seq {} output seq {}  , word tokens {} ntokens batch {} output_norm_not_norm : word pred {} ".format(
                sent_i, batch.input_seq[sent_i, 1, :], batch.output_seq[sent_i, 1, :], batch.output_norm_not_norm[sent_i, 1], batch.ntokens, batch.output_word[sent_i, 0]))
            print(
                "sent {}  word 3 input seq {} output seq {}  , word tokens {} ntokens batch {} output_norm_not_norm : word pred {} ".format(
                    sent_i, batch.input_seq[sent_i, 3, :], batch.output_seq[sent_i, 3, :],
                    batch.output_norm_not_norm[sent_i, 3], batch.ntokens, batch.output_word[sent_i, 3]))
            print("sent {}  word -1 input seq {} output seq {}  , word tokens {} ntokens batch {} output_norm_not_norm : word pred {} ".format(
                    sent_i, batch.input_seq[sent_i, -1, :], batch.output_seq[sent_i, -1, :], batch.output_norm_not_norm[sent_i, -1], batch.ntokens, batch.output_word[sent_i, -1]))
        checking_out = False
        if checking_out:
            for label, _batch in zip(["input", "output"],
                                     [batch.input_seq, batch.output_seq]):
                for sent_i in range(_batch.size(0)):
                    if verbose >= 3:
                        print("sent {}  word 0  ".format(sent_i), _batch[sent_i, 0, :], )
                        print("sent {}  word 1 ".format(sent_i), _batch[sent_i, 1, :])
                        print("sent {}  word last ".format(sent_i), _batch[sent_i, -1, :])
                    check = (_batch[sent_i, 0, :] == torch.tensor([1 for _ in range(_batch.size(2))]))
                    test = (check == 1).all()
                    assert test.data == 0, "ERROR : for {} sentence {} of batch {} is empty ".format(label, sent_i, _batch)
    n_batch = data[-1]//batch_size
    #if data[-1]-batch_size*n_batch != 1:
    try:
        assert n_sents_outputed == data[-1]
        print("All sentence seen {} ".format(n_sents_outputed))
    except:
        try:
            assert n_sents_outputed == data[-1] - 1
            print("All sentence seen {} except 1 sentence to avoid batch_size == 1".format(n_sents_outputed))
        except:
            bucket_size = data[2]
            if not bucket and not get_batch_mode:
                raise(Exception("bucket is False : we should have skipped one max {} outputed and data {} ".format(n_sents_outputed, data[-1]-1)))
            elif not get_batch_mode:
                assert abs(n_sents_outputed - data[-1]) < len(bucket_size), "ERROR {}".format(len(bucket_size))
                print("TEST : {} sentences seen out of {} , due to skipping batch 1 ".format(n_sents_outputed, data[-1],
                                                                                             len(bucket_size)))
    if not get_batch_mode:
        assert len(set(orders)) == len(orders)
        print("All {} sentences were different (so iterator doing the job) [len(set(orders):{} len(orders):{}".
          format(n_sents_outputed, len(set(orders)), len(orders)))
    else:
        print("{} unique sentences were seen out of {} outputted of data {} extended {} time".format(len(set(orders)),
                                                                                                     n_sents_outputed, path, extend_n_batch))
    return orders


def _test_iterator_get_batch_mode_False_no_bucket(batch_size):
    bucket=False
    get_batch_mode=False
    _test_iterator_get_batch_mode_False(batch_size, bucket=bucket, get_batch_mode=get_batch_mode)


def _test_iterator_get_batch_mode_False_bucket(batch_size):
    bucket = True
    get_batch_mode = False
    _test_iterator_get_batch_mode_False(batch_size, bucket=bucket, get_batch_mode=get_batch_mode)


def _info_iterator_get_batch_mode_True_no_bucket(batch_size, verbose):
    print("Not a test ")
    bucket = True
    get_batch_mode = True

    orders_1 = _test_iterator_get_batch_mode_False(batch_size, bucket=bucket, get_batch_mode=get_batch_mode, verbose=verbose, extend_n_batch=2)

    #checking common sentences from two runs in which we reload reader, iterator
    orders_2 = _test_iterator_get_batch_mode_False(batch_size, bucket=bucket, get_batch_mode=get_batch_mode, verbose=verbose, extend_n_batch=2)
    print(len(list(set(orders_1) & set(orders_2))))


def _test_iterator_multi_task(batch_size, get_batch_mode, tasks, print_raw=False):

    data_set = [TEST]
    #tasks = ["normalize"]
    norm_not_norm = False
    word_decoder = False
    extend_n_batch = 1

    word_dictionary, word_dictionary_norm, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = conllu_data.create_dict(dict_path="../dictionaries/",
                                                               train_path=LIU_DEV,
                                                               dev_path=LIU_DEV,
                                                               test_path=None,
                                                               word_embed_dict={},
                                                               dry_run=False, pos_specific_data_set=DEMO,
                                                               tasks=tasks,
                                                               add_start_char=1)

    readers = readers_load(datasets=data_set, tasks=tasks, word_dictionary=word_dictionary,
                           word_dictionary_norm=word_dictionary_norm, char_dictionary=char_dictionary,
                           pos_dictionary=pos_dictionary, xpos_dictionary=xpos_dictionary,
                           type_dictionary=type_dictionary, use_gpu=None,
                           norm_not_norm=norm_not_norm, word_decoder=word_decoder,
                           add_start_char=1, add_end_char=1, symbolic_end=True, symbolic_root=True,
                           verbose=1)

    iterator_multi = data_gen_multi_task_sampling_batch(tasks=tasks, readers=readers, batch_size=batch_size,
                                                        word_dictionary=word_dictionary,
                                                        word_dictionary_norm=word_dictionary_norm,
                                                        char_dictionary=char_dictionary,
                                                        pos_dictionary=pos_dictionary,
                                                        get_batch_mode=get_batch_mode,
                                                        extend_n_batch=extend_n_batch,
                                                        print_raw=print_raw,
                                                        verbose=1)

    counter_sent_input = 0
    while True:
        try:
            batch = iterator_multi.__next__()
            print(batch)
            counter_sent_input += batch.input_seq.size(0)
        except StopIteration:
            break

    return counter_sent_input, readers


def _test_iterator_multi_task_get_batch_False(print_raw=False):

    tasks = ["pos"]
    get_batch_mode = False

    counter, readers = _test_iterator_multi_task(tasks=tasks, get_batch_mode=get_batch_mode, batch_size=1, print_raw=print_raw)
    total = 0
    for task in tasks:
        total += readers[task][-1]
    print("Seen sentences : {} total {} with distinction per task {}".format(total, counter, [(readers[task][-1], task) for task in tasks]))


def _test_iterator_multi_task_get_batch_True():

    tasks = ["normalize", "pos"]
    get_batch_mode = True

    counter, readers = _test_iterator_multi_task(tasks=tasks, get_batch_mode=get_batch_mode, batch_size=5)
    total = 0
    for task in tasks:
        total += readers[task][-1]
    print("Seen sentences : {} total {} with distinction per task {}".format(total, counter, [(readers[task][-1], task) for task in tasks]))
    # TODO ADD REAL TEST


if __name__=="__main__":
     #should not be impacted by the seed
    torch.manual_seed(11)
    np.random.seed(11)
    #for batch_size in [2, 3, 4, 10, 100]:
    test_get_batch = False
    test_iterator = False
    mt = True
    for batch_size in [10]:
        if test_iterator:
            _test_iterator_get_batch_mode_False_no_bucket(batch_size)
            _test_iterator_get_batch_mode_False_bucket(batch_size)
            print("Test passed for batch_fsize both bucketted and not bucktete", batch_size)
        if test_get_batch:
            _info_iterator_get_batch_mode_True_no_bucket(batch_size, verbose=3)
        if mt:
            _test_iterator_multi_task_get_batch_False(print_raw=True)
            #_test_iterator_multi_task_get_batch_True()


