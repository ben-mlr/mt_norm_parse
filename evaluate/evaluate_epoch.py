from model.sequence_prediction import greedy_decode_batch, decode_seq_str, decode_interacively
import pdb
from model.seq2seq import LexNormalizer, Generator
from io_.data_iterator import data_gen_conllu
from io_.dat import conllu_data



dict_path = "../dictionaries/"
train_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/en-ud-train.conllu"
dev_pat = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/owoputi.integrated"
test_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated"

normalization = True
add_start_char = 1

word_dictionary, char_dictionary, pos_dictionary,\
xpos_dictionary, type_dictionary = \
        conllu_data.create_dict(dict_path=dict_path,
                                train_path=test_path,
                                dev_path=test_path,
                                test_path=None,
                                add_start_char=add_start_char,
                                word_embed_dict={},
                                dry_run=False,
                                vocab_trim=True)

verbose = 2

model = LexNormalizer(generator=Generator, load=True, model_full_name="7fd2", dir_model="../test/test_models",
                      verbose=verbose)
batch_size = 2
nbatch = 2
verbose = 2
data_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated.demo2"
batchIter = data_gen_conllu(data_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                            type_dictionary, batch_size=batch_size, nbatch=nbatch, add_start_char=add_start_char,
                            add_end_char=0,
                            normalization=normalization,
                            print_raw=True,  verbose=verbose)

V = model.arguments["voc_size"]
hidden_size_decoder = model.arguments["hidden_size_decoder"]
model.eval()

batch_decoding = True

#loss = run_epoch(batchIter, model, LossCompute(model.generator, verbose=verbose),
#                     i_epoch=0, n_epochs=1,
#                     verbose=verbose,
#                     log_every_x_batch=100)
#print("LOSS", loss)
if batch_decoding:
    score_to_compute_ls = ["edit","exact"]
    score_dic = greedy_decode_batch(char_dictionary=char_dictionary, verbose=2, gold_output=True,
                                        score_to_compute_ls=score_to_compute_ls,
                                        evaluation_metric="mean",
                                        batchIter=batchIter, model=model, batch_size=batch_size)
    # NB : each batch should have the same size !! same number of words : otherwise averaging is wrong
    try:
        for score in score_to_compute_ls:
            print("MODEL Normalization {} score is {} in average out of {} tokens on {} batches evaluation based on {} "
                  .format(score,score_dic[score]/score_dic[score+"total_tokens"], score_dic[score+"total_tokens"], nbatch, data_path ))
    except ZeroDivisionError as e:
        print("ERROR catched {} ".format(e))
