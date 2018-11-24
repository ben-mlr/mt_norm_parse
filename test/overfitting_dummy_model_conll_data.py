import sys

sys.path.insert(0,"/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/ELMoLex_sosweet/")
from dat import conllu_data
from model.seq2seq import LexNormalizer, Generator
from io_.data_iterator import data_gen_conllu
from training.train import run_epoch
from model.loss import LossCompute
from tracking.plot_loss import simple_plot
import torch
dict_path = "../dictionaries/"
train_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/en-ud-train.conllu"
dev_pat = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/owoputi.integrated"
test_path = "/Users/benjaminmuller/Desktop/Work/INRIA/dev/parsing/normpar/data/lexnorm.integrated"


if __name__=="__main__":

    loss_training = []
    verbose = 0
    epochs = 20
    batch_size = 20
    nbatch = 50
    lr = 0.001
    word_dictionary, char_dictionary, pos_dictionary, \
            xpos_dictionary, type_dictionary = \
                conllu_data.create_dict(dict_path=dict_path,
                                        train_path=train_path,
                                        dev_path=dev_pat,
                                        test_path=test_path,
                                        word_embed_dict={},
                                        dry_run=False,
                                        vocab_trim=True)

    print("char_dictionary", char_dictionary.instance2index)
    V = len(char_dictionary.instance2index)+1
    print("Character vocabulary is {} length".format(V))
    model = LexNormalizer(generator=Generator, char_embedding_dim=20, voc_size=V, hidden_size_encoder=50,
                          hidden_size_decoder=50, verbose=verbose)
    adam = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    for epoch in range(epochs):
        model.train()
        batchIter = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                    type_dictionary, batch_size=batch_size, nbatch=nbatch)

        run_epoch(batchIter, model, LossCompute(model.generator, opt=adam), verbose=verbose, n_epoch=epoch)

        model.eval()
        batchIter_eval = data_gen_conllu(test_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                                         type_dictionary, batch_size=batch_size, nbatch=nbatch)
        loss = run_epoch(batchIter_eval, model, LossCompute(model.generator))
        loss_training.append(loss)
        if verbose >= 1:
            print("Final Loss {} ".format(loss))

    simple_plot(final_loss=loss, loss_ls=loss_training, epochs=epochs, save=False,
                lr=lr, prefix="test-LARGER-overfit_conll_dummy")
