from model.seq2seq import LexNormalizer
from model.generator import Generator
import torch
from training.epoch_train import run_epoch
from io_.data_iterator import data_gen_dummy
from model.loss import LossCompute
from tqdm import tqdm
from io_.info_print import disable_tqdm_level
import pdb
EPOCHS = 1
verbose = 2
SEED = 42
from model.seq2seq import TEMPLATE_INFO_CHECKPOINT


def train_1_epoch(epochs=EPOCHS, seq_len=10, generalize_extra=0, nbatches=50, verbose=2, lr=0.001, V=5, batch=2):

    model = LexNormalizer(generator=Generator, char_embedding_dim=5, hidden_size_encoder=11, voc_size=9,
                          hidden_size_sent_encoder=9, output_dim=10,
                          hidden_size_decoder=11, verbose=verbose)
    # optimizer
    adam = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    for epoch in tqdm(range(epochs),disable_tqdm_level(verbose=verbose, verbose_level=0)):
        model.train()
        run_epoch(data_gen_dummy(V=V, batch=batch, nbatches=nbatches, sent_len=seq_len, verbose=verbose, seed=SEED),
                         model, LossCompute(model.generator, opt=adam, verbose=verbose), verbose=verbose, i_epoch=epoch,
                         n_epochs=EPOCHS, n_batches=nbatches)

    loss = run_epoch(data_gen_dummy(V, batch=batch, nbatches=10, seed=SEED, sent_len=seq_len+generalize_extra,verbose=verbose), model, LossCompute(model.generator))
    print("Final Loss {} ".format(loss))
    dir, model_full_name = model.save("./test_logs", model, verbose=verbose, info_checkpoint=TEMPLATE_INFO_CHECKPOINT)
    return dir, model_full_name, loss


def evaluate(model_full_name, dir, nbatches=50, V=5,batch=2,
             seq_len=10, generalize_extra=0, verbose=2):
    model = LexNormalizer(load=True, dir_model=dir, model_full_name=model_full_name, generator=Generator,
                          voc_size=9,
                          verbose=verbose)
    model.eval()
    loss = run_epoch(data_gen_dummy(V, batch=batch, nbatches=10, seed=SEED,
                                    sent_len=seq_len+generalize_extra, verbose=verbose),
                     model, LossCompute(model.generator))
    print("Final Loss {} ".format(loss))
    return loss


def run_test_save_load():
    pdb.set_trace = lambda: 1
    dir, model_full_name, loss = train_1_epoch()
    loss_2 = evaluate(model_full_name, dir)
    print("loss of saved model {} , loss of loaded model {} ".format(loss, loss_2))

    assert loss == loss_2, "ERROR : saved model and loaded model don't have the same loss "
    print("Test saving + reloading passed (based on matching loss) ")


if __name__ == "__main__":
    run_test_save_load()
