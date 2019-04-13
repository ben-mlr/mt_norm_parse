import os
from io_.dat import conllu_data
from toolbox.load_w2v import load_emb

from env.project_variables import PROJECT_PATH, TRAINING, DEV, DIR_TWEET_W2V, \
    TEST, DIR_TWEET_W2V, CHECKPOINT_DIR, DEMO, DEMO2, REPO_DATASET, LIU, LEX_TRAIN, LEX_TEST, LEX_LIU_TRAIN, LIU_DEV, LIU_TRAIN, EWT_DEV,\
    CP_PASTE_DEV, CP_PASTE_TRAIN, CP_PASTE_TEST


def make_dictionary():

    train_path = LIU_TRAIN
    dev_path = LIU_DEV
    test_path = None

    word_embed_dic = load_emb(extern_emb_dir=DIR_TWEET_W2V)

    pos_specific_path = DEV
    word_decoding = True

    dict_path = os.path.join("./test_dictionaries")
    print("WARNING : {} should be empty".format(dict_path))
    word_dictionary, word_nom_dictionary, char_dictionary, \
    pos_dictionary, xpos_dictionary, type_dictionary = \
        conllu_data.load_dict(dict_path=dict_path,
                              test_path=test_path,
                              train_path=train_path, dev_path=dev_path,
                              word_embed_dict=word_embed_dic, dry_run=False,
                              expand_vocab=True,
                              pos_specific_data_set=pos_specific_path,
                              word_normalization=word_decoding,
                              add_start_char=1, verbose=1)

if __name__ == "__main__":

    make_dictionary()
    #TODO : add a real test with tiny datasets that you can check