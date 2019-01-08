import torch.nn as nn
import os
import json
from uuid import uuid4
from model.encoder import CharEncoder
from model.decoder import CharDecoder
from env.project_variables import CHECKPOINT_DIR
import torch
from io_.info_print import printing
from toolbox.git_related import get_commit_id
from toolbox.sanity_check import sanity_check_info_checkpoint
from env.project_variables import PROJECT_PATH
from io_.dat import conllu_data
from toolbox.deep_learning_toolbox import count_trainable_parameters
from toolbox.sanity_check import get_timing
import time
import re
import pdb
from collections import OrderedDict

#DEV = True
#DEV_2 = True
#DEV_4 = True
#DEV_5 = True

TEMPLATE_INFO_CHECKPOINT = {"n_epochs": 0, "batch_size": None,
                            "train_data_path": None, "dev_data_path": None,
                            "other": None, "git_id": None}


class LexNormalizer(nn.Module):

    def __init__(self, generator, char_embedding_dim=None, hidden_size_encoder=None,output_dim=None,
                 hidden_size_sent_encoder=None,
                 n_layers_word_encoder=1,
                 hidden_size_decoder=None, voc_size=None, model_id_pref="", model_name="",
                 dropout_sent_encoder=0., dropout_word_encoder=0., dropout_word_decoder=0.,
                 dir_sent_encoder=1,
                 dict_path=None, model_specific_dictionary=False, train_path=None, dev_path=None, add_start_char=None,
                 verbose=0, load=False, dir_model=None, model_full_name=None, use_gpu=False, timing=False):
        """
        character level Sequence to Sequence model for normalization
        :param generator:
        :param char_embedding_dim:
        :param hidden_size_encoder:
        :param output_dim:
        :param hidden_size_sent_encoder:
        :param hidden_size_decoder:
        :param voc_size:
        :param model_id_pref: will model_full_name prefix
        :param model_name: model name (will come after id in the naming)
        :param dict_path:
        :param model_specific_dictionary: if True will compute (if new model) or load (if load) a dictionary
        and add it as attributes of the model
        :param train_path: training data path needed to compute the dictionary if model_specific_dictionary
        :param dev_path: dev data path
        :param add_start_char: add or not start symbol for computing the dictionary
        :param verbose:
        :param load:
        :param dir_model:
        :param model_full_name: for loading model
        :param use_gpu:
        """
        super(LexNormalizer, self).__init__()
        # initialize dictionaries
        self.timing = timing
        self.dict_path, self.word_dictionary, self.char_dictionary, self.pos_dictionary, self.xpos_dictionary, self.type_dictionary = None, None, None, None, None, None
        # new model : we create an id , and a saving directory for the model (checkpoints, reporting, arguments)
        if not load:
            printing("Defining new model ", verbose=verbose, verbose_level=0)
            assert dir_model is None and model_full_name is None
            model_id = str(uuid4())[0:4]
            model_id_pref += "_" if len(model_id_pref) > 0 else ""
            model_id += "_" if len(model_name) > 0 else ""
            model_full_name = model_id_pref+model_id+model_name
            printing("Model name is {} with pref_id {} , "
                     "generated id {} and label {} ", var=(model_full_name, model_id_pref, model_id, model_name),
                     verbose=verbose, verbose_level=5)
            # defined at save time
            checkpoint_dir = ""
            self.args_dir = None
            dir_model = os.path.join(PROJECT_PATH, "checkpoints", "{}-folder".format(model_full_name))
            os.mkdir(dir_model)
            printing("Dir {} created", var=(dir_model), verbose=verbose, verbose_level=0)
            git_commit_id = get_commit_id()
            # create dictionary
        # we create/load model specific dictionary
        if model_specific_dictionary:
            if not load:
                # as new model : we need data_path to create nex dictionary
                assert train_path is not None and dev_path is not None and add_start_char is not None, \
                    "ERROR train_path {} dev_path  {} and add_start_char {} are required to load/create dictionary ".format(train_path, dev_path, add_start_char)
                assert voc_size is None and dict_path is None, \
                    "ERROR voc_size will be defined with the new dictionary , dict_path should be None"
                dict_path = os.path.join(dir_model, "dictionaries")
                os.mkdir(dict_path)
                self.dict_path = dict_path
                print("INFO making dict_path {} ".format(dict_path))
            else:
                assert train_path is None and dev_path is None and add_start_char is None
                # we make sure the dictionary dir exists and is located in dict_path
                assert dict_path is not None, "ERROR dict_path should be specified"
                assert os.path.isdir(dict_path), "ERROR : dict_path {} does not exist".format(dict_path)

            self.word_dictionary, self.char_dictionary, \
            self.pos_dictionary, self.xpos_dictionary, self.type_dictionary = \
                conllu_data.load_dict(dict_path=dict_path,
                          train_path=train_path, dev_path=dev_path, test_path=None,
                          word_embed_dict={}, dry_run=False, vocab_trim=True,
                          add_start_char=add_start_char, verbose=1)
            voc_size = len(self.char_dictionary.instance2index) + 1
            printing("char_dictionary {} ", var=(self.char_dictionary.instance2index), verbose=verbose, verbose_level=1)
            printing("Character vocabulary is {} length", var=(len(self.char_dictionary.instance2index) + 1),
                     verbose=verbose, verbose_level=0)

        if not load:
            self.arguments = {"checkpoint_dir": checkpoint_dir,
                              "info_checkpoint": {"n_epochs": 0, "batch_size": None, "train_data_path": None,
                                                  "dev_data_path": None, "other": None,
                                                  "git_id": git_commit_id},
                              "hyperparameters": {
                                  "n_trainable_parameters": None,
                                  "char_embedding_dim": char_embedding_dim,
                                  "encoder_arch": {"cell_word": "GRU", "cell_sentence": "GRU",
                                                   "n_layers_word_encoder":n_layers_word_encoder,
                                                   "attention": "No", "dir_word": "uni",
                                                   "dir_sent_encoder":dir_sent_encoder,
                                                   "dropout_word_encoder":dropout_word_decoder,
                                                   "dropout_sent_encoder":dropout_sent_encoder,
                                                   "dir_sent": "uni"},
                                  "decoder_arch": {"cell_word": "GRU", "cell_sentence": "GRU",
                                                   "attention": "No", "dir_word": "uni",
                                                   "dropout_word_decoder": dropout_word_decoder,
                                                   "dir_sent": "uni"},
                                  "hidden_size_encoder": hidden_size_encoder,
                                  "hidden_size_sent_encoder": hidden_size_sent_encoder,
                                  "hidden_size_decoder": hidden_size_decoder,
                                  "voc_size": voc_size, "output_dim": output_dim
                                 }}
        # we load argument.json and define load weights
        else:
            assert model_full_name is not None and dir_model is not None, \
                "ERROR  model_full_name is {} and dir_model {}  ".format(model_full_name, dir_model)
            printing("Loading existing model {} from {} ", var=(model_full_name, dir_model), verbose=verbose, verbose_level=5)
            assert char_embedding_dim is None and hidden_size_encoder is None and hidden_size_decoder is None and output_dim is None
            if not model_specific_dictionary:
                assert voc_size is not None, "ERROR : voc_size is required for sanity checking as wr recompute the dictionary "

            args, checkpoint_dir, args_dir = self.load(dir_model, model_full_name, verbose=verbose)
            self.arguments = args
            args = args["hyperparameters"]
            # -1 because when is passed for checking it accounts for the unkwnown which is
            # actually not appear in the dictionary
            assert args["voc_size"] == voc_size, "ERROR : voc_size loaded and voc_size " \
                                                 "redefined in dictionnaries do not " \
                                                 "match {} vs {} ".format(args["voc_size"], voc_size)
            char_embedding_dim, hidden_size_encoder, \
            hidden_size_decoder, voc_size, output_dim, hidden_size_sent_encoder, \
             dropout_sent_encoder, dropout_word_encoder, dropout_word_decoder, n_layers_word_encoder, \
                    dir_sent_encoder = \
                args["char_embedding_dim"], args["hidden_size_encoder"], \
                    args["hidden_size_decoder"], args["voc_size"], \
                        args.get("output_dim"), args.get("hidden_size_sent_encoder"), \
                                args["encoder_arch"].get("dropout_sent_encoder"), \
                args["encoder_arch"].get("dropout_word_encoder"), args["decoder_arch"].get("dropout_word_decoder"), \
                    args["encoder_arch"].get("n_layers_word_encoder"), args["encoder_arch"].get("dir_sent_encoder")

            self.args_dir = args_dir

        self.dir_model = dir_model
        self.model_full_name = model_full_name
        printing("Model arguments are {} ".format(self.arguments), verbose, verbose_level=0)
        printing("Model : NB : defined drop outs are the reloaded one ", verbose, verbose_level=0)
        # 1 shared character embedding layer
        self.char_embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=char_embedding_dim)
        self.encoder = CharEncoder(self.char_embedding, input_dim=char_embedding_dim,
                                   hidden_size_encoder=hidden_size_encoder,
                                   dropout_sent_cell=dropout_sent_encoder, dropout_word_cell=dropout_word_encoder,
                                   hidden_size_sent_encoder=hidden_size_sent_encoder,bidir_sent=dir_sent_encoder-1,
                                   n_layers_word_cell=n_layers_word_encoder,timing=timing,
                                   verbose=verbose)
        self.decoder = CharDecoder(self.char_embedding, input_dim=char_embedding_dim,
                                   hidden_size_decoder=hidden_size_decoder,timing=timing,
                                   dropout_word_cell=dropout_word_decoder,
                                   verbose=verbose)
        self.generator = generator(hidden_size_decoder=hidden_size_decoder, voc_size=voc_size,
                                   output_dim=output_dim, verbose=verbose)
        self.verbose = verbose
        # bridge between encoder hidden representation and decoder
        self.bridge = nn.Linear(hidden_size_encoder*n_layers_word_encoder+hidden_size_sent_encoder*dir_sent_encoder, hidden_size_decoder)
        if load:
            # TODO : see if can be factorized
            if use_gpu:
                self.load_state_dict(torch.load(checkpoint_dir))
            else:
                self.load_state_dict(torch.load(checkpoint_dir, map_location=lambda storage, loc: storage))

    def forward(self, input_seq, output_seq, input_word_len, output_word_len):
        # [batch, seq_len ] , batch of sequences of indexes (that corresponds to character 1-hot encoded)
        # char_vecs_input = self.char_embedding(input_seq)
        # [batch, seq_len, input_dim] n batch of sequences of embedded character
        timing = self.timing
        printing("TYPE  input_seq {} input_word_len ", var=(input_seq.is_cuda, input_word_len.is_cuda),
                 verbose=0, verbose_level=4)
        # input_seq : [batch, max sentence length, max word length] : batch of sentences
        start = time.time() if timing else None
        h, sent_len_max_source = self.encoder.sent_encoder_source(input_seq, input_word_len)
        source_encoder, start = get_timing(start)
        # [] [batch, , hiden_size_decoder]
        pdb.set_trace()
        h = self.bridge(h)
        pdb.set_trace()
        bridge, start = get_timing(start)
        printing("TYPE  encoder {} is cuda ", var=h.is_cuda, verbose=0, verbose_level=4)
        output = self.decoder.sent_encoder_target(output_seq, h, output_word_len,
                                                  sent_len_max_source=sent_len_max_source)
        target_encoder, start = get_timing(start)
        printing("TYPE  decoder {} is cuda ", var=(output.is_cuda), verbose=0, verbose_level=4)
        # output_score = nn.ReLU()(self.output_predictor(h_out))
        # [batch, output_voc_size], one score per output character token
        printing("DECODER full  output sequence encoded of size {} ", var=(output.size()), verbose=self.verbose,
                 verbose_level=3)
        printing("DECODER full  output sequence encoded of {}", var=(output), verbose=self.verbose, verbose_level=5)
        time_report = OrderedDict([("source_encoder", source_encoder), ("target_encoder",target_encoder), ("bridge",bridge)])
        if timing:
            print("time report {}".format(time_report))

        return output

    @staticmethod
    def save(dir, model, info_checkpoint, suffix_name="", verbose=0):
        """
        saving model as and arguments as json
        """
        sanity_check_info_checkpoint(info_checkpoint, template=TEMPLATE_INFO_CHECKPOINT)
        assert os.path.isdir(dir), " ERROR : dir {} does not exist".format(dir)
        checkpoint_dir = os.path.join(dir, model.model_full_name + "-"+ suffix_name + "-" + "checkpoint.pt")
        # we update the checkpoint_dir
        model.arguments["hyperparameters"]["n_trainable_parameters"] = count_trainable_parameters(model)
        model.arguments["info_checkpoint"] = info_checkpoint
        model.arguments["info_checkpoint"]["git_id"] = get_commit_id()
        model.arguments["checkpoint_dir"] = checkpoint_dir
        # the arguments dir does not change !
        arguments_dir = os.path.join(dir,  model.model_full_name + "-" + "args.json")
        model.args_dir = arguments_dir
        printing("Warning : overwriting checkpoint {} ", var=(checkpoint_dir), verbose=verbose, verbose_level=0)

        if os.path.isfile(arguments_dir):
            printing("Overwriting argument file (checkpoint dir updated with {}  ) ", var=(checkpoint_dir),
                     verbose=verbose, verbose_level=0)
        printing("Checkpoint info are now {} ", var=(model.arguments["info_checkpoint"]), verbose=verbose,
                 verbose_level=1)
        torch.save(model.state_dict(), checkpoint_dir)
        printing(model.arguments, verbose=verbose, verbose_level=1)
        json.dump(model.arguments, open(arguments_dir, "w"))
        printing("Saving model weights and arguments as {}  and {} ", var=(checkpoint_dir, arguments_dir),
                 verbose=verbose, verbose_level=0)
        return dir, model.model_full_name

    @staticmethod
    def load(dir, model_full_name, verbose=0):
        args = model_full_name+"-args.json"
        args_dir = os.path.join(dir, args)
        assert os.path.isfile(args_dir), "ERROR {} does not exits".format(args_dir)
        args = json.load(open(args_dir, "r"))
        args_checkpoint = args["checkpoint_dir"]
        if not os.path.isfile(args_checkpoint):
            printing("WARNING : checkpoint_dir as indicated in args.json is not found, "
                     "lt checkpoint_dir {}", var=(CHECKPOINT_DIR), verbose=verbose, verbose_level=0)
            # we assume the naming convention /model_name-folder/model_name--checkpoint.pt
            match = re.match(".*/(.*-folder/.*)$", args_checkpoint)
            assert match.group(1) is not None, "ERROR : no match found in {}".format(args_checkpoint)
            args_checkpoint = os.path.join(CHECKPOINT_DIR, match.group(1))
        assert os.path.isfile(args_checkpoint), "ERROR {} does not exits".format(args_checkpoint)
        printing("Checkpoint dir is {} ", var=(args_checkpoint), verbose=verbose, verbose_level=1)
        return args, args_checkpoint, args_dir

