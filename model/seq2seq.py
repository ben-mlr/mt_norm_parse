import torch.nn as nn
import os
import json
from uuid import uuid4
import numpy as np
from model.encoder import CharEncoder
from model.decoder import CharDecoder
from model.pos_predictor import PosPredictor
from model.normalize_not import BinaryPredictor
from env.project_variables import CHECKPOINT_DIR
import torch
from torch.autograd import Variable

from io_.info_print import printing
from toolbox.load_w2v import load_emb
from toolbox.git_related import get_commit_id
from toolbox.sanity_check import sanity_check_info_checkpoint
from env.project_variables import PROJECT_PATH
from io_.dat import conllu_data
from toolbox.deep_learning_toolbox import count_trainable_parameters
from toolbox.sanity_check import get_timing
import time
import re
import pdb
from toolbox.checkpointing import get_args
from collections import OrderedDict
from model.decoder import WordDecoder


TEMPLATE_INFO_CHECKPOINT = {"n_epochs": 0, "batch_size": None,
                            "train_data_path": None, "dev_data_path": None,
                            "other": None, "git_id": None}


class LexNormalizer(nn.Module):

    def __init__(self, generator,
                 auxilliary_task_norm_not_norm=False, dense_dim_auxilliary=None,dense_dim_auxilliary_2=None,
                 auxilliary_task_pos=False, dense_dim_auxilliary_pos=None, dense_dim_auxilliary_pos_2=None,
                 char_embedding_dim=None, hidden_size_encoder=None,output_dim=None,
                 hidden_size_sent_encoder=None,
                 weight_binary_loss=None,
                 n_layers_word_encoder=1,
                 hidden_size_decoder=None, voc_size=None, word_voc_output_size=0, # set to 0 for fact checking
                 model_id_pref="", model_name="",
                 drop_out_sent_encoder_cell=0., drop_out_word_encoder_cell=0., drop_out_word_decoder_cell=0.,
                 dir_word_encoder=1,
                 drop_out_bridge=0, drop_out_sent_encoder_out=0, drop_out_word_encoder_out=0,
                 drop_out_char_embedding_decoder=0,
                 dir_sent_encoder=1, word_recurrent_cell_encoder=None, word_recurrent_cell_decoder=None,
                 word_voc_input_size=0, word_embedding_dim=0, word_embed=False, word_embed_dir=None, word_embedding_projected_dim= None,
                 unrolling_word=False,
                 dict_path=None, model_specific_dictionary=False, train_path=None, dev_path=None, add_start_char=None, pos_specific_path=None,
                 char_src_attention=False, shared_context="all", teacher_force=False,
                 stable_decoding_state=False, init_context_decoder=True,
                 word_decoding=False, dense_dim_word_pred=None, dense_dim_word_pred_2=None, dense_dim_word_pred_3=None,
                 char_decoding=True,
                 n_layers_sent_cell=1,
                 symbolic_end=False, symbolic_root=False,
                 extend_vocab_with_test=False, test_path=None,
                 activation_char_decoder=None, activation_word_decoder=None,
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
        # TODO factorize as args_checking
        assert (word_decoding or char_decoding) and not (word_decoding and char_decoding), "ERROR sttricly  one of word,char decoding should be True"
        assert init_context_decoder or stable_decoding_state or char_src_attention, "ERROR : otherwise no information passes from the encoder to the decoder"
        if word_embed:
            assert word_embedding_dim>0, "ERROR word_embedding_dim should be >0 as word_embed"
        else:
            assert word_embedding_dim == 0 and word_embedding_projected_dim is None, "ERROR  word_embedding_dim needs to be 0 " \
                                                                                     "and word_embedding_projected_dim None if not word_embed "
        if char_decoding:
            assert dense_dim_word_pred is None or dense_dim_word_pred == 0, "ERROR dense_dim_word_pred should be None as not word_decoding"
        if auxilliary_task_pos:
            assert dense_dim_auxilliary_pos is not None and dense_dim_auxilliary_pos > 0
        else:
            assert dense_dim_auxilliary_pos == 0 or dense_dim_auxilliary_pos is None
        if not auxilliary_task_norm_not_norm:
            assert dense_dim_auxilliary_2 is None or dense_dim_auxilliary_2 == 0, \
                "ERROR dense_dim_auxilliary_2 shound be None or 0 when auxilliary_task_norm_not_norm is False"
            assert dense_dim_auxilliary is None or dense_dim_auxilliary == 0,\
                "ERROR dense_dim_auxilliary shound be None or 0 when auxilliary_task_norm_not_norm is False"
        if dense_dim_auxilliary is None or dense_dim_auxilliary == 0:
            assert dense_dim_auxilliary_2 == 0 or dense_dim_auxilliary_2 is None, "dense_dim_auxilliary_2 should be 0 or None as dense_dim_auxilliary is "
        # initialize dictionaries
        self.timing = timing
        self.dict_path, self.word_dictionary, self.word_nom_dictionary,  self.char_dictionary, self.pos_dictionary, self.xpos_dictionary, self.type_dictionary = None, None, None, None, None, None, None
        self.auxilliary_task_norm_not_norm = auxilliary_task_norm_not_norm
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
            printing("Dir {} created", var=([dir_model]), verbose=verbose, verbose_level=0)
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
                if word_embed_dir is not None:
                    word_embed_dic = load_emb(word_embed_dir, verbose)
                else:
                    word_embed_dic = {}
            else:
                assert train_path is None and dev_path is None and add_start_char is None
                # we make sure the dictionary dir exists and is located in dict_path
                assert dict_path is not None, "ERROR dict_path should be specified"
                assert os.path.isdir(dict_path), "ERROR : dict_path {} does not exist".format(dict_path)
                word_embed_dic = None
            # we are loading the dictionary now because we need it to define the model

            self.word_dictionary, self.word_nom_dictionary, word_embed_np, self.char_dictionary, \
            self.pos_dictionary, self.xpos_dictionary, self.type_dictionary =\
                conllu_data.load_dict(dict_path=dict_path,
                                      train_path=train_path, dev_path=dev_path,
                                      word_embed_dict=word_embed_dic, dry_run=False,
                                      vocab_trim=not extend_vocab_with_test, test_path=test_path,
                                      pos_specific_data_set=pos_specific_path,
                                      word_normalization=word_decoding, add_start_char=add_start_char, verbose=1)
            voc_size = len(self.char_dictionary.instance2index) + 1
            if word_decoding:
                assert self.word_nom_dictionary is not None, "ERROR self.word_nom_dictionary should not be None"
            word_voc_input_size = len(self.word_dictionary.instance2index) + 1
            word_voc_output_size = len(self.word_nom_dictionary.instance2index)+1 if self.word_nom_dictionary is not None else None
            printing("char_dictionary {} ", var=([self.char_dictionary.instance2index]), verbose=verbose, verbose_level=1)
            printing("Character vocabulary is {} length", var=(len(self.char_dictionary.instance2index) + 1),
                     verbose=verbose, verbose_level=0)

        # argument saving
        if not load:
            self.arguments = {"checkpoint_dir": checkpoint_dir,
                              "info_checkpoint": {"n_epochs": 0, "batch_size": None, "train_data_path": None,
                                                  "dev_data_path": None, "other": None,
                                                  "git_id": git_commit_id},
                              "hyperparameters": {
                                  "lr": None, "lr_policy": None, "extend_vocab_with_test": extend_vocab_with_test,
                                  "shared_context": shared_context,
                                  "symbolic_end": symbolic_end, "symbolic_root": symbolic_root,
                                  "gradient_clipping": None,
                                  "tasks_schedule_policy": None,
                                  "auxilliary_arch": {
                                                  "weight_binary_loss": weight_binary_loss,
                                                  "auxilliary_task_norm_not_norm": self.auxilliary_task_norm_not_norm,
                                                  "auxilliary_task_norm_not_norm-dense_dim": dense_dim_auxilliary,
                                                  "auxilliary_task_norm_not_norm-dense_dim_2": dense_dim_auxilliary_2,
                                                  "auxilliary_task_pos": auxilliary_task_pos, "dense_dim_auxilliary_pos": dense_dim_auxilliary_pos,
                                                  "dense_dim_auxilliary_pos_2": dense_dim_auxilliary_pos_2,
                                                      },
                                  "n_trainable_parameters": None,
                                  "char_embedding_dim": char_embedding_dim,
                                  "encoder_arch": {"cell_word": word_recurrent_cell_encoder,
                                                   "cell_sentence": "LSTM", "n_layers_sent_cell": n_layers_sent_cell,
                                                   "word_embed": word_embed,  "word_embedding_dim": word_embedding_dim,
                                                   "word_embedding_projected_dim": word_embedding_projected_dim,
                                                   "n_layers_word_encoder": n_layers_word_encoder,
                                                   "word_embed_init": word_embed_dir,
                                                   "dir_sent_encoder": dir_sent_encoder,
                                                   "dir_word_encoder": dir_word_encoder,
                                                   "drop_out_sent_encoder_out": drop_out_sent_encoder_out,
                                                   "drop_out_word_encoder_out": drop_out_word_encoder_out,
                                                   "dropout_word_encoder_cell": drop_out_word_decoder_cell,
                                                   "dropout_sent_encoder_cell": drop_out_sent_encoder_cell,
                                                 },
                                  "decoder_arch": {"cell_word": word_recurrent_cell_decoder, "cell_sentence": "none",
                                                   "dir_word": "uni",
                                                   "char_decoding": char_decoding,
                                                   "word_decoding": word_decoding,
                                                   "dense_dim_word_pred":dense_dim_word_pred, "dense_dim_word_pred_2":dense_dim_word_pred_2,"dense_dim_word_pred_3":dense_dim_word_pred_3,
                                                   "drop_out_bridge": drop_out_bridge, "drop_out_char_embedding_decoder": drop_out_char_embedding_decoder,
                                                   "drop_out_word_decoder_cell": drop_out_word_decoder_cell,
                                                   "char_src_attention": char_src_attention,
                                                   "unrolling_word": unrolling_word,
                                                   "teacher_force": teacher_force,
                                                   "stable_decoding_state": stable_decoding_state,
                                                   "init_context_decoder": init_context_decoder,
                                                   "activation_word_decoder": str(activation_word_decoder),
                                                   "activation_char_decoder": str(activation_char_decoder),
                                                  },
                                  "hidden_size_encoder": hidden_size_encoder,
                                  "hidden_size_sent_encoder": hidden_size_sent_encoder,
                                  "hidden_size_decoder": hidden_size_decoder,
                                  "voc_size": voc_size, "output_dim": output_dim,
                                  "word_voc_output_size": word_voc_output_size,
                                  "word_voc_input_size": word_voc_input_size,
                                 }}
        # we load argument.json and define load weights
        else:
            assert model_full_name is not None and dir_model is not None, \
                "ERROR  model_full_name is {} and dir_model {}  ".format(model_full_name, dir_model)
            printing("Loading existing model {} from {} ", var=(model_full_name, dir_model), verbose=verbose,
                     verbose_level=5)
            assert char_embedding_dim is None and hidden_size_encoder is None and\
                   hidden_size_decoder is None and output_dim is None
            if not model_specific_dictionary:
                assert voc_size is not None, "ERROR : voc_size is required for sanity checking " \
                                             "as we recompute the dictionary "

            args, checkpoint_dir, args_dir = self.load(dir_model, model_full_name, verbose=verbose)
            self.arguments = args
            args = args["hyperparameters"]
            # -1 because when is passed for checking it accounts for the unkwnown which is
            # actually not appear in the dictionary
            assert args.get("word_voc_input_size", 0) == word_voc_input_size, "ERROR : voc_size word_voc_input_size and voc_size " "redefined in dictionnaries do not match {} vs {} ".format(args.get("word_voc_input_size",0), word_voc_input_size)
            assert args["voc_size"] == voc_size, "ERROR : voc_size loaded and voc_size " \
                                                 "redefined in dictionnaries do not " \
                                                 "match {} vs {} ".format(args["voc_size"], voc_size)
            assert args["word_voc_output_size"] == word_voc_output_size, "ERROR mismatch of stored voc and passed voc {} and passed {} ".format(args["word_voc_output_size"], word_voc_output_size)
            word_voc_output_size = args.get("word_voc_output_size", None)

            char_embedding_dim, output_dim, hidden_size_encoder, hidden_size_sent_encoder, drop_out_sent_encoder_cell,\
            drop_out_word_encoder_cell, drop_out_sent_encoder_out, drop_out_word_encoder_out,\
            n_layers_word_encoder, n_layers_sent_cell, dir_sent_encoder, word_recurrent_cell_encoder, dir_word_encoder,\
            hidden_size_decoder,  word_recurrent_cell_decoder, drop_out_word_decoder_cell, drop_out_char_embedding_decoder, \
                    self.auxilliary_task_norm_not_norm, unrolling_word, char_src_attention, dense_dim_auxilliary, shared_context,\
                teacher_force, dense_dim_auxilliary_2, stable_decoding_state, init_context_decoder, \
            word_decoding, char_decoding, auxilliary_task_pos, dense_dim_auxilliary_pos, dense_dim_auxilliary_pos_2, \
                dense_dim_word_pred, dense_dim_word_pred_2,dense_dim_word_pred_3, \
                symbolic_root, symbolic_end, word_embedding_dim, word_embed, word_embedding_projected_dim, \
                activation_char_decoder, activation_word_decoder = get_args(args, False)


            printing("Loading model with argument {}", var=[args], verbose=0, verbose_level=0)
            self.args_dir = args_dir
        # adjusting for directions : the hidden_size_sent_encoder provided and are the dir x hidden_dim dimensions
        hidden_size_sent_encoder = int(hidden_size_sent_encoder/(n_layers_sent_cell))
        hidden_size_encoder = int(hidden_size_encoder/dir_word_encoder)
        printing("WARNING : Model : hidden dim of word level and sentence leve encoders are divided by the number of directions",
                 verbose_level=1, verbose=verbose)
        self.symbolic_end, self.symbolic_root = symbolic_end, symbolic_root
        self.dir_model = dir_model
        self.model_full_name = model_full_name
        printing("Model arguments are {} ".format(self.arguments), verbose, verbose_level=0)
        printing("Model : NB : defined drop outs are the reloaded one ", verbose, verbose_level=1)
        # 1 shared character embedding layer
        self.char_embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=char_embedding_dim)
        self.word_embedding = nn.Embedding(num_embeddings=word_voc_input_size,
                                           embedding_dim=word_embedding_dim) if word_embed else None
        self.word_embedding_project = nn.Linear(word_embedding_dim, word_embedding_projected_dim) if word_embed and word_embedding_projected_dim is not None else None

        if word_embed_np is not None:
            printing("W2V INFO : loaded embedding shape is {} : {} and {} ", var=[word_embed_np.shape, np.mean(word_embed_np),
                                                                                  np.mean(np.std(word_embed_np, axis=1))],
                     verbose=verbose, verbose_level=1)
            self.word_embedding.weight.data = self.word_embedding.weight.data.copy_(torch.from_numpy(word_embed_np))

        self.encoder = CharEncoder(self.char_embedding, input_dim=char_embedding_dim,
                                   hidden_size_encoder=hidden_size_encoder,
                                   word_recurrent_cell=word_recurrent_cell_encoder,
                                   drop_out_sent_encoder_out=drop_out_sent_encoder_out,
                                   drop_out_word_encoder_out=drop_out_word_encoder_out,
                                   dropout_sent_encoder_cell=drop_out_sent_encoder_cell,
                                   dropout_word_encoder_cell=drop_out_word_encoder_cell,
                                   hidden_size_sent_encoder=hidden_size_sent_encoder, bidir_sent=dir_sent_encoder-1,
                                   n_layers_word_cell=n_layers_word_encoder, timing=timing,
                                   n_layers_sent_cell=n_layers_sent_cell,
                                   dir_word_encoder=dir_word_encoder,context_level=shared_context,
                                   add_word_level=word_embed,
                                   word_embedding_dim_inputed=word_embedding_projected_dim if word_embedding_projected_dim is not None else word_embedding_dim,
                                   verbose=verbose)

        p_word = 1 if shared_context in ["word", "all", "none"] else 0
        p_sent = 1 if shared_context in ["sent", "all"] else 0
        # in sent case : the word embedding only ges int to the word encoder so no need of larger bridge
        p_word_emb = 1 if shared_context != "sent" else 0
        self.shared_context = shared_context

        self.bridge = nn.Linear(
            hidden_size_encoder * dir_word_encoder * n_layers_word_encoder*p_word + hidden_size_sent_encoder*dir_sent_encoder*p_sent
            +(word_embedding_projected_dim if word_embedding_projected_dim is not None else word_embedding_dim )*p_word_emb,#*dir_sent_encoder : added diviion by 2 if dir 2
            hidden_size_decoder)
        self.hidden_size_decoder = hidden_size_decoder
        #self.layer_norm = nn.LayerNorm(hidden_size_decoder, elementwise_affine=False) if True else None
        self.dropout_bridge = nn.Dropout(p=drop_out_bridge)
        dropout_char_encoder = 0.3
        self.dropout_char_encoder = nn.Dropout(p=dropout_char_encoder )
        self.normalize_not_normalize \
            = BinaryPredictor(input_dim=hidden_size_decoder,
                              dense_dim=dense_dim_auxilliary,
                              dense_dim_2=dense_dim_auxilliary_2) if self.auxilliary_task_norm_not_norm else None
        #self.char_embedding_2 = nn.Embedding(num_embeddings=voc_size, embedding_dim=char_embedding_dim)
        self.generator = generator(hidden_size_decoder=hidden_size_decoder, voc_size=voc_size,
                                   activation=activation_char_decoder,
                                   output_dim=output_dim, verbose=verbose)
        self.decoder = CharDecoder(self.char_embedding, input_dim=char_embedding_dim,
                                   hidden_size_decoder=hidden_size_decoder,timing=timing,
                                   drop_out_char_embedding_decoder=drop_out_char_embedding_decoder,
                                   drop_out_word_cell=drop_out_word_decoder_cell,
                                   char_src_attention=char_src_attention,
                                   word_recurrent_cell=word_recurrent_cell_decoder, unrolling_word=unrolling_word,
                                   hidden_size_src_word_encoder=hidden_size_encoder*dir_word_encoder,
                                   init_context_decoder=init_context_decoder,
                                   generator=self.generator if not teacher_force else None, shared_context=shared_context,
                                   stable_decoding_state=stable_decoding_state,
                                   verbose=verbose) if char_decoding else None

        self.word_decoder = WordDecoder(voc_size=word_voc_output_size, input_dim=hidden_size_decoder,
                                        dense_dim=dense_dim_word_pred, dense_dim_2=dense_dim_word_pred_2,
                                        activation=activation_word_decoder,
                                        dense_dim_3=dense_dim_word_pred_3) if word_decoding else None
        voc_pos_size = len(self.pos_dictionary.instance2index)+1
        self.pos_predictor = PosPredictor(voc_pos_size=voc_pos_size, input_dim=hidden_size_decoder, dense_dim=dense_dim_auxilliary_pos) if auxilliary_task_pos else None
        self.verbose = verbose
        # bridge between encoder hidden representation and decoder
        if load:
            # TODO : see if can be factorized
            if use_gpu:
                self.load_state_dict(torch.load(checkpoint_dir))
                self = self.cuda()
            else:
                self.load_state_dict(torch.load(checkpoint_dir, map_location=lambda storage, loc: storage))

    def forward(self, input_seq, input_word_len, word_embed_input=None,
                output_word_len=None, output_seq=None, word_level_predict=False,
                proportion_pred_train=None):
        # [batch, seq_len ] , batch of sequences of indexes (that corresponds to character 1-hot encoded)
        # char_vecs_input = self.char_embedding(input_seq)
        # [batch, seq_len, input_dim] n batch of sequences of embedded character
        timing = self.timing
        if self.decoder and not word_level_predict:
            assert output_seq is not None and output_word_len is not None, \
                "ERROR : output_seq is {} and output_word le, {}".format(output_seq, output_word_len)

        printing("TYPE  input_seq {} input_word_len ", var=(input_seq.is_cuda, input_word_len.is_cuda),
                 verbose=0, verbose_level=4)
        # input_seq : [batch, max sentence length, max word length] : batch of sentences
        start = time.time() if timing else None

        if self.word_embedding is not None:
            word_embed_input = self.word_embedding(word_embed_input)
            if self.word_embedding_project is not None:
                word_embed_input = self.word_embedding_project(word_embed_input)
        context, sent_len_max_source, \
        char_seq_hidden_encoder, word_src_sizes = self.encoder.forward(input_seq, input_word_len,
                                                                       word_embed_input=word_embed_input)

        source_encoder, start = get_timing(start)
        # [] [batch, , hiden_size_decoder]
        printing("DECODER hidden state before bridge size {}", var=[context.size() if context is not None else 0],
                 verbose=0, verbose_level=3)
        pdb.set_trace()
        context = torch.tanh(self.bridge(context))
        #h = self.layer_norm(h) if self.layer_norm is not None else h
        context = self.dropout_bridge(context)
        bridge, start = get_timing(start)

        printing("TYPE  encoder {} is cuda ", var=context.is_cuda, verbose=0, verbose_level=4)
        printing("DECODER hidden state after bridge size {}", var=[context.size()], verbose=0, verbose_level=3)

        norm_not_norm_hidden = self.normalize_not_normalize(context) if self.auxilliary_task_norm_not_norm else None

        if self.auxilliary_task_norm_not_norm:
            printing("DECODER hidden state after norm_not_norm_hidden size {}", var=[norm_not_norm_hidden.size()],
                     verbose=0, verbose_level=4)
        if self.decoder is not None and not word_level_predict:
            pdb.set_trace()
            output, attention_weight_all = self.decoder.forward(output_seq, context, output_word_len,
                                                                word_src_sizes=word_src_sizes,
                                                                char_seq_hidden_encoder=char_seq_hidden_encoder,
                                                                proportion_pred_train=proportion_pred_train,
                                                                sent_len_max_source=sent_len_max_source)
        else:
            output = None
            attention_weight_all = None

        word_pred_state = self.word_decoder.forward(context) if self.word_decoder is not None else None

        if self.pos_predictor is not None:
            pos_pred_state = self.pos_predictor.forward(context)
        else:
            pos_pred_state = None

        target_encoder, start = get_timing(start)
        printing("TYPE  decoder {} is cuda ", var=output.is_cuda if output is not None else None,
                 verbose=0, verbose_level=4)
        # output_score = nn.ReLU()(self.output_predictor(h_out))
        # [batch, output_voc_size], one score per output character token
        printing("DECODER full  output sequence encoded of size {} ", var=[output.size()] if output is not None else None, verbose=self.verbose,
                 verbose_level=3)
        printing("DECODER full  output sequence encoded of {}", var=[output] if output is not None else None, verbose=self.verbose, verbose_level=5)
        if timing:
            time_report = OrderedDict(
                [("source_encoder", source_encoder), ("target_encoder", target_encoder), ("bridge", bridge)])
            print("time report {}".format(time_report))

        return output, word_pred_state, pos_pred_state, norm_not_norm_hidden, attention_weight_all

    @staticmethod
    def save(dir, model, info_checkpoint, suffix_name="",verbose=0):
        """
        saving model as and arguments as json
        """
        sanity_check_info_checkpoint(info_checkpoint, template=TEMPLATE_INFO_CHECKPOINT)
        assert os.path.isdir(dir), " ERROR : dir {} does not exist".format(dir)
        checkpoint_dir = os.path.join(dir, model.model_full_name + "-" + suffix_name + "-" + "checkpoint.pt")
        # we update the checkpoint_dir
        model.arguments["hyperparameters"]["n_trainable_parameters"] = count_trainable_parameters(model)
        printing("MODEL : trainable parameters : {} ",  var=model.arguments["hyperparameters"]["n_trainable_parameters"]
                 , verbose_level=0, verbose=verbose)
        model.arguments["hyperparameters"]["batch_size"] = info_checkpoint["batch_size"]
        model.arguments["hyperparameters"]["gradient_clipping"] = info_checkpoint["gradient_clipping"]
        model.arguments["hyperparameters"]["tasks_schedule_policy"] = info_checkpoint["tasks_schedule_policy"]
        model.arguments["hyperparameters"]["lr"] = info_checkpoint["other"]["lr"]
        model.arguments["hyperparameters"]["lr_policy"] = info_checkpoint["other"]["optim_strategy"]
        model.arguments["hyperparameters"]["weight_binary_loss"] = info_checkpoint["other"]["weight_binary_loss"]
        model.arguments["hyperparameters"]["weight_pos_loss"] = info_checkpoint["other"]["weight_pos_loss"]
        model.arguments["hyperparameters"]["ponderation_normalize_loss"] = info_checkpoint["other"]["ponderation_normalize_loss"]
        model.arguments["info_checkpoint"] = info_checkpoint
        model.arguments["info_checkpoint"]["git_id"] = get_commit_id()
        model.arguments["checkpoint_dir"] = checkpoint_dir
        # the arguments dir does not change !
        arguments_dir = os.path.join(dir,  model.model_full_name + "-" + "args.json")
        model.args_dir = arguments_dir
        printing("WARNING : overwriting checkpoint {} ", var=[checkpoint_dir], verbose=verbose, verbose_level=0)

        if os.path.isfile(arguments_dir):
            printing("Overwriting argument file (checkpoint dir updated with {}  ) ", var=[checkpoint_dir],
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
        printing("Checkpoint dir is {} ", var=([args_checkpoint]), verbose=verbose, verbose_level=1)
        return args, args_checkpoint, args_dir

