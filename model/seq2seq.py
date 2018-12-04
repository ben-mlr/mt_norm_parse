import torch.nn as nn
import os
import json
from uuid import uuid4
import pdb
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from io_.info_print import printing

DEV = True
DEV_2 = True
DEV_3 = False


class CharEncoder(nn.Module):

    def __init__(self, char_embedding, input_dim, hidden_size_encoder, verbose=2):
        super(CharEncoder, self).__init__()

        self.char_embedding_ = char_embedding
        self.verbose = verbose
        self.seq_encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_size_encoder,
                                  num_layers=1, #nonlinearity='tanh',
                                  bias=True, batch_first=True,
                                  bidirectional=False)

    def forward(self, input, input_mask, input_word_len=None):
        # [batch, seq_len] , batch of (already) padded sequences
        # of indexes (that corresponds to character 1-hot encoded)

        printing("SOURCE dim {} ".format(input.size()), self.verbose, verbose_level=3)
        printing("SOURCE DATA {} ".format(input), self.verbose, verbose_level=5)
        printing("SOURCE DATA mask {} ".format(input_mask), self.verbose, verbose_level=6)
        if DEV:
            printing("SOURCE Word lenght size {} ".format(input_word_len.size()), self.verbose, verbose_level=5)
            printing("SOURCE : Word  length  {}  ".format(input_word_len), self.verbose, verbose_level=3)
            input_word_len, perm_idx = input_word_len.squeeze().sort(0, descending=True)
            # reordering by sequence len
            # [batch, seq_len]
            input = input[perm_idx, :]
        # [batch, max seq_len, dim char embedding]
        char_vecs = self.char_embedding_(input)

        printing("SOURCE embedding dim {} ".format(char_vecs.size()), self.verbose, verbose_level=3)
        if DEV:
            printing("SOURCE  word lengths after  {} dim".format(input_word_len.size()), self.verbose, verbose_level=4)
            packed_char_vecs = pack_padded_sequence(char_vecs, input_word_len.squeeze().cpu().numpy(), batch_first=True)
            printing("SOURCE Packed data shape {} ".format(packed_char_vecs.data.shape), self.verbose, verbose_level=4)
        # all sequence encoding [batch, max seq_len, n_dir x encoding dim] ,
        # last complete hidden state: [dir*n_layer, batch, dim encoding dim]
        if DEV:
            output, h_n = self.seq_encoder(packed_char_vecs)
            printing("SOURCE ENCODED all {}  , hidden {}  (output (includes all the "
                     "hidden states of last layers), last hidden hidden for each dir+layers)".format(output.data.shape, h_n.size()),
                     self.verbose, verbose_level=3)
            output, _ = pad_packed_sequence(output, batch_first=True)
            # useless a we only use h_n as oupput !
        else:
            output, h_n = self.seq_encoder(char_vecs)
        printing("SOURCE ENCODED UNPACKED {}  , hidden {}  (output (includes all the "
                 "hidden states of last layers), last hidden hidden for each dir+layers)".format(output.data.shape, h_n.size()),
                 self.verbose, verbose_level=3)
        # TODO : check that usinh packed sequence indded privdes the last state of the sequence (not the end of the padded one ! )
        # + check this dimension ? why are we loosing a dimension
        return h_n


class CharDecoder(nn.Module):
    def __init__(self, char_embedding, input_dim, hidden_size_decoder, verbose=0):
        super(CharDecoder, self).__init__()
        self.char_embedding_decoder = char_embedding
        self.seq_decoder = nn.GRU(input_size=input_dim, hidden_size=hidden_size_decoder,
                                  num_layers=1, #nonlinearity='tanh',
                                  bias=True, batch_first=True, bidirectional=False)
        self.verbose = verbose
        #self.pre_output_layer = nn.Linear(hidden_size_decoder,, bias=False)

    def forward_step(self, hidden, prev_embed):
        #char_vecs = self.char_embedding_decoder(output_seq)
        # no attention on the target sequence
        #context_vector = conditionning
        # for now straight concatanation of source encoding and c_{t-1}
        #rnn_input = torch.cat([prev_embed, context_vector])
        #output, h_n = self.seq_decoder(rnn_input , hidden)
        #pre_output = self.pre_output_layer(pre_output)
        # update rnn hidden state
        rnn_input = prev_embed#torch.cat([prev_embed, conditionning], dim=2)
        output, hidden = self.seq_decoder(rnn_input, hidden)
        # TOD we can make pre_output more complex later
        #  TOD : we can add context_vector that can be made more complex with some decoding step  attention
        pre_output = output#torch.cat([prev_embed, output, conditionning], dim=2)

        return output, hidden, output

    def forward(self, output, conditioning, output_mask, output_word_len):
        # TODO DEAL WITH MASKING (padding and prediction oriented ?)
        printing("TARGET size {} ".format(output.size()), verbose=self.verbose, verbose_level=3)
        printing("TARGET data {} ".format(output), verbose=self.verbose, verbose_level=5)
        printing("TARGET mask data {} mask {} ".format(output_mask,output_mask.size()), verbose=self.verbose, verbose_level=6)
        printing("TARGET  : Word  length  {}  ".format(output_word_len), self.verbose, verbose_level=5)

        if DEV and DEV_2:
            output_word_len, perm_idx_output = output_word_len.squeeze().sort(0, descending=True)
            output = output[perm_idx_output, :]

        char_vecs = self.char_embedding_decoder(output)

        printing("TARGET EMBEDDING size {} ".format(char_vecs.size()), verbose=self.verbose, verbose_level=3)
        printing("TARGET EMBEDDING data {} ".format(char_vecs), verbose=self.verbose, verbose_level=5)

        max_len = output_word_len.max().data
        pre_output_vectors = []
        #  UNROLLING BY HAND
        if DEV_3:
            # NB 1 : I think it is needed for
            # having a target side contextual attention layer on the source but that packed sequence
            # is fine when we do simpler model as in DEV+DEV_2
            # NB 2 : but actually in this case I don't see how you really handle masking :
            # as you still at training time padded sequence to the rnn --> Does having ignore
            # _index in the loss is enough ?
            decoder_states = []
            hidden = conditioning

            for i in range(max_len):
                prev_embed = char_vecs[:, i, :].unsqueeze(1) #we need 3 dim ; batch, seq, dim emb
                output, h_n, pre_output = self.forward_step(hidden, prev_embed)
                decoder_states.append(output)
                pre_output_vectors.append(pre_output)
            output = torch.cat(pre_output_vectors, dim=1)
            # NB : output is defined very differently here than in the DEV{_2}
        #  USING PACKED SEQUENCE
        if DEV and DEV_2:
            # THe shapes are fine !! -->
            printing("TARGET  word lengths after  {} dim".format(output_word_len.size()), self.verbose, verbose_level=4)
            packed_char_vecs_output = pack_padded_sequence(char_vecs, output_word_len.squeeze().cpu().numpy(), batch_first=True)
            printing("TARGET packed_char_vecs {}  dim".format(packed_char_vecs_output.data.shape), verbose=self.verbose, verbose_level=3)#.size(), packed_char_vecs)
            # conditioning is the output of the encoder (work as the first initial state of the decoder)
            output, h_n = self.seq_decoder(packed_char_vecs_output, conditioning)
            printing("TARGET ENCODED {} output {} h_n (output (includes all the hidden states of last layers), "
                     "last hidden hidden for each dir+layers)".format(output, h_n), verbose=self.verbose,
                     verbose_level=5)
            printing("TARGET ENCODED  SIZE {} output {} h_n (output (includes all the hidden states of last layers), "
                     "last hidden hidden for each dir+layers)".format(output.data.shape, h_n.size()), verbose=self.verbose, verbose_level=3)
            output, output_sizes = pad_packed_sequence(output, batch_first=True)
        # First implementation without accounted for padding
        elif not DEV_3:
            output, h_n = self.seq_decoder(char_vecs, conditioning)

        printing("TARGET ENCODED UNPACKED  {} output {} h_n (output (includes all the hidden states of last layers), "
                 "last hidden hidden for each dir+layers)".format(output, h_n), verbose=self.verbose,
                 verbose_level=5)

        printing("TARGET ENCODED UNPACKED SIZE {} output {} h_n (output (includes all "
                 "  the hidden states of last layers),"
                 "last hidden hidden for each dir+layers)".format(output.size(), h_n.size()),
                 verbose=self.verbose, verbose_level=3)
        return output, h_n


class LexNormalizer(nn.Module):

    def __init__(self, generator, char_embedding_dim=None, hidden_size_encoder=None,
                 hidden_size_decoder=None, voc_size=None, model_id_pref="", model_name="",
                 verbose=0, load=False, dir_model=None, model_full_name=None):
        super(LexNormalizer, self).__init__()
        if not load:
            printing("Defining new model ", verbose=verbose, verbose_level=0)
            assert dir_model is None and model_full_name is None
            assert hidden_size_decoder == hidden_size_encoder, "Warning : For now {} should equal {} because pf the " \
                                                             "init hidden state (cf. TODO for more flexibility )" \
                                                             "".format(hidden_size_encoder, hidden_size_decoder)
            model_id = str(uuid4())[0:4]
            model_id_pref += "_" if len(model_id_pref) > 0 else ""
            model_id += "_" if len(model_name) > 0 else ""
            model_full_name = model_id_pref+model_id+model_name
            printing("Model name is {} with pref_id {} , "
                     "generated id {} and label {} ".format(model_full_name,
                                                           model_id_pref,
                                                           model_id, model_name), verbose=verbose, verbose_level=0)
        else:
            printing("Loading existing model {} from {} ".format(model_full_name, dir_model), verbose=verbose, verbose_level=0)
            assert char_embedding_dim is None and hidden_size_encoder is None and hidden_size_decoder is None and voc_size is None
            assert model_full_name is not None
            args, checkpoint_dir = self.load(dir_model, model_full_name, verbose=verbose)
            char_embedding_dim, hidden_size_encoder, \
            hidden_size_decoder, voc_size = args["char_embedding_dim"], args["hidden_size_encoder"], \
                                                        args["hidden_size_decoder"], args["voc_size"]

        self.model_full_name = model_full_name

        #TODO : add projection of hidden_encoder for getting more flexibility
        self.arguments = {"char_embedding_dim": char_embedding_dim, "hidden_size_encoder": hidden_size_encoder,
                          "hidden_size_decoder": hidden_size_decoder, "voc_size": voc_size}
        printing("Model arguments are {} ".format(self.arguments), verbose, verbose_level=0)
        # 1 share character embedding layer
        self.char_embedding = nn.Embedding(num_embeddings=voc_size, embedding_dim=char_embedding_dim)
        self.encoder = CharEncoder(self.char_embedding, input_dim=char_embedding_dim, hidden_size_encoder= hidden_size_encoder, verbose=verbose)
        self.decoder = CharDecoder(self.char_embedding, input_dim=char_embedding_dim, hidden_size_decoder=hidden_size_decoder, verbose=verbose)
        self.generator = generator(hidden_size_decoder=hidden_size_decoder, voc_size=voc_size, output_dim = 50, verbose=verbose)
        self.verbose = verbose

        self.bridge = nn.Linear(hidden_size_encoder, hidden_size_decoder)
        if load:
            self.load_state_dict(torch.load(checkpoint_dir))

        #self.output_predictor = nn.Linear(in_features=hidden_size_decoder, out_features=voc_size)

    def forward(self, input_seq, output_seq, input_mask, input_word_len, output_mask, output_word_len):
        # [batch, seq_len ] , batch of sequences of indexes (that corresponds to character 1-hot encoded)
        #char_vecs_input = self.char_embedding(input_seq)
        # [batch, seq_len, input_dim] n batch of sequences of embedded character
        h = self.encoder.forward(input_seq, input_mask, input_word_len)
        # [] [batch, , hiden_size_decoder]
        #char_vecs_output = self.char_embedding(output_seq)
        h = self.bridge(h)
        print("debug", h)
        output, h_n = self.decoder.forward(output_seq, h, output_mask, output_word_len)

        # output_score = nn.ReLU()(self.output_predictor(h_out))
        # [batch, output_voc_size], one score per output character token
        # return output
        printing("DECODER full  output sequence encoded of size {} ".format(output.size()), verbose=self.verbose,
                 verbose_level=3)
        printing("DECODER full  output sequence encoded of {}  ".format(output), verbose=self.verbose, verbose_level=5)
        return output

    @staticmethod
    def save(dir, model, verbose=0):
        "saving model as and arguments as json "
        assert os.path.isdir(dir), " ERROR : dir {} does not exist".format(dir)
        checkpoint_dir = os.path.join(dir, model.model_full_name + "-" + "checkpoint.pt")
        arguments_dir = os.path.join(dir,  model.model_full_name + "-" + "args.json")
        assert not os.path.isfile(checkpoint_dir), "Don't want to overwrite {} ".format(checkpoint_dir)
        assert not os.path.isfile(arguments_dir), "Don't want to overwrite {} ".format(arguments_dir)
        torch.save(model.state_dict(), checkpoint_dir)
        print(model.arguments)
        json.dump(model.arguments, open(arguments_dir, "w"))
        printing("Saving model weights and arguments as {}  and {} ".format(checkpoint_dir, arguments_dir),verbose, verbose_level=0)
        return dir, model.model_full_name
    @staticmethod
    def load(dir, model_full_name, verbose=0):
        args = model_full_name+"-args.json"
        checkpoint = model_full_name+"-checkpoint.pt"
        args_dir = os.path.join(dir, args)
        args_checkpoint = os.path.join(dir, checkpoint)
        assert os.path.isfile(args_dir), "ERROR {} does not exits".format(args_dir)
        assert os.path.isfile(args_checkpoint) , "ERROR {} does not exits".format(args_checkpoint)
        args = json.load(open(args_dir,"r"))
        return args, args_checkpoint


class Generator(nn.Module):
    " Define standard linear + softmax generation step."
    def __init__(self, hidden_size_decoder, output_dim, voc_size, verbose=0):
        super(Generator, self).__init__()
        self.dense = nn.Linear(hidden_size_decoder, output_dim)
        self.proj = nn.Linear(output_dim, voc_size)
        self.verbose = verbose
    # TODO : check if relu is needed or not
    # Is not masking needed here ?

    def forward(self, x):
        # return F.log_softmax(self.proj(x), dim=-1)
        # the log_softmax is done within the loss
        y = nn.ReLU()(self.dense(x))
        proj = self.proj(y)
        if self.verbose >= 3:
            print("PROJECTION {} size".format(proj.size()))
        if self.verbose >= 5:
            print("PROJECTION data {} ".format(proj))
        return proj


