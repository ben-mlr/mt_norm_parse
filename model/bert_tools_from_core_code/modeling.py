# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
from collections import OrderedDict
import re
import shutil
import tarfile
import tempfile
import sys
from io import open

from env.importing import torch, nn, CrossEntropyLoss, F, np, pdb
from io_.dat.constants import PAD_ID_LOSS_STANDART
from io_.dat.constants import NUM_LABELS_N_MASKS
from model.bert_tools_from_core_code.tools import get_key_name_num_label

#from .file_utils import cached_path
from model.bert_tools_from_core_code.tools import *

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415BertConfig
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02, normalization_module=False, mask_n_predictor=False,
                 layer_wise_attention=False):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.normalization_module = normalization_module
        self.layer_wise_attention = layer_wise_attention
        self.mask_n_predictor = mask_n_predictor
        self.dense_n_masks_size = None
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range

        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        #num_labels = bert_model_embedding_weights.size(0)+1 if config.normalization_module else bert_model_embedding_weights.size(0)
        num_labels = bert_model_embedding_weights.size(0)

        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), num_labels, bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertMaskNPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertMaskNPredictionHead, self).__init__()
        if config.dense_n_masks_size is None:
            config.dense_n_masks_size = 50
            print("MODEL setting dense_n_masks_size  to default {}".format(config.dense_n_masks_size))
        self.mask_predictor_dense = nn.Linear(config.hidden_size, config.dense_n_masks_size)
        self.mask_predictor_proj = nn.Linear(config.dense_n_masks_size, NUM_LABELS_N_MASKS)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, sequence_output):
        mask_predictor_state = self.activation(self.mask_predictor_dense(sequence_output))
        prediction_scores = self.mask_predictor_proj(mask_predictor_state)
        return prediction_scores


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores,


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, state_dict=None, cache_dir=None,
                        dropout_custom=0.,normalization_mode=True, layer_wise_attention=False,
                        mask_n_predictor=False,
                        from_tf=False, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config

        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.isfile(config_file):
            folder_name = re.match(".*\/(.*).tar.gz", resolved_archive_file).group(1)
            serialization_dir = os.path.join(serialization_dir, folder_name)
            print("Appending {} to {}".format(folder_name, serialization_dir))
            config_file = os.path.join(serialization_dir, CONFIG_NAME)
        assert os.path.isfile(config_file), "ERROR : {} not found".format(config_file)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        if normalization_mode:
            config.normalization_module = normalization_mode
        if layer_wise_attention:
            config.layer_wise_attention = layer_wise_attention
        if dropout_custom > 0:
            config.hidden_dropout_prob = dropout_custom
            config.attention_probs_dropout_prob = dropout_custom
        print("CONFIG updated", config, config_file)
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path,
                                    map_location='cpu' if not torch.cuda.is_available() else None)
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)

        if mask_n_predictor:
            print("MODEL : adding extra post-loading for masks prediction")
            model.mask_n_predictor = BertMaskNPredictionHead(config)

        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


from model.parser_modules import (CHAR_LSTM, MLP, Biaffine, BiLSTM, IndependentDropout, SharedDropout)


class BertTokenHead(nn.Module):
    def __init__(self, config, num_labels, dropout_classifier=None):
        super(BertTokenHead, self).__init__()
        self.num_labels = num_labels
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(dropout_classifier) if dropout_classifier is not None else None
        #self.apply(self.init_bert_weights)

    def forward(self, x, head_mask=None):
        assert head_mask is None, "ERROR : not need of active logits only : handled in the loss for training " \
                                       "attention_mask"
        if self.dropout is not None:
            x = self.dropout(x)
        logits = self.classifier(x)
        # Only keep active parts of the loss
        # NB : the , is mandatory !
        return logits,


class BertGraphHeadKyungTae():

    def __init__(self):
        pass
    def forward(self):
        pass

class BertGraphHead(nn.Module):
    # the MLP layers
    def __init__(self, config, dropout_classifier=None, num_labels=None):
        super(BertGraphHead, self).__init__()
        assert dropout_classifier is None
        n_mlp_arc = 300
        n_mlp_rel = 300

        n_rels = num_labels
        mlp_dropout = 0.1

        pad_index = 1
        unk_index = 0

        self.mlp_arc_h = MLP(n_in=config.hidden_size,
                             n_hidden=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_arc_d = MLP(n_in=config.hidden_size,
                             n_hidden=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=config.hidden_size,
                             n_hidden=n_mlp_rel,
                             dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=config.hidden_size,
                             n_hidden=n_mlp_rel,
                             dropout=mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)

        self.pad_index = pad_index
        self.unk_index = unk_index

    def forward(self, x, head_mask=None):
        # apply MLPs to the BiLSTM output states
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)
        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_heads = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_labels = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        if head_mask is not None:
            # NB ? : is it necessary : as we only keep
            # set the scores that exceed the length of each sentence to -inf
            head_mask = head_mask.byte()
            s_heads.masked_fill_(~head_mask.unsqueeze(1), float('-inf'))


        return s_heads, s_labels


# THIS IS THE ID CARD OF EACH SUPPORTED TASKS in THE PROJECT
# to add a ne tasks : we should just fill all the required fields and that's it
# NB : label fiels should be a list of labels as name in the Batch class

from env.tasks_settings import TASKS_PARAMETER


class BertMultiTask(BertPreTrainedModel):
    """
    BERT model which can call any other modules
    """
    def __init__(self, config, tasks, num_labels_per_task):
        super(BertMultiTask, self).__init__(config)
        self.bert = BertModel(config)
        self.config = config
        assert isinstance(num_labels_per_task, dict)
        assert isinstance(tasks, list) and len(tasks) >= 1, "config.tasks should be a list of len >=1"
        self.head = nn.ModuleDict()
        self.tasks = tasks # tasks we use for a given run
        self.tasks_available = tasks # all tasks available in the model (not only the one we want to use at a given run (self.tasks))
        self.task_parameters = TASKS_PARAMETER
        self.layer_wise_attention = None
        self.labels_supported = [label for task in tasks for label in self.task_parameters[task]["label"]]

        self.sanity_checking_num_labels_per_task(num_labels_per_task, tasks, self.task_parameters)

        self.num_labels_dic = num_labels_per_task
        ##if "mlm" in tasks:
        #   self.num_labels_dic["mlm"] = self.bert.embeddings.word_embeddings.weight.size(0)
        for task in TASKS_PARAMETER:
            if task in tasks:
                #assert task in TASKS_PARAMETER, "ERROR : task {} is not in {}".format(task, TASKS_PARAMETER)
                num_label = get_key_name_num_label(task, self.task_parameters)
                #num_label = task+"-"+self.task_parameters[task]["label"][0] if len(self.task_parameters[task]["label"]) == 1 else task+"-"+self.task_parameters[task]["num_labels_mandatory_to_check"][0] # assuming 1 in num_labels_mandatory_to_check
                if not self.task_parameters[task]["num_labels_mandatory"]:
                    # in this case we need to define and load MLM head of the model
                    self.head[task] = eval(self.task_parameters[task]["head"])(config, self.bert.embeddings.word_embeddings.weight)
                else:
                    self.head[task] = eval(self.task_parameters[task]["head"])(config, num_labels=self.num_labels_dic[num_label])
            else:
                # we define empty heads for downstream use
                self.head[task] = None

    def forward(self, input_ids_dict, token_type_ids=None, attention_mask=None, labels=None, head_masks=None):
        if labels is None:
            labels = OrderedDict()
        if head_masks is None:
            head_masks = OrderedDict()
        sequence_output_dict = OrderedDict()
        logits_dict = OrderedDict()
        loss_dict = OrderedDict()
        # sanity check the labels : they should all be in
        for label, value in labels.items():
            assert label in self.labels_supported, "label {} in {} not supported".format(label, self.labels_supported)

        # task_wise layer attention
        for input_name, input_tensors in input_ids_dict.items():
            sequence_output, _ = self.bert(input_tensors, token_type_ids=None,
                                           attention_mask=attention_mask[input_name],
                                           output_all_encoded_layers=self.layer_wise_attention is not None)
            sequence_output_dict[input_name] = sequence_output

        for task in self.tasks:
            # we don't use mask for parsing heads (cf. test performed below : the -1 already ignore the heads we don't want)
            # NB : head_masks for parsing only applies to heads not types
            head_masks_task = None#head_masks.get(task, None) if task != "parsing" else None
            # NB : head_mask means masks specific the the module heads (nothing related to parsing !! )
            assert self.task_parameters[task]["input"] in sequence_output_dict, \
                "ERROR input {} of task {} was not found in input_ids_dict {}" \
                " and therefore not in sequence_output_dict {} ".format(self.task_parameters[task]["input"],
                                                                        task, input_ids_dict.keys(),
                                                                        sequence_output_dict.keys())

            if not isinstance(self.head[task], BertOnlyMLMHead):
                logits_dict[task] = self.head[task](sequence_output_dict[self.task_parameters[task]["input"]],
                                                           head_mask=head_masks_task)
            else:
                logits_dict[task] = self.head[task](sequence_output_dict[self.task_parameters[task]["input"]])
            # test performed : (logits_dict[task][0][1,2,:20]==float('-inf'))==(labels["parsing_heads"][1,:20]==-1)
            # handle several labels at output (e.g  parsing)
            logits_dict = self.rename_multi_modal_task_logits(labels=self.task_parameters[task]["label"],  task=task, logits_dict=logits_dict, task_parameters=self.task_parameters)

            for logit_label in logits_dict:
            #for label in self.task_parameters[task]["label"]:
                # HANDLE HERE MULTI MODAL TASKS
                label = re.match("(.*)-(.*)", logit_label)
                assert label is not None, "ERROR logit_label {}".format(logit_label)
                label = label.group(2)
                if label in labels:
                    loss_dict[logit_label] = self.get_loss(self.task_parameters[task]["loss"], label, self.num_labels_dic, labels, logits_dict, task, logit_label)
        # thrid output is for potential attention weights
        return logits_dict, loss_dict, None

    def append_extra_heads_model(self, downstream_tasks, num_labels_dic_new):

        self.labels_supported.extend([label for task in downstream_tasks for label in self.task_parameters[task]["label"]])
        self.sanity_check_new_num_labels_per_task(num_labels_new=num_labels_dic_new, num_labels_original=self.num_labels_dic)
        self.num_labels_dic.update(num_labels_dic_new)
        for new_task in downstream_tasks:
            if new_task in self.tasks:
                pass
                #printing("MODEL : task {} in downstream usage was already define
                # in pretrained model", var=[new_task], verbose_level=1, verbose=1)
            else:
                num_label = get_key_name_num_label(new_task, self.task_parameters)
                self.head[new_task] = eval(self.task_parameters[new_task]["head"])(self.config, num_labels=num_labels_dic_new[num_label])

        # we update the tasks attributes
        self.tasks_available = list(set(self.tasks+downstream_tasks))
        self.tasks = downstream_tasks # tasks to be used at prediction time (+ possibly train)

    @staticmethod
    def get_loss(loss_func, label, num_label_dic, labels, logits_dict, task, logit_label):
        if label not in ["heads", "types"]:
            try:
                loss = loss_func(logits_dict[logit_label].view(-1, num_label_dic[logit_label]), labels[label].view(-1))
            except Exception as e:
                print(e)
                print("ERROR task {} num_label {} , labels {} ".format(task, num_label_dic, labels[label].view(-1)))
                raise(e)

        elif label == "heads":
            # trying alternative way for loss
            loss = CrossEntropyLoss(ignore_index=-1, reduction="mean")(logits_dict[logit_label].view(-1, logits_dict[logit_label].size(2)), labels[label].view(-1))
            # other possibilities is to do log softmax then L1 loss (lead to other results)
            if loss < 1e-3:
                pdb.set_trace()
        elif label == "types":
            # gold label after removing 0 gold
            gold = labels["types"][labels["heads"] != PAD_ID_LOSS_STANDART]
            # pred logits (after removing -1) on the gold heads
            pred = logits_dict["parsing-types"][(labels["heads"] != PAD_ID_LOSS_STANDART).nonzero()[:, 0],
                                                (labels["heads"] != PAD_ID_LOSS_STANDART).nonzero()[:, 1], labels["heads"][labels["heads"] != PAD_ID_LOSS_STANDART]]
            # remark : in the way it's coded for paring : the padding is already removed (so ignore index is null)
            loss = loss_func(pred, gold)

        return loss



    @staticmethod
    def rename_multi_modal_task_logits(labels, logits_dict, task, task_parameters):
        #if n_pred == 2:

        n_pred = len(list(logits_dict[task]))
        # try:
        assert n_pred == len(task_parameters[task]["label"]), \
            "ERROR : not as many labels as prediction for task {} : {} vs {} ".format(task, task_parameters[task][
                "label"], logits_dict[task])

        for i_label, label in enumerate(labels):
            # NB : the order of self.task_parameters[task]["label"] must be the same as the head output
            logits_dict[task+"-"+label] = logits_dict[task][i_label]
        del logits_dict[task]
        #elif n_pred == 1:
        #    logits_dict[task+"-"] = logits_dict[logits_label][0]
        #else:
        #    raise (Exception("More than 3 tensors as prediction is not supported (task {})".format(task)))
        return logits_dict



    @staticmethod
    def sanity_checking_num_labels_per_task(num_labels_per_task, tasks, task_parameters):
        for task in tasks:
            # for mwe_prediction no need of num_label we use the embedding matrix
            # do we need to check num_label for this task ? and is only 1 label assosiated to this task
            if task_parameters[task]["num_labels_mandatory"] and len(task_parameters[task]["label"]) == 1:
                if task != "parsing":
                    assert task+"-"+task_parameters[task]["label"][0] in num_labels_per_task,\
                        "ERROR : no num label for task+label {} ".format(task+"-"+task_parameters[task]["label"][0])
                else:
                    assert task in num_labels_per_task, "ERROR : no num label for task {} ".format(task)
            elif task_parameters[task]["num_labels_mandatory"] and len(task_parameters[task]["label"])>1:
                num_labels_mandatory_to_check = task_parameters[task].get("num_labels_mandatory_to_check")
                assert num_labels_mandatory_to_check is not None, "ERROR : task {} is related to at least 2 labels :" \
                                                                  " we need to know which one requires a num_label " \
                                                                  "to define the model head but field {} " \
                                                                  "not found in {}".format(task, "num_labels_mandatory_to_check", task_parameters[task])
                for label in num_labels_mandatory_to_check:
                    assert task+"-"+label in num_labels_per_task, "ERROR : task {} label {} not in num_labels_per_task {} dictionary".format(task, label, num_labels_per_task)
    @staticmethod
    def sanity_check_new_num_labels_per_task(num_labels_original, num_labels_new):
        for label in num_labels_new:
            if label in num_labels_original:
                assert num_labels_original[label] == num_labels_new[label], \
                    "ERROR new num label provided for existing task not the same as original original:{} new:{} ".format(num_labels_original[label], num_labels_new[label])





class BertForTreePrediction(BertPreTrainedModel):
    """
    BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the the Dozat Biaffine parsing module
    """
    def __init__(self, config):
        super(BertForTreePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertGraphHead()
        self.layer_wise_attention = None
        # ?
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=self.layer_wise_attention is not None)
        # from bpe : extract only first bpe of each word
        # feed it to BERT
        # then --> x
        s_arc, s_rel = self.cls(sequence_output)
        #pred_arcs, pred_rels = self.decode(s_arc, s_rel)
        if labels is not None:
            # output loss Cross entropy ?
            pass
        return s_arc, s_rel

    def decode(self, s_arc, s_rel):
        pred_arcs = s_arc.argmax(dim=-1)
        pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)

        return pred_arcs, pred_rels


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels_2=None):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.normalization_module = config.normalization_module
        self.cls = BertOnlyMLMHead(config,
                                   self.bert.embeddings.word_embeddings.weight)

        self.apply(self.init_bert_weights)
        layer_wise_attention = config.layer_wise_attention
        self.classifier_task_2 = None
        self.layer_wise_attention = nn.Linear(config.hidden_size, 1) if layer_wise_attention else None
        self.mask_n_predictor = BertMaskNPredictionHead(config) if config.mask_n_predictor else None
        self.num_labels_n_mask = NUM_LABELS_N_MASKS
        self.num_labels_2 = num_labels_2

        self.loss_weights_default = OrderedDict([("loss_task_1", 1), ("loss_task_2", 1), ("loss_task_n_mask_prediction", 1)])

        print("WARNING : NB in forward(modelling) aggregating_bert_layer_mode is ignore in BertForMaskedLM")

    def forward(self, input_ids,
                token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, labels=None, labels_task_2=None, labels_n_masks=None,
                multi_task_loss_ponderation=None, head_mask=None,
                aggregating_bert_layer_mode=None, output_all_encoded_layers=False,
                mask_token_index=None):

        if masked_lm_labels is None:
            masked_lm_labels = labels

        # masked_lm_labels  : what is it ??
        if multi_task_loss_ponderation is None:
            multi_task_loss_ponderation = self.loss_weights_default
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=self.layer_wise_attention is not None)
        if self.mask_n_predictor is not None:
            sequence_output_masks, _ = self.bert(input_ids, token_type_ids=torch.zeros_like(input_ids), attention_mask=(input_ids != 0) & (input_ids != mask_token_index), output_all_encoded_layers=None)
        softmax_weight = None
        if self.layer_wise_attention is not None:
            stacked_layers = torch.stack(sequence_output, dim=-1).transpose(3, 2).squeeze(-1)
            # for each batch, each input token we have a ponderation over the layers
            energy = self.layer_wise_attention(stacked_layers).squeeze(-1)
            if np.random.random() < 0.1:
                layer_n = np.random.randint(12)
                energy[:, :, layer_n] = -float("Inf")
                print("DROPING LAYER {} IN ATTENTION".format(layer_n))
            softmax_weight = F.softmax(energy, dim=-1).unsqueeze(-1)
            stacked_layers = stacked_layers.transpose(3, 2)
            batch_dim = softmax_weight.size(0)
            len_seq = softmax_weight.size(1)
            stacked_layers = stacked_layers.view(batch_dim*len_seq, stacked_layers.size(2), stacked_layers.size(3))
            softmax_weight = softmax_weight.view(batch_dim*len_seq, softmax_weight.size(2), softmax_weight.size(3))
            new_sequence = torch.bmm(stacked_layers, softmax_weight)
            softmax_weight = softmax_weight.view(batch_dim, len_seq, stacked_layers.size(2))
            sequence_output = new_sequence.view(batch_dim, len_seq, stacked_layers.size(1))

        prediction_scores, = self.cls(sequence_output)
        loss_dict = OrderedDict([("loss", None), ("loss_task_1", 0), ("loss_task_2", 0),
                                 ("loss_task_n_mask_prediction", 0)])
        pred_dict = OrderedDict([("logits_task_1", None), ("logits_task_2", None), ("logits_n_mask_prediction", None)])

        if self.mask_n_predictor is not None:
            assert self.num_labels_n_mask > 0, "ERROR  "
            logits_n_mask_prediction = self.mask_n_predictor(sequence_output_masks)
            #logits_n_mask_prediction = self.mask_n_predictor(sequence_output)
            pred_dict["logits_n_mask_prediction"] = logits_n_mask_prediction
        if labels is not None and self.mask_n_predictor is not None:
            assert labels_n_masks is not None, \
                "ERROR : you provided labels for normalization and" \
                " self.mask_n_predictor : so you should provide labels_n_mask_prediction"
            total_ponderation = 90
            weight = torch.Tensor([5 / total_ponderation, 20 / total_ponderation, 20 / total_ponderation,
                                   20 / total_ponderation, 20 / total_ponderation])
            if logits_n_mask_prediction.is_cuda:
                weight = weight.cuda()
            loss_fct_masks_pred = CrossEntropyLoss(ignore_index=-1, weight=weight)
            loss_dict["loss_task_n_mask_prediction"] = loss_fct_masks_pred(logits_n_mask_prediction.view(-1, self.num_labels_n_mask), labels_n_masks.view(-1))

        if self.classifier_task_2 is not None:
            assert self.num_labels_2 is not None, "num_labels_2 required"
            logits_task_2 = self.classifier_task_2(sequence_output)
            pred_dict["logits_task_2"] = logits_task_2
        if labels_task_2 is not None:
            loss_fct_task_2 = CrossEntropyLoss(ignore_index=-1)
            assert self.classifier_task_2 is not None, \
                "labels_task_2 was provided but self.classifier_task_2 has not been defined"
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits_task_2.view(-1, self.num_labels_2)[active_loss]

                active_labels = labels_task_2.view(-1)[active_loss]
                loss_task_2 = loss_fct_task_2(active_logits, active_labels)
            else:
                loss_task_2 = loss_fct_task_2(logits_task_2.view(-1, self.num_labels_2), labels_task_2.view(-1))
            loss_dict["loss_task_2"] = loss_task_2

        if masked_lm_labels is not None:
            if self.mask_n_predictor is not None:
                assert labels_n_masks is not None, "If have a n_masks predictor then we should give labels for it any time we do it for bpe prediction"
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            num_labels = self.config.vocab_size
            if self.normalization_module:
                num_labels += 1
            masked_lm_loss = loss_fct(prediction_scores.view(-1, num_labels), masked_lm_labels.view(-1))
            loss_dict["loss_task_1"] = masked_lm_loss
            loss_dict["loss"] = loss_dict["loss_task_1"]+loss_dict["loss_task_n_mask_prediction"]
            # TODO : add weights for the loss
        if labels is not None or labels_task_2 is not None:

            assert loss_dict.get("loss_task_2") is None or loss_dict.get("loss_task_2") == 0, \
                "ERROR : task_2 not supported anymore in arg.multitask = 0 "
            loss_dict["loss"] = multi_task_loss_ponderation["loss_task_1"] * loss_dict["loss_task_1"] + multi_task_loss_ponderation["loss_task_n_mask_prediction"] * loss_dict["loss_task_n_mask_prediction"]

            return loss_dict, softmax_weight
        else:
            pred_dict["logits_task_1"] = prediction_scores
            return pred_dict, softmax_weight


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls( pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForMultipleChoice(BertPreTrainedModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_choices):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


class BertForTokenClassification(BertPreTrainedModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)

    ```
    """
    def __init__(self, config, num_labels, dropout_classifier=None, num_labels_2=None):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)

        if dropout_classifier is None:
            dropout_classifier = config.hidden_dropout_prob
            log = "DEFAULT"
        else:
            log = "CUSTOM"
        print("{} : DROPOUT CLASSIFIER set to {} ".format(log, dropout_classifier))
        self.dropout = nn.Dropout(dropout_classifier)
        self.classifier_task_1 = nn.Linear(config.hidden_size, num_labels)
        self.classifier_task_2 = None  #nn.Linear(config.hidden_size, num_labels_2) if num_labels_2 is not None else None
        self.classifier_n_mask = None
        self.num_labels_n_mask = 5
        self.num_labels_2 = num_labels_2
        self.loss_weights_default = OrderedDict([("loss_task_1", 1), ("loss_task_2", 1), ("loss_task_n_mask_prediction", 1)])

        self.classifier_task_1 = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                labels_task_2=None, loss_weights=None, labels_n_mask_prediction=None,
                aggregating_bert_layer_mode=None):
        if aggregating_bert_layer_mode is not None:
            AVAILABLE_BERT_AGGREGATION_MODE = ["sum", "last"]
            assert aggregating_bert_layer_mode in AVAILABLE_BERT_AGGREGATION_MODE or (isinstance(aggregating_bert_layer_mode, int) and aggregating_bert_layer_mode<=11), \
                "ERROR aggregating_bert_layer_mode should be in {} or an int <=11 but is {} ".format(AVAILABLE_BERT_AGGREGATION_MODE, aggregating_bert_layer_mode)
        if loss_weights is None:
            loss_weights = self.loss_weights_default
        else:
            assert isinstance(loss_weights, dict) and loss_weights.get("loss_task_1") is not None \
                   and loss_weights.get("loss_task_2") is not None

        output_all_encoded_layers = (aggregating_bert_layer_mode != "last") if aggregating_bert_layer_mode is not None else False
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=output_all_encoded_layers)
        if output_all_encoded_layers:
            if isinstance(aggregating_bert_layer_mode, int):
                sequence_output = sequence_output[aggregating_bert_layer_mode]
            elif aggregating_bert_layer_mode == "sum":
                sequence_output = torch.sum(torch.stack(sequence_output, dim=-1).squeeze(-1), dim=-1)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier_task_1(sequence_output)

        loss_dict = OrderedDict([("loss", None), ("loss_task_1", 0), ("loss_task_2", 0)])
        pred_dict = OrderedDict([("logits_task_1", None), ("logits_task_2", None)])

        if self.classifier_task_2 is not None:
            assert self.num_labels_2 is not None, "num_labels_2 required"
            logits_task_2 = self.classifier_task_2(sequence_output)
            pred_dict["logits_task_2"] = logits_task_2
        if labels_task_2 is not None:
            loss_fct_task_2 = CrossEntropyLoss(ignore_index=-1)
            assert self.classifier_task_2 is not None, \
                "labels_task_2 was provided but self.classifier_task_2 has not been defined"
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits_task_2.view(-1, self.num_labels_2)[active_loss]

                active_labels = labels_task_2.view(-1)[active_loss]
                loss_task_2 = loss_fct_task_2(active_logits, active_labels)
            else:
                loss_task_2 = loss_fct_task_2(logits_task_2.view(-1, self.num_labels_2), labels_task_2.view(-1))
            loss_dict["loss_task_2"] = loss_task_2

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss_dict["loss_task_1"] = loss
        else:
            pred_dict["logits_task_1"] = logits

        if self.classifier_n_mask is not None:
            assert self.num_labels_n_mask > 0, "ERROR  "
            logits_n_mask_prediction = self.classifier_n_mask(sequence_output)
            pred_dict["logits_n_mask_prediction"] = logits_n_mask_prediction
        if labels is not None and self.classifier_n_mask is not None:
            assert labels_n_mask_prediction is not None, "ERROR : you provided labels for normalization and self.classifier_n_mask : so you should provide labels_n_mask_prediction"
            loss_fct_masks_pred = CrossEntropyLoss(ignore_index=-1)
            loss_dict["loss_task_n_mask_prediction"] = loss_fct_masks_pred(logits_n_mask_prediction.view(-1, self.num_labels_n_mask),
                                                                           labels_n_mask_prediction.view(-1))

        # returning
        if labels is not None or labels_task_2 is not None:
            loss_dict["loss"] = loss_weights["loss_task_1"]*loss_dict["loss_task_1"] + \
                                loss_weights["loss_task_2"]*loss_dict["loss_task_2"] + \
                                loss_weights["loss_task_n_mask_prediction"]*loss_dict["loss_task_n_mask_prediction"]
            return loss_dict, None
        else:
            return pred_dict, None


class BertForQuestionAnswering(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits
