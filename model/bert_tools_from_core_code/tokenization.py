# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open

#from .file_utils import cached_path

from model.bert_tools_from_core_code.tools import *

import pdb

logger = logging.getLogger(__name__)

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
}
VOCAB_NAME = 'vocab.txt'


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                              never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)


    def tokenize_origin(self, text, verbose=1):
        split_tokens = []
        alignement_index = []
        basic_tokenization, alignement_with_original_index = self.basic_tokenizer.tokenize(text)
        for token, index in zip(basic_tokenization, alignement_with_original_index):
            word_piece_token = self.wordpiece_tokenizer.tokenize(token)
            for sub_token in word_piece_token:
                split_tokens.append(sub_token)
            alignement_index.extend([index for _ in range(len(word_piece_token))])
        return split_tokens, alignement_index


    def tokenize(self, text, target=None, aligne=False, verbose=1):

        split_tokens = []
        split_tokens_gold = []
        alignement_index = []
        alignement_index_gold = []
        basic_tokenization, alignement_with_original_index = self.basic_tokenizer.tokenize(text)
        #assert aligne, "only align supported here"
        if aligne:
            assert target is not None
            basic_tokenization_target, alignement_with_original_index_target = self.basic_tokenizer.tokenize(target)
        bpe_reading_ind_gold = 0
        bpe_reading_ind = 0
        word_piece_token, word_piece_token_gold = None, None
        breakpoint=False
        attachement_index_shift_gold = 0

        while True:
            #for ind_token, (token, index) in enumerate(zip(basic_tokenization, alignement_with_original_index)):
            if aligne:
                # n to 1
                mask_input = False
                space_gold = False
                if alignement_with_original_index[bpe_reading_ind] < alignement_with_original_index_target[bpe_reading_ind_gold]:
                    bpe_reading_ind_gold -= 1
                    space_gold = True
                    #pdb.set_trace()
                    # two possibilities : or we split more gold or we add space somewhere
                # 1 to n
                elif alignement_with_original_index[bpe_reading_ind] > alignement_with_original_index_target[bpe_reading_ind_gold]:
                    bpe_reading_ind -= 1
                    mask_input = True
                    #bpe_reading_ind_gold -= 1
                    #pdb.set_trace()
                try:
                    if basic_tokenization_target[bpe_reading_ind_gold]=="@":
                        pdb.set_trace()
                    print("TEXT",text)

                    if mask_input:
                        pdb.set_trace()
                        word_piece_token_gold = self.wordpiece_tokenizer.tokenize(basic_tokenization_target[bpe_reading_ind_gold])
                        word_piece_token = ["[MASK]"]
                        attachement_index_shift_gold -= 1
                        breakpoint = False

                    else:

                        word_piece_token, word_piece_token_gold, former_gold = \
                        self.wordpiece_tokenizer.tokenize_aligned(basic_tokenization[bpe_reading_ind],
                                                                  basic_tokenization_target[bpe_reading_ind_gold],
                                                                  former_src=word_piece_token,
                                                                  former_gold=word_piece_token_gold)


                except Exception as e:
                    raise(e)
            else:
                word_piece_token = self.wordpiece_tokenizer.tokenize(basic_tokenization[bpe_reading_ind])
                #bpe_reading_ind += 1
            for sub_token in word_piece_token:
                split_tokens.append(sub_token)
            if aligne:
                for sub_token_gold in word_piece_token_gold:
                    split_tokens_gold.append(sub_token_gold)
                if breakpoint:
                    pass
                    #pdb.set_trace()
                alignement_index_gold.extend([alignement_with_original_index_target[bpe_reading_ind_gold] for _ in range(len(word_piece_token_gold))])
            else:
                split_tokens_gold = None
                alignement_index_gold = None
            alignement_index.extend([alignement_with_original_index[bpe_reading_ind] for _ in range(len(word_piece_token))])
            pdb.set_trace()
            bpe_reading_ind_gold += 1
            bpe_reading_ind += 1
            if bpe_reading_ind == len(alignement_with_original_index):
                print("bpe_reading_ind {} ouf of / {} ".format(bpe_reading_ind, len(alignement_with_original_index)))
                break
            if aligne:
                if bpe_reading_ind_gold == len(alignement_with_original_index_target):
                    print("bpe_reading_ind {} ouf of / {} ".format(bpe_reading_ind, bpe_reading_ind_gold,
                                                                   len(alignement_with_original_index),
                                                                   len(alignement_with_original_index_target)))
                    break
        if breakpoint:
            print("ADDED MASK")
            print("split_tokens", split_tokens)
            print("split_tokens_gold", split_tokens_gold)
        return split_tokens, alignement_index, split_tokens_gold, alignement_index_gold

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            if token != "[SPACE]":
                ids.append(self.vocab[token])
            else:
                print("WARNING : adding SPACE WITH {} index", len(self.vocab))
                ids.append(len(self.vocab))
        if len(ids) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids, special_extra_token=None, special_token_string=None):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            if special_extra_token is not None and i == special_extra_token:
                assert special_token_string is not None
                appending = special_token_string
            else:
                appending = self.ids_to_tokens[i]
            tokens.append(appending)
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            vocab_file = pretrained_model_name_or_path
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)
        # redirect to the cache, if necessary
        try:
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    vocab_file))
            return None
        if resolved_vocab_file == vocab_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
            # than the number of positional embeddings
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer.
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)
        return tokenizer


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""

        alignement_index_ls = []
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for alignement_index, token in enumerate(orig_tokens):
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            token_splitted_on_punc = self._run_split_on_punc(token)
            alignement_index_ls.extend([alignement_index for _ in range(len(token_splitted_on_punc))])
            split_tokens.extend(token_splitted_on_punc)
        # MAYBE THIS CREATE ALSO SOME INDEX SHIFT IN SOME CASES
        output_tokens = whitespace_tokenize(" ".join(split_tokens))

        return output_tokens, alignement_index_ls

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


def get_biggest_bpe_in(char_list, vocab, token_begining=True):
    ind_start = 0
    while ind_start < len(char_list):
        ind_end = len(char_list)
        while ind_start < ind_end:
            substr = "".join(char_list[ind_start:ind_end])
            if not token_begining > 0:
                substr = "##" + substr
            if substr in vocab:
                cur_substr = substr
                print("WARNING : LEAVING ABREVIATION MATCH (SHOULD ADD MASK)", char_list[ind_end:],char_list)
                return cur_substr, ind_end, char_list[ind_end:]
            ind_end -= 1
        # sub_tokens_gold_is_abbreviation.append(cur_substr)
        ind_start = ind_end
    raise (Exception("WARNING  : not match found for char_list {}".format(char_list)))


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize_aligned(self, text, text_target, former_gold, former_src):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        output_tokens_gold = []
        import pdb

        for token, token_gold in zip(whitespace_tokenize(text), whitespace_tokenize(text_target)):
            chars = list(token)
            chars_gold = list(token_gold)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            if len(chars_gold) > self.max_input_chars_per_word:
                output_tokens_gold.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            sub_tokens_gold = []
            start_gold = 0
            left_out_gold = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                cur_substr_gold = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        if not ((len(cur_substr) == 1 and start == 0) or (len(cur_substr) == 3 and start > 0)) and \
                                (start == 0 and cur_substr == "".join(chars_gold[start:end])) or (start > 0 and cur_substr[2:] == "".join(chars_gold[start:end])):
                            # means we have BPE alignement between src ang gold
                            # exception : if it's only a 1 letter bpe where we want the possibility to find larger bpe
                            cur_substr_gold = cur_substr
                        else:
                            # we look in the substrings of chars_gold[start_gold:end_gold] if there are bpe
                            # if we reached the end of the source sequence :
                            # we want to match all the rest of the gold sequence otherwise we can split
                            end_gold = len(chars_gold) if end == len(chars) else end
                            _end_gold = end_gold
                            # we start at the same character as noisy
                            start_gold = start
                            #if token_gold != "[SPACE]":
                            while start_gold < _end_gold:
                                substr_gold = "".join(chars_gold[start_gold:_end_gold])
                                if start_gold > 0:
                                    substr_gold = "##" + substr_gold
                                if substr_gold in self.vocab:
                                    cur_substr_gold = substr_gold
                                    left_out_gold = chars_gold[_end_gold:end_gold]
                                    print("FOUND gold substring of {} : src:{} of token ({}) --> {} LEAVING {}".format(chars_gold, cur_substr, chars, substr_gold, left_out_gold))
                                    break
                                _end_gold -= 1
                            if cur_substr_gold is None:
                                cur_substr_gold = "[SPACE]"
                        #start_gold = _end_gold
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                #if token_gold != "[SPACE]":
                sub_tokens_gold.append(cur_substr_gold)
                #else:
                #    sub_tokens_gold.append("[SPACE]")
                start = end

            is_n_to_1 = len(left_out_gold) > 0
            if is_n_to_1:
                print("HANDLING as is_n_to_1 src:{} gold:{}".format(sub_tokens, sub_tokens_gold))
            if is_n_to_1:
                end_gold_is_n_to_1 = len(chars_gold)
                start_gold_is_n_to_1 = 0
                _end_gold_is_1_to_n = end_gold_is_n_to_1
                is_abbrebiation = False
                if is_n_to_1:
                    # if noise smaller we test if it can be an abbreviation
                    remember_index_char_gold = {}
                    remember_index_char_gold_former = {}
                    if len(token) < len(token_gold):
                        n_char_in_gold = 0
                        char_former = "£"#token_gold[0]
                        for ind_char, char in enumerate(token):
                            if char in token_gold:
                                indices_char = [i for i, x in enumerate(token_gold) if x == char]
                                if len(indices_char) > 1:
                                    first_meet = remember_index_char_gold.get(char, None) is None
                                    if first_meet:
                                        remember_index_char_gold[char] = 0
                                    occurence = remember_index_char_gold[char]
                                    remember_index_char_gold[char] += 1
                                else:
                                    occurence = 0

                                indices_char_former = [i for i, x in enumerate(token_gold) if x == char_former]

                                if len(indices_char_former) > 1:
                                    first_meet = remember_index_char_gold_former.get(char_former, None) is None
                                    if first_meet:
                                        remember_index_char_gold_former[char_former] = 0
                                    occurence_former_char = remember_index_char_gold_former[char_former]
                                    remember_index_char_gold_former[char_former] += 1
                                else:
                                    occurence_former_char = 0

                                if (ind_char == 0 or indices_char[occurence] > indices_char_former[occurence_former_char] or (len(indices_char) > occurence+1 and indices_char[occurence+1] > indices_char_former[occurence_former_char])):
                                        try:
                                            if indices_char[occurence+1] > indices_char_former[occurence_former_char] and not indices_char[occurence] > indices_char_former[occurence_former_char]:
                                                print("WARNING looking for abbreviarion add to look into occurence+1 to find it",token, token_gold)
                                        except Exception:
                                            pass

                                        n_char_in_gold += 1
                                if char == char_former:
                                    print("WARNING : mishandling double")
                                    # TODO HANDE DOUBLE LETTERS AS REAL ABREVIATION
                                    n_char_in_gold += 0
                                char_former = char
                        if n_char_in_gold == len(token):
                            is_abbrebiation = True
                if is_abbrebiation:

                    print("HANDLING {} as an abbreviation normalized as {} ".format(token, token_gold))
                    sub_tokens_is_abbreviation = []
                    sub_tokens_gold_is_abbreviation = []
                    for ind, letter in enumerate(chars):
                        if ind > 0:
                            letter = "##"+letter
                        assert letter in self.vocab
                        sub_tokens_is_abbreviation.append(letter)

                    remember_index = {}
                    remember_index_next = {}
                    forgotten = []
                    print("sub_tokens_is_abbreviation", sub_tokens_is_abbreviation)
                    to_alert = False
                    for split in range(len(sub_tokens_is_abbreviation)):
                        letter_start = sub_tokens_is_abbreviation[split]
                        #print("LOOKING FOR ABBREVIATION MATCH", letter_start)
                        _letter_start_real = letter_start if not letter_start.startswith("##") else letter_start[2:]
                        # where is this letter in the gold token : we assume that their could be doubles but then the first will be accounted
                        indices = [i for i, x in enumerate(chars_gold) if x == _letter_start_real]
                        # handling multiple occurences of a same leter
                        if len(indices) > 1:
                            first_meet = remember_index.get(_letter_start_real, None) is None
                            if first_meet:
                                remember_index[_letter_start_real] = 0
                            occurence = remember_index[_letter_start_real]
                            remember_index[_letter_start_real] += 1
                        else:
                            occurence = 0
                        if split < len(sub_tokens_is_abbreviation)-1:
                            letter_next = sub_tokens_is_abbreviation[split+1]
                            _letter_next_real = letter_next if not letter_next.startswith("##") else letter_next[2:]
                            indices_next = [i for i, x in enumerate(chars_gold) if x == _letter_next_real]
                            if len(indices_next) == 0:
                                occurence_next_letter = 0
                            else:
                                first_meet_as_next = remember_index.get(_letter_next_real) is None#remember_index_next.get(_letter_next_real) is None
                                if first_meet_as_next:
                                    occurence_next_letter = 0
                                    remember_index_next[_letter_next_real] = 0
                                else:
                                    occurence_next_letter = remember_index[_letter_next_real]
                        else:
                            occurence_next_letter = 0
                            indices_next = [len(chars_gold)]

                        substring_to_look = chars_gold[indices[occurence]:indices_next[occurence_next_letter]]

                        if indices[occurence] > indices_next[occurence_next_letter]:
                            try:
                                substring_to_look = chars_gold[indices[occurence]:indices_next[occurence_next_letter+1]]
                                if _letter_next_real in remember_index:
                                    remember_index[_letter_next_real] += 1
                                else:
                                    remember_index[_letter_next_real] = 1
                            except Exception as e:
                                print("ERROR did not find next letter as after the current letter ")
                                raise(e)
                        print("LOOKING INTO ", substring_to_look, indices[occurence], indices_next[occurence_next_letter])
                        cur_substr_gold, ind_end, forgotten_list = get_biggest_bpe_in(char_list=substring_to_look,
                                                                      token_begining=indices[occurence] == 0,
                                                                      vocab=self.vocab)
                        print("FOUND cur_substr_gold ", cur_substr_gold)
                        if len(forgotten_list)>0:
                            to_alert = True
                            to_alert_forgotten = forgotten_list
                        forgotten.append(forgotten_list)
                        sub_tokens_gold_is_abbreviation.append(cur_substr_gold)
                    if to_alert:
                        print("WARNING : LEAVING OUT ", to_alert_forgotten)
                    print("FINAL sub_tokens_gold_is_abbreviation {} : WARNING forgot {}".format(sub_tokens_gold_is_abbreviation, forgotten))
                    sub_tokens = sub_tokens_is_abbreviation
                    sub_tokens_gold = sub_tokens_gold_is_abbreviation
                else:
                    print("1 to n but not abbreviation")
                    start = 0
                    while start < len(left_out_gold):
                        cur_substr_gold, ind_end, forgotten_list = get_biggest_bpe_in(left_out_gold[start:],
                                                                                      self.vocab, False)
                        sub_tokens_gold.append(cur_substr_gold)
                        sub_tokens.append("[MASK]")
                        start += ind_end
                    print("ADDING MASK in sub_tokens {} to match {} ".format(sub_tokens, sub_tokens_gold))

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                if token == "[MASK]":
                    sub_tokens = ["[MASK]"]
                output_tokens.extend(sub_tokens)
            if token_gold == "[SPACE]":
                sub_tokens_gold = "[SPACE]"
            if None in sub_tokens_gold:
                output_tokens_gold.append(self.unk_token)
            else:
                # handling doublons
                if former_gold is not None and former_gold[-1] == sub_tokens_gold[0]:
                    print("DOUBLONS")
                    #pdb.set_trace()
                output_tokens_gold.extend(sub_tokens_gold)

        #print("FINAL output_tokens {}  output_tokens_gold {} ".format(output_tokens, output_tokens_gold))
        return output_tokens, output_tokens_gold, former_gold

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []

            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
