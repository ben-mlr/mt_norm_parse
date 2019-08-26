from .ioutils import DependencyInstance, Sentence, SentenceWordPieced
from .constants import DIGIT_RE, MAX_CHAR_LENGTH, NUM_CHAR_PAD, ROOT, ROOT_CHAR, ROOT_POS, ROOT_TYPE, PAD, END_CHAR, END_POS, END_TYPE, END, ROOT_HEADS_INDEX, END_HEADS_INDEX, CLS_BERT, SEP_BERT, MASK_BERT
from io_.info_print import printing
from io_.dat.conllu_get_normalization import get_normalized_token
from env.project_variables import AVAILABLE_TASKS
from env.tasks_settings import TASKS_PARAMETER
from env.importing import sys, os, codecs, pdb, re
#from env.importing import *
from env.project_variables import PROJECT_PATH


class CoNLLReader(object):

  def __init__(self, file_path, word_dictionary,
               char_dictionary, pos_dictionary, type_dictionary, xpos_dictionary,
               lemma_dictionary, word_norm_dictionary=None,
               bert_tokenizer=None,
               case=None,
               max_char_len=MAX_CHAR_LENGTH):
    """
    NB : naming conventions : we call words : syntactic words , tokens : raw unsegmentd tokens
    :param file_path:
    :param word_dictionary:
    :param char_dictionary:
    :param pos_dictionary:
    :param type_dictionary:
    :param xpos_dictionary:
    :param lemma_dictionary:
    :param word_norm_dictionary:
    :param bert_tokenizer:
    :param case:
    :param max_char_len:
    """
    self.__source_file = codecs.open(file_path, 'r', 'utf-8', errors='ignore')
    self.__file_path = file_path
    self.__word_dictionary = word_dictionary
    self.__char_dictionary = char_dictionary
    self.__lemma_dictionary = lemma_dictionary
    self.__word_norm_dictionary = word_norm_dictionary

    self.__pos_dictionary = pos_dictionary
    self.__xpos_dictionary = xpos_dictionary

    self.__type_dictionary = type_dictionary
    self.case = case

    self.bert_tokenizer = bert_tokenizer
    if bert_tokenizer is not None:
      printing("INFO Reader : will provide BERT bpe tokens", verbose=1, verbose_level=1)

    if max_char_len is None:
      max_char_len = MAX_CHAR_LENGTH
    printing("MODEL : max_char_len set to {} in CoNLLREADER ", var=max_char_len, verbose_level=1, verbose=1)
    self.max_char_len = max_char_len

  def close(self):
    self.__source_file.close()

  def getNext(self, tasks, normalize_digits=True,
              symbolic_root=False, symbolic_end=False,
              word_decoder=False, must_get_norm=True,
              load_everything=False,
              verbose=0):
    line = self.__source_file.readline()
    n_words = None
    if tasks is None:
      tasks = []
    else:
      assert len(list(set(tasks) & set(AVAILABLE_TASKS))) > 0,\
        "ERROR tasks provided to iterator is not in AVAILABLE TASKS".format(tasks,AVAILABLE_TASKS)

    normalization = False
    for task in tasks:
      if TASKS_PARAMETER[task]["normalization"]:
        normalization = True
        break
    # skip multiple blank lines.could not handled mismatch
    raw_text = []

    while len(line) > 0 and (len(line.strip()) == 0 or line.strip()[0] == '#'):
      if not len(line.strip()) == 0 and line.strip()[0] == '#':
        raw_text.append(line)
        id_stop_mwe = 0
      line = self.__source_file.readline()
    
    if len(line) == 0:
      return None

    lines = []
    while len(line.strip()) > 0:
      line = line.strip()
      lines.append(line.split('\t'))
      line = self.__source_file.readline()

    length = len(lines)
    if length == 0:
      return None

    words = []
    word_ids = []
    char_seqs = []
    char_id_seqs = []
    lemmas = []
    lemma_ids = []
    
    postags = []
    pos_ids = []
    xpostags = []
    xpos_ids = []

    types = []
    type_ids = []
    heads = []

    norm_words = []
    norm_word_ids = []

    char_norm_id_seqs = []
    char_norm_str_seq = []
    # 1 per raw token (not 1 per word)
    is_mwe = [-1]
    n_masks_to_add_in_raw_label = [-1]
    if self.bert_tokenizer is not None:
      # NB : for the raw tokens we consider the pre-tokenization of the CONLLU format so far
      word_piece_words_index = [0]
      word_piece_normalization_index = [0]
      word_piece_raw_tokens_index = [0]
      word_piece_raw_tokens_aligned_index = [0]
      word_piece_lemmas_index = [0]
      # we start with 0 for bert special characters
      all_indexes = ["0"]

      is_first_bpe_of_token = [-1]
      is_first_bpe_of_words = [-1]

      word_piece_raw_tokens = self.bert_tokenizer.convert_tokens_to_ids([CLS_BERT])
      word_piece_raw_tokens_aligned = self.bert_tokenizer.convert_tokens_to_ids([CLS_BERT])
      word_piece_words = self.bert_tokenizer.convert_tokens_to_ids([CLS_BERT])
      word_piece_lemmas = self.bert_tokenizer.convert_tokens_to_ids([CLS_BERT])
      if normalization:
        is_first_bpe_of_norm = [-1]
        word_piece_normalization = self.bert_tokenizer.convert_tokens_to_ids([CLS_BERT])
      else:
        is_first_bpe_of_norm = []
        word_piece_normalization = []
    else:
      is_first_bpe_of_norm = []
      is_first_bpe_of_token = []
      is_first_bpe_of_words = []

      word_piece_raw_tokens = []
      word_piece_raw_tokens_index = []

      word_piece_raw_tokens_aligned = []
      word_piece_raw_tokens_aligned_index = []

      word_piece_words = []
      word_piece_words_index = []

      word_piece_lemmas = []
      word_piece_lemmas_index = []

      word_piece_normalization = []
      word_piece_normalization_index = []

    if symbolic_root:
      words.append(ROOT)
      word_ids.append(self.__word_dictionary.get_index(ROOT))
      if normalization:
        norm_words.append(ROOT)
      if self.__word_norm_dictionary is not None:
        norm_word_ids.append(self.__word_norm_dictionary.get_index(ROOT))
      char_seqs.append([ROOT_CHAR, ])
      char_id_seqs.append([self.__char_dictionary.get_index(ROOT_CHAR), ])
      lemmas.append(ROOT)

      char_norm_id_seqs.append([self.__char_dictionary.get_index(ROOT_CHAR), ])
      char_norm_str_seq.append(([ROOT_CHAR, ]))
      #lemma_ids.append(self.__lemma_dictionary.get_index(ROOT))
      postags.append(ROOT_POS)
      pos_ids.append(self.__pos_dictionary.get_index(ROOT_POS))
      xpostags.append(ROOT_POS)
      xpos_ids.append(self.__xpos_dictionary.get_index(ROOT_POS))

      types.append(ROOT_TYPE)
      type_ids.append(self.__type_dictionary.get_index(ROOT_TYPE))
      heads.append(ROOT_HEADS_INDEX)

    for tokens in lines:

      # reading a MWE : we append to the raw tokens
      if '-' in tokens[0] or '.' in tokens[0]:
        matching_mwe_ind = re.match("([0-9]+)-([0-9]+)", tokens[0])

        assert matching_mwe_ind is not None, "ERROR : tokens[0] {} - or . byt did not match mwe pattern".format(tokens[0])
        mwe = self.bert_tokenizer.tokenize_origin(tokens[1])[0]

        all_indexes.append(tokens[0])

        word_piece_raw_tokens.extend(self.bert_tokenizer.convert_tokens_to_ids(mwe))
        # we add indexes range to highlight MWE
        word_piece_raw_tokens_index.extend([tokens[0] for _ in mwe])

        is_mwe.append(1)
        is_mwe.extend([-1 for _ in range(len(mwe)-1)])

        is_first_bpe_of_token.append(1)
        is_first_bpe_of_token.extend([0 for _ in range(len(mwe)-1)])

        word_piece_raw_tokens_aligned.extend(self.bert_tokenizer.convert_tokens_to_ids(mwe))
        word_piece_raw_tokens_aligned_index.extend([tokens[0] for _ in mwe])
        index_mwe = tokens[0]



        id_stop_mwe = eval(matching_mwe_ind.group(2))
        assert isinstance(id_stop_mwe, int), "ERROR : {} not int while it should".format(id_stop_mwe)
        id_start_mwe =eval(matching_mwe_ind.group(1))
        continue

      if len(tokens) < 10:
        sys.stderr.write("Sentence broken for unkwown reasons {} \n {} ".format(tokens, lines))
        if os.environ.get("EXPERIENCE") is not None:
          print("WARNING : WRITING corrupted gold data in {} ".format(os.path.join(os.environ["EXPERIENCE"], "logs/catching_errors.txt")))
          open(os.path.join(os.environ["EXPERIENCE"], "logs/catching_errors.txt"), "a").write("Line broken {} because of tokens {} from {} file \n ".format(lines, tokens,self.__file_path))
        continue

      n_words = tokens[0]
      all_indexes.append(tokens[0])

      # is_mwe labels : 1 label per words (not raw token but tokenized words)

      n_exception = 0
      if normalization:
        # includes sequence level and word level
        normalized_token, n_exception = get_normalized_token(norm_field=tokens[9], n_exception=n_exception,
                                                             predict_mode_only=not must_get_norm,
                                                             verbose=verbose)
        if self.bert_tokenizer is not None:
          normalized_token = self.bert_tokenizer.tokenize_origin(normalized_token)[0]

          is_first_bpe_of_norm.append(1)
          is_first_bpe_of_norm.extend([0 for _ in range(len(normalized_token)-1)])

          word_piece_normalization.extend(self.bert_tokenizer.convert_tokens_to_ids(normalized_token))
          word_piece_normalization_index.extend([tokens[0] for _ in range(normalized_token)])
        if self.case is not None and self.case == "lower":
          normalized_token = normalized_token.lower()
        # extracting normalized words as sequence of characters as string and ids, string and ids
        if word_decoder:
          normalized_token_id = self.__word_norm_dictionary.get_index(normalized_token)
          norm_word_ids.append(normalized_token_id)
        else:
          normalized_token_id = None

        norm_words.append(normalized_token)
        char_norm_ids = []
        char_norm_str = []

        for char in normalized_token:
          char_norm_ids .append(self.__char_dictionary.get_index(char))
          char_norm_str.append(char)

        if len(char_norm_ids) > self.max_char_len:
          char_norm_ids = char_norm_ids[:self.max_char_len]
          char_norm_str = char_norm_str[:self.max_char_len]

        char_norm_str_seq.append(char_norm_str)
        char_norm_id_seqs.append(char_norm_ids)

        printing("Normalized word is {} encoded as {} "
                 "normalized character sequence is {} "
                 "encoded as {} ".format(normalized_token, normalized_token_id, char_norm_str_seq,
                                         char_norm_id_seqs), verbose_level=6, verbose=verbose)
      chars = []
      char_ids = []

      for char in tokens[1]:
        chars.append(char)
        char_ids.append(self.__char_dictionary.get_index(char))
      # we cut the characters in regard to the GENERAL MAX_CHAR_LENGTH (not bucket specific)
      if len(chars) > self.max_char_len:
        chars = chars[:self.max_char_len]
        char_ids = char_ids[:self.max_char_len]
      char_seqs.append(chars)
      char_id_seqs.append(char_ids)
      #pdb.set_trace()
      #sys.stderr.write("CHAR FILLED \n")
      _word = tokens[1]

      if self.bert_tokenizer is not None:
        bpe_word = self.bert_tokenizer.tokenize_origin(_word)[0]
        word_piece_words.extend(self.bert_tokenizer.convert_tokens_to_ids(bpe_word))

        word_piece_words_index.extend([tokens[0] for _ in bpe_word])
        is_first_bpe_of_words.append(1)
        is_first_bpe_of_words.extend([0 for _ in range(len(bpe_word)-1)])
        # lemmas
        bpe_lemma = self.bert_tokenizer.tokenize_origin(tokens[2])[0]
        word_piece_lemmas_index.extend([tokens[0] for _ in bpe_lemma])
        word_piece_lemmas.extend(self.bert_tokenizer.convert_tokens_to_ids(bpe_lemma))

        # if we are not in a mwe we add every tokens in raw tokens
        if eval(tokens[0]) > id_stop_mwe:
          mwe_splits_save = []
          bert_pre_tokens = self.bert_tokenizer.tokenize_origin(tokens[1])[0]

          word_piece_raw_tokens_index.extend([tokens[0] for _ in bert_pre_tokens])
          word_piece_raw_tokens.extend(self.bert_tokenizer.convert_tokens_to_ids(bert_pre_tokens))

          word_piece_raw_tokens_aligned_index.extend([tokens[0] for _ in bert_pre_tokens])
          word_piece_raw_tokens_aligned.extend(self.bert_tokenizer.convert_tokens_to_ids(bert_pre_tokens))

          is_mwe.append(0)
          is_mwe.extend([-1 for _ in range(len(bert_pre_tokens)-1)])
          n_masks_to_add_in_raw_label.append(0)
          n_masks_to_add_in_raw_label.extend([-1 for _ in range(len(bert_pre_tokens) - 1)])
          is_first_bpe_of_token.append(1)
          is_first_bpe_of_token.extend([0 for _ in range(len(bert_pre_tokens) - 1)])
        # if we are reading words that are comming within a MWE we save them to know the alignement
        # when we reached the end of the MWE we add the numbers of masks necessary
        # to align bpes of raw sentence and  bpes of tokenized sequence
        elif id_start_mwe <= eval(tokens[0]) <= id_stop_mwe:
          mwe_splits_save.append(tokens[1])
          if eval(tokens[0]) == id_stop_mwe:
            mwe_splits_save = self.bert_tokenizer.tokenize_origin(" ".join(mwe_splits_save))[0]
            n_masks_to_add_in_raw = len(mwe_splits_save)-len(mwe)
            assert n_masks_to_add_in_raw >= 0, "ERROR : n_masks_to_add_in_raw should be an int : pb with tokens {} ".format(tokens)
            # we index masks inserted it in the sequence as -1
            word_piece_raw_tokens_aligned_index.extend([index_mwe for _ in range(n_masks_to_add_in_raw)])
            word_piece_raw_tokens_aligned.extend(self.bert_tokenizer.convert_tokens_to_ids([MASK_BERT for _ in range(n_masks_to_add_in_raw)]))

            n_masks_to_add_in_raw_label.append(n_masks_to_add_in_raw)
            n_masks_to_add_in_raw_label.extend([-1 for _ in range(len(mwe)-1)])

      if self.case is not None and self.case == "lower":
        _word = _word.lower()
      words.append(_word)
      lemmas.append(tokens[2])
      #sys.stderr.write("LEMMAS  FILLED \n")
      #word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
      word = DIGIT_RE.sub(b"0", str.encode(_word)).decode()
      word_ids.append(self.__word_dictionary.get_index(word))
      #lemma_ids.append(self.__lemma_dictionary.get_index(tokens[2]))
      pos = tokens[3]# if tokens[4]=='_' else tokens[3]+'$$$'+tokens[4]
      if "pos" in tasks or "all" in tasks:
        assert pos != "_", "ERROR : pos not found for line {} ".format(lines)
      xpos = tokens[4]
      postags.append(pos)
      xpostags.append(xpos)
      pos_ids.append(self.__pos_dictionary.get_index(pos))
      xpos_ids.append(self.__xpos_dictionary.get_index(xpos))
      head = tokens[6]
      type = tokens[7]
      if "parsing" in tasks:
        assert head != "_", "ERROR : head not found for line {} while tasks is {}".format(lines, tasks)
        assert type != "_", "ERROR : type not found for line {} while tasks is {}".format(lines, tasks)
      types.append(type)
      type_ids.append(self.__type_dictionary.get_index(type))
      heads.append(head)


    if symbolic_end:
      words.append(END)
      word_ids.append(self.__word_dictionary.get_index(END))
      if normalization:
        norm_words.append(END)
      if self.__word_norm_dictionary is not None:
        norm_word_ids.append(self.__word_norm_dictionary.get_index(END))

      char_seqs.append([END, ])
      char_id_seqs.append([self.__char_dictionary.get_index(END), ])

      char_norm_id_seqs.append([self.__char_dictionary.get_index(END), ])
      char_norm_str_seq.append(([END, ]))

      postags.append(END_POS)
      xpostags.append(END_POS)
      pos_ids.append(self.__pos_dictionary.get_index(END_POS))
      xpos_ids.append(self.__xpos_dictionary.get_index(END_POS))
      types.append(END_TYPE)
      type_ids.append(self.__type_dictionary.get_index(END_TYPE))
      heads.append(END_HEADS_INDEX)
      is_mwe.append(-1)
      n_masks_to_add_in_raw_label.append(-1)


    if self.bert_tokenizer is not None:

      is_first_bpe_of_words.append(-1)
      is_first_bpe_of_token.append(-1)
      if normalization:
          is_first_bpe_of_norm.append(-1)

      # we add one indx for SEP token
      word_piece_normalization_index.append(int(n_words)+1)
      word_piece_raw_tokens_index.append(int(n_words)+1)
      word_piece_raw_tokens_aligned_index.append(int(n_words)+1)
      word_piece_words_index.append(int(n_words)+1)
      word_piece_lemmas_index.append(int(n_words)+1)
      all_indexes.append(str(int(n_words)+1))

      word_piece_raw_tokens.extend(self.bert_tokenizer.convert_tokens_to_ids([SEP_BERT]))
      word_piece_raw_tokens_aligned.extend(self.bert_tokenizer.convert_tokens_to_ids([SEP_BERT]))
      word_piece_words.extend(self.bert_tokenizer.convert_tokens_to_ids([SEP_BERT]))
      word_piece_lemmas.extend(self.bert_tokenizer.convert_tokens_to_ids([SEP_BERT]))

      if normalization:
        word_piece_normalization.extend(self.bert_tokenizer.convert_tokens_to_ids([SEP_BERT]))

      sentence_word_piece = SentenceWordPieced(word_piece_lemmas=word_piece_lemmas,
                                               word_piece_normalization=word_piece_normalization,
                                               word_piece_raw_tokens_aligned=word_piece_raw_tokens_aligned,
                                               word_piece_raw_tokens=word_piece_raw_tokens,
                                               word_piece_words=word_piece_words,
                                               is_mwe=is_mwe, n_masks_to_add_in_raw_label=n_masks_to_add_in_raw_label,
                                               word_piece_raw_tokens_aligned_index=word_piece_raw_tokens_aligned_index,
                                               word_piece_words_index=word_piece_words_index,
                                               word_piece_raw_tokens_index=word_piece_raw_tokens_index,
                                               is_first_bpe_of_token=is_first_bpe_of_token,
                                               is_first_bpe_of_norm=is_first_bpe_of_norm,
                                               is_first_bpe_of_words=is_first_bpe_of_words,
                                               )
      sentence_word_piece.sanity_check_len(normalization=normalization, n_words=n_words)
    else:
      sentence_word_piece = None

    return DependencyInstance(Sentence(words, word_ids, char_seqs,char_id_seqs,
                                       [lines, raw_text],
                                       all_indexes=all_indexes,
                                       word_norm=norm_words,
                                       word_norm_ids=norm_word_ids,
                                       char_norm_ids_seq=char_norm_id_seqs,
                                       char_norm_seq=char_norm_str_seq),
                              postags, pos_ids, xpostags, xpos_ids, lemmas, lemma_ids, heads, types, type_ids,
                              sentence_word_piece)

# TODO : add end begin symbol both for character sequence and normalized character sequence




# hid is as ['0', '3', '1'] ? why '' ??