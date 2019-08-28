from env.importing import os, codecs, torch, np, Variable, pdb

from io_.info_print import printing
from .constants import MAX_CHAR_LENGTH, NUM_CHAR_PAD, PAD_CHAR, PAD_POS, PAD_TYPE, ROOT_CHAR, ROOT_POS, PAD, \
  ROOT_TYPE, END_CHAR, END_POS, END_TYPE, _START_VOCAB, ROOT, PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG, DIGIT_RE, CHAR_START_ID, CHAR_START, CHAR_END_ID, PAD_ID_CHAR, PAD_ID_NORM_NOT_NORM, END,\
  MEAN_RAND_W2V, SCALE_RAND_W2V, PAD_ID_EDIT, PAD_ID_HEADS, PAD_ID_BERT, PAD_ID_LOSS_STANDART
from env.project_variables import W2V_LOADED_DIM, MAX_VOCABULARY_SIZE_WORD_DIC
from .conllu_reader import CoNLLReader
from .dictionary import Dictionary
from io_.dat.conllu_get_normalization import get_normalized_token
from io_.signal_aggregation import get_transform_normalized_standart


def load_dict(dict_path, train_path=None, dev_path=None, test_path=None,
              word_normalization=False, pos_specific_data_set=None,
              word_embed_dict=None, tasks=None,
              dry_run=0, expand_vocab=False, add_start_char=None,
              case=None,
              force_new_dic=False, verbose=1):

  # TODO : CLEAN THIS to_create
  to_create = False
  for dict_type in ["word", "character", "pos", "xpos", "type"]:
    if not os.path.isfile(os.path.join(dict_path, "{}.json".format(dict_type))):
      to_create = True
      printing("WARNING : did not find dictionaries {} so creating it ", var=[dict_path], verbose=verbose, verbose_level=1)

  to_create = True if force_new_dic else to_create

  if to_create:
    assert isinstance(train_path, list) and (isinstance(dev_path, list) or dev_path is None), \
      "ERROR : TRAIN:{} not list or DEV:{} not list ".format(train_path, dev_path)
    assert train_path is not None and dev_path is not None and add_start_char is not None

    printing("Creating dictionary in {} ".format(dict_path), verbose=verbose, verbose_level=1)
    word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, \
    xpos_dictionary, type_dictionary = create_dict(dict_path,
                                                   train_path=train_path,
                                                   dev_path=dev_path,
                                                   test_path= test_path,  dry_run=dry_run,
                                                   word_embed_dict=word_embed_dict,
                                                   expand_vocab_bool=expand_vocab, add_start_char=add_start_char,
                                                   pos_specific_data_set=pos_specific_data_set,
                                                   tasks=tasks,case=case,
                                                   word_normalization=word_normalization, verbose=verbose)
  else:
    # ??
    assert train_path is None and dev_path is None and test_path is None and add_start_char is None, \
      "train_path {} dev_path {} test_path {} add_start_char {}".format(train_path,
                                                                        dev_path,
                                                                        test_path,
                                                                        add_start_char)
    printing("Loading dictionary from {} ".format(dict_path), verbose=verbose, verbose_level=1)
    word_dictionary = Dictionary('word', default_value=True, singleton=True)
    word_norm_dictionary = Dictionary('word_norm', default_value=True, singleton=True) if word_normalization else None
    char_dictionary = Dictionary('character', default_value=True)
    pos_dictionary = Dictionary('pos', default_value=True)
    xpos_dictionary = Dictionary('xpos', default_value=True)
    type_dictionary = Dictionary('type', default_value=True)
    dic_to_load_names = ["word", "character", "pos", "xpos", "type"]
    dict_to_load = [word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary]
    if word_normalization:
      dic_to_load_names.append("word_norm")
      dict_to_load.append(word_norm_dictionary)
    for name, dic in zip(dic_to_load_names, dict_to_load):
      dic.load(input_directory=dict_path, name=name)

  return word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary


def pos_specific_dic_builder(pos_specific_data_set, pos_dictionary):
  if pos_specific_data_set is not None:
    assert os.path.exists(pos_specific_data_set), "{} does not exist".format(pos_specific_data_set)
    with codecs.open(pos_specific_data_set, 'r', 'utf-8', errors='ignore') as file:
      li = 0
      for line in file:
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
          continue
        tokens = line.split('\t')
        if '-' in tokens[0] or '.' in tokens[0]:
          continue
        pos = tokens[3]  # if tokens[4]=='_' else tokens[3]+'$$$'+tokens[4]
        #xpos = tokens[4]
        pos_dictionary.add(pos)
        #xpos_dictionary.add(xpos)
    printing("VOCABULARY : POS Vocabulary : pos dictionary built on {} ".format(pos_specific_data_set), verbose_level=1, verbose=1)
    return pos_dictionary
  printing("VOCABULARY : POS Vocabulary : pos dictionary untouched", verbose_level=1, verbose=1)
  return pos_dictionary


def create_dict(dict_path, train_path, dev_path, test_path, tasks,
                dry_run, word_normalization=False, expand_vocab_bool=False, add_start_char=0,
                min_occurence=0, pos_specific_data_set=None,word_embed_dict=None, case=None,
                verbose=1,
               ):
  """
  Given train, dev, test treebanks and a word embedding matrix :
  - basic mode : create key_value instanes for each CHAR, WORD, U|X-POS , Relation with special cases for Roots, Padding and End symbols
  - expanding is done on dev set (we assume that dev set is accessible)
  - min_occurence : if <= considered as singleton otherwise ad
  - if expand_vocab == True : we also perform expansion on test set if test_path is not None and on dev_path
  - DEPRECIATED : based on word_embed_dict a new numpy matrix is created that will be used to th : ONLY expansion decision made on word_embed_dict
  - if pos_specific_data_set not None  :
      - build pos_dictionary from it
      - expand word dictionaries with it
  #WARNING singleton as been removed in a hardcoded way cause it was not clear what it was doing/done for

  TODO : to be tested : test based on a given conll --> vocab word is correct
      in regard to min_occurence and that the created matrix is correct also   (index --> vetor correct
  """
  printing("WARNING : CASING IS {} for dictionary ", var=[case], verbose=verbose, verbose_level=1)
  default_value = True
  if word_embed_dict is None:
    word_embed_dict = {}
  word_dictionary = Dictionary('word', default_value=default_value, singleton=True)
  word_norm_dictionary = Dictionary('word_norm', default_value=default_value, singleton=True) if word_normalization else None
  char_dictionary = Dictionary('character', default_value=default_value)
  pos_dictionary = Dictionary('pos', default_value=default_value)
  xpos_dictionary = Dictionary('xpos', default_value=default_value)
  type_dictionary = Dictionary('type', default_value=default_value)
  counter_match_train = 0
  counter_other_train = 0
  char_dictionary.add(PAD_CHAR)

  if add_start_char:
    char_dictionary.add(CHAR_START)

  char_dictionary.add(ROOT_CHAR)
  char_dictionary.add(END_CHAR)
  char_dictionary.add(END)

  pos_dictionary.add(PAD_POS)
  xpos_dictionary.add(PAD_POS)
  type_dictionary.add(PAD_TYPE)

  pos_dictionary.add(ROOT_POS)
  xpos_dictionary.add(ROOT_POS)
  type_dictionary.add(ROOT_TYPE)

  pos_dictionary.add(END_POS)
  xpos_dictionary.add(END_POS)
  type_dictionary.add(END_TYPE)

  vocab = dict()
  vocab_norm = dict()
  # read training file add to Vocab directly except for words (not word_norm)
  # ## for which we need filtering so we add them to vocab()

  assert isinstance(train_path, list)
  assert tasks is not None, "ERROR : we need tasks information along with dataset to know how to commute label dictionary"

  for train_dir, simultaneous_task_ls in zip(train_path, tasks):
    printing("VOCABULARY : computing dictionary for word, char on {} for task {} ", var=[train_dir, simultaneous_task_ls], verbose=verbose, verbose_level=1)
    if len(set(simultaneous_task_ls) & set(["normalize", "all"])) > 0:
      printing("VOCABULARY : computing dictionary for normalized word also {} ", var=[train_dir, simultaneous_task_ls], verbose=verbose, verbose_level=1)
    # NB if pos OR parsing is here we actually compte the Dictionary for pos AND parsing
    elif len(set(simultaneous_task_ls) & set(["pos", "parsing", "all"])) > 0:
      printing("VOCABULARY : computing dictionary for pos and/or parsing word also ", verbose=verbose, verbose_level=1)
    with codecs.open(train_dir, 'r', 'utf-8', errors='ignore') as file:
      li = 0
      for line in file:
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
          continue
        tokens = line.split('\t')
        if '-' in tokens[0] or '.' in tokens[0]:
          continue
        for char in tokens[1]:
          char_dictionary.add(char)
        word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
        if case == "lower":
          word = word.lower()
        pos = tokens[3]  #if tokens[4]=='_' else tokens[3]+'$$$'+tokens[4]
        xpos = tokens[4]
        typ = tokens[7]
        #if pos_specific_data_set is None:
        # otherwise : pos-dictionary will be build with pos_specific_data_set
        #  pos_dictionary.add(pos)
        if len(set(simultaneous_task_ls) & set(["all", "pos", "parsing"]))>0:
          pos_dictionary.add(pos)
          xpos_dictionary.add(xpos)
          type_dictionary.add(typ)
        if word_normalization and len(set(simultaneous_task_ls) & set(["normalize", "all"])) > 0:
          token_norm, _ = get_normalized_token(tokens[9], 0, verbose=verbose)
          if case == "lower":
            token_norm = token_norm.lower()
          if token_norm in vocab_norm:
            vocab_norm[token_norm] += 1
          else:
            vocab_norm[token_norm] = 1
        if word in vocab:
          vocab[word] += 1
        else:
          vocab[word] = 1
        li = li + 1
        if dry_run and li == 100:
          break
  # collect singletons
  singletons = set([word for word, count in vocab.items() if count <= min_occurence])
  # if a singleton is in pretrained embedding dict, set the count to min_occur + c
  for word in vocab.keys():
    if word in word_embed_dict or word.lower() in word_embed_dict:
      # if words are in word_embed_dict we want them even they appear less then min_occurence
      vocab[word] += min_occurence
  for word_norm in vocab_norm.keys():
    # TODO : should do something if we allow word embedding on the target standart side
    pass
    #if word in word_embed_dict or word.lower() in word_embed_dict:
  vocab_norm_list = _START_VOCAB + sorted(vocab_norm, key=vocab_norm.get, reverse=True)
  # WARNING / same min_occurence for source and target word vocabulary
  vocab_norm_list = [word for word in vocab_norm_list if word in _START_VOCAB or vocab_norm[word] > min_occurence]
  if len(vocab_norm_list) > MAX_VOCABULARY_SIZE_WORD_DIC:
    printing("VOCABULARY : norm vocabulary cut to {}  tokens", var=[MAX_VOCABULARY_SIZE_WORD_DIC],verbose=verbose, verbose_level=1)
    vocab_norm_list = vocab_norm_list[:MAX_VOCABULARY_SIZE_WORD_DIC]

  vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
  # filter strictly above min_occurence
  vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurence]
  max_vocabulary_size = MAX_VOCABULARY_SIZE_WORD_DIC
  if len(vocab_list) > max_vocabulary_size:
    printing("VOCABULARY : target vocabulary cut to {} tokens", var=[MAX_VOCABULARY_SIZE_WORD_DIC], verbose=verbose, verbose_level=1)
    vocab_list = vocab_list[:max_vocabulary_size]
  word_dictionary.inv_ls = vocab_list
  printing("VOCABULARY INV added {} ".format(vocab_list), verbose_level=3, verbose=verbose)

  pos_dictionary = pos_specific_dic_builder(pos_specific_data_set, pos_dictionary)

  def expand_vocab(data_paths):
    counter_match_dev = 0
    expand = 0
    vocab_set = set(vocab_list)
    vocab_norm_set = set(vocab_norm_list)
    for data_path in data_paths:
      with codecs.open(data_path, 'r', 'utf-8', errors='ignore') as file:
        li = 0
        for line in file:
          line = line.strip()
          if len(line) == 0 or line[0] == '#':
            continue
          tokens = line.split('\t')
          if '-' in tokens[0] or '.' in tokens[0]:
            continue
          for char in tokens[1]:
            char_dictionary.add(char)
          word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
          if case == "lower":
            word = word.lower()
          pos = tokens[3] # if tokens[4]=='_' else tokens[3]+'$$$'+tokens[4]
          xpos = tokens[4]
          typ = tokens[7]
          # TODO SOMEHITNG
          if word_normalization:
            token_norm, _ = get_normalized_token(tokens[9], 0, verbose=0)
          if word_normalization:
            # TODO : add word_norm_embed_dict to allow expansion !
            if False and word_norm not in vocab_norm_set:
              vocab_norm_set.add(word_norm)
              vocab_norm_list.append(word_norm)
          # TODO : ANswer : WHY WOULD WE LIKE TO EXPAND IT ON DEV, TEST ?
          #if pos_specific_data_set is None:
          #  pos_dictionary.add(pos)
          #xpos_dictionary.add(xpos)
          #type_dictionary.add(typ)
          # if word not already in vocab_set (loaded as trained and each time expand_vocab was called :
          # but found in new dataset and appear in word_embed_dict then we add it to vocab # otherwise not need to load them to vocab (they won't have any representation)
          # but found in new dataset and appear in word_embed_dict then we add it to vocab # otherwise not need to load them to vocab (they won't have any representation)
          if word not in vocab_set and (word in word_embed_dict or word.lower() in word_embed_dict):
            vocab_set.add(word)
            expand += 1
            vocab_list.append(word)
          li = li + 1
          if dry_run and li == 100:
            break
        printing("VOCABULARY EXPAND word source vocabulary expanded of {} tokens based on {} ", var=[expand, data_path],
                 verbose=verbose, verbose_level=0)
  if expand_vocab_bool:
    assert len(word_embed_dict)>0, "ERROR : how do you want to expand if no wod embedding dict"
    if isinstance(dev_path, str):
      dev_path = [dev_path]
    expand_vocab(dev_path)
    printing("VOCABULARY : EXPANDING vocabulary on {} ", var=[dev_path], verbose_level=0, verbose=verbose)
    if test_path is not None:
      if isinstance(test_path, str):
        test_path = [test_path]
      printing("VOCABULARY : EXPANDING vocabulary on {} ", var=[test_path], verbose_level=0, verbose=verbose)
      expand_vocab(test_path)
      # TODO : word_norm should be handle spcecifically
  # TODO : what is singletons for ?
  singletons = []
  if word_norm_dictionary is not None:
    for word_norm in vocab_norm_list:
      word_norm_dictionary.add(word_norm)

  for word in vocab_list:
    word_dictionary.add(word)
    if word in word_embed_dict:
      counter_match_train += 1
    else:
      counter_other_train +=1
    if word in singletons :
      word_dictionary.add_singleton(word_dictionary.get_index(word))
  word_dictionary.save(dict_path)
  if word_norm_dictionary is not None:
    word_norm_dictionary.save(dict_path)
    word_norm_dictionary_size = word_norm_dictionary.size()
    word_norm_dictionary.close()
  else:
    word_norm_dictionary_size = 0
  char_dictionary.save(dict_path)
  pos_dictionary.save(dict_path)
  xpos_dictionary.save(dict_path)
  type_dictionary.save(dict_path)
  word_dictionary.close()
  char_dictionary.close()
  pos_dictionary.close()
  xpos_dictionary.close()
  type_dictionary.close()
  if word_embed_dict != {}:
    printing("VOCABULARY EXPANSION Match with preexisting word embedding {} match in train and dev and {} no match tokens ".format(
    counter_match_train, counter_other_train), verbose=1, verbose_level=1)
  else:
    printing(
      "VOCABULARY WORDS was not expanded on dev or test cause no external word embedding dict wa provided", verbose=1, verbose_level=1)
  printing("VOCABULARY : {} word {} word_norm {} char {} xpos {} pos {} type encoded in vocabulary "
           "(including default token, an special tokens)",
           var=[word_dictionary.size(), word_norm_dictionary_size, char_dictionary.size(), xpos_dictionary.size(),
                pos_dictionary.size(), type_dictionary.size()],
           verbose=verbose, verbose_level=0)

  return word_dictionary, word_norm_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary


def read_data(source_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary, type_dictionary, max_size=None,
              word_norm_dictionary=None,
              normalize_digits=True, word_decoder=False, 
              normalization=False, bucket=False, max_char_len=None,
              symbolic_root=False, symbolic_end=False, dry_run=False, tasks=None,
              must_get_norm=True, bert_tokenizer=None, bucketing_level=None,
              verbose=0):
  """
  Given vocabularies , data_file :
  - creates a  list of bucket
  - each bucket is a list of unicode encoded worrds, character, pos tags, relations, ... based on DependancyInstances()
   and Sentence() objects
  """
  if bucketing_level is None:
    bucketing_level = "wordpiece"
  assert bucketing_level in ["word", "wordpiece"], "ERROR : {} should be word or wordpiece"
  printing("INFO ITERATOR : bucketing is done based on {} level", var=[bucketing_level], verbose_level=1, verbose=verbose)
  if bucket:
    _buckets = [20, 50, 100, -1]
    # in bpe
    buckets_length_bpe_words = [20, 50, 100, -1]

    #printing("WARNING : bucket limited to 40", verbose=verbose, verbose_level=1)
  else:
    buckets_length_bpe_words = [-1]
    _buckets = [-1]
    printing("WARNING : for validation we don't bucket the data : bucket len is {} (-1 means will be based "
             "on max sent length lenght) ", var=_buckets[0], verbose=verbose, verbose_level=1)

  assert len(_buckets) == len(buckets_length_bpe_words), "ERROR bucket word level and bpe level should be same len "

  if buckets_length_bpe_words[-1] == -1 or _buckets[-1] == -1:
      assert buckets_length_bpe_words[-1] == _buckets[-1], \
        "ERROR : if last bucket is -1 should be the same for both word buckt and bpe bucket"

  printing('DATA READER : Reading data from %s' % source_path, verbose_level=1, verbose=verbose)
  counter = 0
  reader = CoNLLReader(source_path, word_dictionary, char_dictionary, pos_dictionary, type_dictionary, xpos_dictionary,
                       max_char_len=max_char_len, bert_tokenizer=bert_tokenizer,
                       lemma_dictionary=None, word_norm_dictionary=word_norm_dictionary)
  printing("DATA iterator based on {} tasks", var=tasks, verbose_level=1, verbose=verbose)

  inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end, must_get_norm=must_get_norm, word_decoder=word_decoder, tasks=tasks)

  if bucketing_level == "word":
    _buck = _buckets
  elif bucketing_level == "wordpiece":
    _buck = buckets_length_bpe_words
  last_bucket_id = len(_buck ) - 1
  data = [[] for _ in _buck ]
  max_char_length = [0 for _ in _buck]
  max_char_norm_length = [0 for _ in _buck] if normalization else None

  while inst is not None and (not dry_run or counter < 100):
    if inst == "CORRUPTED":
      inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end,
                            must_get_norm=must_get_norm, word_decoder=word_decoder, tasks=tasks)
      print("WARNING skipping one corrupted sentences in reader")
      continue

    printing("Sentence : counter {} inst : {}".format(counter, inst.sentence.raw_lines[1]),
             verbose=verbose, verbose_level=5)

    sent = inst.sentence
    sent_word_piece = inst.sentence_word_piece


    if bucketing_level == "word":
      inst_size = inst.length()
    elif bucketing_level == "wordpiece":
      inst_size = inst.sentence_word_piece.length()

    for bucket_id, bucket_size in enumerate(_buck):
      if inst_size < bucket_size or bucket_id == last_bucket_id:
        if len(sent.word_ids) > _buckets[bucket_id]:
          print("WARNING : LENTH adjusting bucket id {} of len {} to len _bucket {}".format(bucket_id, _buckets[bucket_id], len(sent.word_ids)+5))
          _buckets[bucket_id] = len(sent.word_ids)+5
        data[bucket_id].append([sent.all_indexes, sent.word_ids, sent.word_norm_ids, sent.char_id_seqs, sent.char_norm_ids_seq, inst.pos_ids, inst.heads, inst.type_ids,
                                counter, sent.words, sent.word_norm, sent.raw_lines, inst.xpos_ids,
                                sent_word_piece.word_piece_raw_tokens,
                                sent_word_piece.word_piece_raw_tokens_aligned,
                                sent_word_piece.word_piece_words,
                                sent_word_piece.word_piece_lemmas,
                                sent_word_piece.word_piece_normalization,
                                sent_word_piece.word_piece_raw_tokens_aligned_index,
                                sent_word_piece.word_piece_words_index,
                                sent_word_piece.word_piece_raw_tokens_index,
                                sent_word_piece.is_mwe,
                                sent_word_piece.n_masks_to_add_in_raw_label,
                                sent_word_piece.is_first_bpe_of_token,
                                sent_word_piece.is_first_bpe_of_norm,
                                sent_word_piece.is_first_bpe_of_words,
                                ])

        max_char_len = max([len(char_seq) for char_seq in sent.char_seqs])
        if normalization:
          max_char_norm_len = max([len(char_norm_seq) for char_norm_seq in sent.char_norm_ids_seq])
        # defining maximum characters lengh per bucket both for noralization and
        # we define a max_char_len per bucket !
        if max_char_length[bucket_id] < max_char_len:
          max_char_length[bucket_id] = max_char_len
        if normalization:
          if max_char_norm_length[bucket_id] < max_char_norm_len:
            max_char_norm_length[bucket_id] = max_char_norm_len

        if bucket_id == last_bucket_id and buckets_length_bpe_words[last_bucket_id] < len(sent_word_piece.word_piece_words):
          buckets_length_bpe_words[last_bucket_id] = len(sent_word_piece.word_piece_words)+2
          if _buckets[last_bucket_id] < len(sent_word_piece.word_piece_words):
            _buckets[bucket_id] = len(sent.word_ids)+2

          # we assumed that raw bpe were smaller or equal to words (assert will raise error otherwise in reader)
          # to do for norm also
        if buckets_length_bpe_words[bucket_id] == -1:
          buckets_length_bpe_words[bucket_id] = len(sent_word_piece.word_piece_words)+2
          _buckets[bucket_id] = len(sent.word_ids)+2
        break
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end,
                          must_get_norm=must_get_norm, word_decoder=word_decoder, tasks=tasks)
    counter += 1
    if inst is None or not (not dry_run or counter < 100):
      printing("Breaking : breaking because inst {} counter<100 {} dry {} ".format(inst is None, counter < 100, dry_run),
               verbose=verbose, verbose_level=3)
  reader.close()

  return data, {"buckets_length_bpe_words": buckets_length_bpe_words if bert_tokenizer is not None else None,
                "max_char_length": max_char_length,
                "max_char_norm_length": max_char_norm_length, "n_sent": counter}, _buck


def read_data_to_variable(source_path, word_dictionary, char_dictionary, pos_dictionary, xpos_dictionary,
                          type_dictionary, max_size=None, normalize_digits=True, symbolic_root=False,word_norm_dictionary=None,
                          symbolic_end=False, use_gpu=False, volatile=False, dry_run=False, lattice=None,args=None,
                          verbose=0, normalization=False, bucket=True, word_decoder=False,
                          tasks=None, max_char_len=None, must_get_norm=True, bert_tokenizer=None,
                          bucketing_level=None,
                          add_end_char=0, add_start_char=0):
  """
  Given data ovject form read_variable creates array-like  variables for character, word, pos, relation, heads ready to be fed to a network
  """
  if max_char_len is None:
    max_char_len = MAX_CHAR_LENGTH
  if "norm_not_norm" in tasks:
    assert normalization, "norm_not_norm can't be set without normalisation info"
  printing("WARNING symbolic root {} is and symbolic end is {} ", var=[symbolic_root, symbolic_end], verbose=verbose, verbose_level=1)
  data, max_char_length_dic, _buckets = read_data(source_path, word_dictionary, char_dictionary, pos_dictionary,
                                                  xpos_dictionary, type_dictionary, bucket=bucket, word_norm_dictionary=word_norm_dictionary,
                                                  verbose=verbose, max_size=max_size, normalization=normalization,
                                                  normalize_digits=normalize_digits, symbolic_root=symbolic_root,
                                                  word_decoder=word_decoder, tasks=tasks,max_char_len=max_char_len,
                                                  must_get_norm=must_get_norm,
                                                  bert_tokenizer=bert_tokenizer,
                                                  bucketing_level=bucketing_level,
                                                  symbolic_end=symbolic_end, dry_run=dry_run)

  max_char_length = max_char_length_dic["max_char_length"]
  max_char_norm_length = max_char_length_dic["max_char_norm_length"]

  printing("DATA MAX_CHAR_LENGTH set to {}".format(max_char_len), verbose=verbose, verbose_level=1)

  bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
  printing("DATA checking bucket sizes {}",var=[bucket_sizes], verbose=verbose, verbose_level=1)
  data_variable = []

  ss = [0] * len(_buckets)
  ss1 = [0] * len(_buckets)

  for bucket_id in range(len(_buckets)):
    bucket_size = bucket_sizes[bucket_id]
    if bucket_size == 0:
      data_variable.append((1, 1))
      continue
    bucket_length = _buckets[bucket_id]+10
    print("WARNING : adding 10 dimension to bucket_length "
          "for avoiding over loading of words buckets due to bpe bucketing (bucket id {})".format(bucket_id))
    if max_char_length_dic["buckets_length_bpe_words"] is not None:
      bucket_length_in_bpe = max_char_length_dic["buckets_length_bpe_words"][bucket_id]
    char_length = min(max_char_len+NUM_CHAR_PAD, max_char_length[bucket_id] + NUM_CHAR_PAD)
    if max_char_len+NUM_CHAR_PAD < max_char_length[bucket_id] + NUM_CHAR_PAD:
      printing("WARNING : Iterator conllu_data CUTTING bucket {} to max allowed length {}",
               var=[max_char_len+NUM_CHAR_PAD],
               verbose=verbose, verbose_level=1)
    wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
    cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
    xpid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
    hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
    tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

    wordpieces_inputs_raw_tokens = np.empty([bucket_size, bucket_length_in_bpe], dtype=np.int64) if max_char_length_dic["buckets_length_bpe_words"] is not None else None

    is_mwe_label = np.empty([bucket_size, bucket_length_in_bpe], dtype=np.int64) if max_char_length_dic["buckets_length_bpe_words"] is not None else None
    n_masks_to_app_in_raw_label = np.empty([bucket_size, bucket_length_in_bpe], dtype=np.int64) if max_char_length_dic["buckets_length_bpe_words"] is not None else None
    wordpieces_words = np.empty([bucket_size, bucket_length_in_bpe], dtype=np.int64) if max_char_length_dic["buckets_length_bpe_words"] is not None else None

    wordpieces_raw_aligned_with_words = np.empty([bucket_size, bucket_length_in_bpe], dtype=np.int64) if max_char_length_dic["buckets_length_bpe_words"] is not None else None

    ind_wordpieces_words_alignement_index = [] if max_char_length_dic["buckets_length_bpe_words"] is not None else None
    ind_wordpieces_inputs_raw_tokens_alignement_index = [] if max_char_length_dic["buckets_length_bpe_words"] is not None else None
    ind_wordpieces_raw_aligned_alignement_index = [] if max_char_length_dic["buckets_length_bpe_words"] is not None else None
    all_indexes = [] if max_char_length_dic["buckets_length_bpe_words"] is not None else None

    if normalization:
      char_norm_length = min(max_char_len+NUM_CHAR_PAD, max_char_norm_length[bucket_id] + NUM_CHAR_PAD)
      cids_norm = np.empty([bucket_size, bucket_length, char_norm_length], dtype=np.int64)
      if word_decoder:
        wid_norm_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
      if "norm_not_norm" in tasks:
        word_norm_not_norm = np.empty([bucket_size, bucket_length], dtype=np.int64)
      if "edit_prediction" in tasks:
        edit = np.empty([bucket_size, bucket_length], dtype=np.float32)

    masks_inputs = np.zeros([bucket_size, bucket_length], dtype=np.float32)
    single_inputs = np.zeros([bucket_size, bucket_length], dtype=np.int64)

    lengths_inputs = np.empty(bucket_size, dtype=np.int64)
    
    order_inputs = np.empty(bucket_size, dtype=np.int64)
    raw_word_inputs, raw_lines = [], []
    words_normalized_str = []

    for i, inst in enumerate(data[bucket_id]):
      ss[bucket_id] += 1
      ss1[bucket_id] = bucket_length
      indexes, wids, wids_norm, cid_seqs, cid_norm_seqs, pids, hids, tids, orderid, word_raw, normalized_str, lines, xpids, \
        word_piece_raw_tokens, word_piece_raw_tokens_aligned, word_piece_words,word_piece_lemmas, word_piece_normalization,\
        word_piece_raw_tokens_aligned_index, word_piece_words_index, word_piece_raw_tokens_index, \
        is_mwe, n_masks_to_add_in_raw_label, is_first_bpe_of_token, is_first_bpe_of_norm, is_first_bpe_of_words = inst
      # TODO : have to handle case were wids is null
      assert len(cid_seqs) == len(wids), "ERROR cid_seqs {} and wids {} are different len".format(cid_seqs, wids)
      if len(wids_norm) > 0 and normalization:
          assert len(wids_norm) == len(wids), "ERROR wids_norm {} and wids {} are different len".format(cid_seqs, wids)
      if len(cid_norm_seqs) > 0 and normalization:
          assert len(cid_seqs) == len(cid_norm_seqs), "ERROR cid_seqs {} and cid_norm_seqs {} have different length ".format(cid_seqs, cid_norm_seqs)
      inst_size = len(wids)#len(wids)
      lengths_inputs[i] = inst_size
      order_inputs[i] = orderid
      raw_word_inputs.append(word_raw)
      words_normalized_str.append(normalized_str)

      # bpe
      if word_piece_raw_tokens is not None and is_mwe is not None:
        inst_size_bpe_raw = len(word_piece_raw_tokens)# same len as is_mwe (sanity checked in reader)
        not_cutting = 1

        wordpieces_inputs_raw_tokens[i, :inst_size_bpe_raw] = word_piece_raw_tokens
        wordpieces_inputs_raw_tokens[i, inst_size_bpe_raw:] = PAD_ID_BERT
        n_masks_to_app_in_raw_label[i, :inst_size_bpe_raw] = n_masks_to_add_in_raw_label
        n_masks_to_app_in_raw_label[i, inst_size_bpe_raw:] = PAD_ID_LOSS_STANDART
        is_mwe_label[i, :inst_size_bpe_raw] = is_mwe
        is_mwe_label[i, inst_size_bpe_raw:] = PAD_ID_LOSS_STANDART
        # we cannot have range as int !!
        ind_wordpieces_inputs_raw_tokens_alignement_index.append(word_piece_raw_tokens_index+[PAD_ID_LOSS_STANDART for _ in range((bucket_length_in_bpe-inst_size_bpe_raw)*not_cutting)])
        #wordpieces_inputs_raw_tokens_alignement_index[i, inst_size_bpe_raw:] = PAD_ID_LOSS_STANDART

      if word_piece_raw_tokens_aligned is not None and word_piece_words is not None:
        inst_size_in_indexes = len(indexes)

        inst_size_bpe_word = len(word_piece_raw_tokens_aligned) # same len as word_piece_words (sanity checked in reader)
        # words indexes (can be used as input for tag/parse/norm or gold labels for tokenization
        no_cutting_word = 1
        wordpieces_words[i, :inst_size_bpe_word] = word_piece_words
        wordpieces_words[i, inst_size_bpe_word:] = PAD_ID_BERT
        # its bpe alignement index with source words
        # raw tokens aligned with words (inserted MASK)
        wordpieces_raw_aligned_with_words[i, :inst_size_bpe_word] = word_piece_raw_tokens_aligned
        wordpieces_raw_aligned_with_words[i, inst_size_bpe_word:] = PAD_ID_BERT
        # is it useful: yes
        ind_wordpieces_raw_aligned_alignement_index.append(word_piece_raw_tokens_aligned_index + [PAD_ID_LOSS_STANDART for _ in range((bucket_length_in_bpe - inst_size_bpe_word)*no_cutting_word)])
        #wordpieces_raw_aligned_alignement_index[i, :inst_size_bpe_word] = word_piece_raw_tokens_aligned_index
        #wordpieces_raw_aligned_alignement_index[i, inst_size_bpe_word:] = PAD_ID_LOSS_STANDART
        ind_wordpieces_words_alignement_index.append(word_piece_words_index + [PAD_ID_LOSS_STANDART for _ in range((bucket_length_in_bpe - inst_size_bpe_word)*no_cutting_word)])

        all_indexes.append(indexes+[str(PAD_ID_LOSS_STANDART) for _ in range(bucket_length+1-inst_size_in_indexes)])

      # word ids
      wid_inputs[i, :len(wids)] = wids
      wid_inputs[i, len(wids):] = PAD_ID_WORD
      # we assume word to word mapping for now
      if normalization and word_decoder:
        wid_norm_inputs[i, :inst_size] = wids_norm
        wid_norm_inputs[i, inst_size:] = PAD_ID_WORD

      shift, shift_end = 0, 0

      if add_start_char:
        shift += 1
      if add_end_char:
        shift_end += 1

      for w, cids in enumerate(cid_seqs):
        if add_start_char:
          cid_inputs[i, w, 0] = CHAR_START_ID

        cid_inputs[i, w, shift:len(cids)+shift] = cids

        if add_end_char:
          cid_inputs[i, w, len(cids)+shift] = CHAR_END_ID
        cid_inputs[i, w, shift+len(cids)+shift_end:] = PAD_ID_CHAR

      cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
      # TODO should factorize character sequence numpysation
      if normalization:
        for word_index, cids in enumerate(cid_norm_seqs):
          if add_start_char:
            cids_norm[i, word_index, 0] = CHAR_START_ID
          cids_norm[i, word_index, shift:len(cids)+shift] = cids
          if add_end_char:
            cids_norm[i, word_index, len(cids)+shift] = CHAR_END_ID
          #we want room to padd it
          cids_norm[i, word_index, len(cids)+shift+shift_end:] = PAD_ID_CHAR
          if "norm_not_norm" in tasks:
            word_norm_not_norm[i, word_index] = get_transform_normalized_standart(cids_norm, cid_inputs, sent_index=i,
                                                                                  word_index=word_index, task="norm_not_norm")
          if "edit_prediction" in tasks:
            edit[i, word_index] = get_transform_normalized_standart(cids_norm, cid_inputs, sent_index=i,
                                                                    word_index=word_index, task="edit_prediction")

        cids_norm[i, inst_size:, :] = PAD_ID_CHAR
        if "norm_not_norm" in tasks:
          word_norm_not_norm[i, inst_size:] = PAD_ID_NORM_NOT_NORM
        if "edit_prediction" in tasks:
          edit[i, inst_size:] = PAD_ID_EDIT

      # pos ids
      pid_inputs[i, :inst_size] = pids
      pid_inputs[i, inst_size:] = PAD_ID_TAG
      # xpos ids
      xpid_inputs[i, :inst_size] = xpids
      xpid_inputs[i, inst_size:] = PAD_ID_TAG
      # type ids
      tid_inputs[i, :inst_size] = tids
      tid_inputs[i, inst_size:] = PAD_ID_TAG
      # heads
      ONLY_PRED = False
      if not ONLY_PRED:
        try:
            hid_inputs[i, :inst_size] = hids
        except:
            printing("WARNING : ASSIGNING 1 to head id cause hids is {} ".format(hids), verbose_level=2, verbose=verbose)
            hid_inputs[i, :inst_size] = 1
        hid_inputs[i, inst_size:] = PAD_ID_HEADS
      #  hid_inputs[i, :0] = None
      #  hid_inputs[i, inst_size:] = None
      masks_inputs[i, :inst_size] = 1.0
      for j, wid in enumerate(wids):
        if word_dictionary.is_singleton(wid):
          single_inputs[i, j] = 1
      raw_lines.append(lines)
    words = Variable(torch.from_numpy(wid_inputs), requires_grad=False)
    chars = Variable(torch.from_numpy(cid_inputs), requires_grad=False)
    word_norm = Variable(torch.from_numpy(wid_norm_inputs), requires_grad=False) if normalization and word_decoder else None
    chars_norm = Variable(torch.from_numpy(cids_norm), requires_grad=False) if normalization else None
    word_norm_not_norm = Variable(torch.from_numpy(word_norm_not_norm), requires_grad=False) if "norm_not_norm" in tasks else None
    edit = Variable(torch.from_numpy(edit), requires_grad=False) if "edit_prediction" in tasks else None

    if word_piece_words is not None:
      wordpieces_words = Variable(torch.from_numpy(wordpieces_words), requires_grad=False)
      wordpieces_raw_aligned_with_words = Variable(torch.from_numpy(wordpieces_raw_aligned_with_words), requires_grad=False)

      is_mwe_label = Variable(torch.from_numpy(is_mwe_label), requires_grad=False)
      n_masks_to_app_in_raw_label = Variable(torch.from_numpy(n_masks_to_app_in_raw_label), requires_grad=False)
      wordpieces_inputs_raw_tokens = Variable(torch.from_numpy(wordpieces_inputs_raw_tokens), requires_grad=False)

      ind_wordpieces_raw_aligned_alignement_index = np.array(ind_wordpieces_raw_aligned_alignement_index)
      ind_wordpieces_inputs_raw_tokens_alignement_index = np.array(ind_wordpieces_inputs_raw_tokens_alignement_index)
      ind_wordpieces_words_alignement_index = np.array(ind_wordpieces_inputs_raw_tokens_alignement_index)

      all_indexes = np.array(all_indexes)

      # we don't put as pytorch alignement indexes
    pos = Variable(torch.from_numpy(pid_inputs), requires_grad=False)
    xpos = Variable(torch.from_numpy(xpid_inputs), requires_grad=False)
    heads = Variable(torch.from_numpy(hid_inputs), requires_grad=False)
    types = Variable(torch.from_numpy(tid_inputs), requires_grad=False)
    masks = Variable(torch.from_numpy(masks_inputs), requires_grad=False)
    single = Variable(torch.from_numpy(single_inputs), requires_grad=False)
    lengths = torch.from_numpy(lengths_inputs)

    if use_gpu:
      word_char_one_hot_in_gpu = False
      # not used anymore
      if word_char_one_hot_in_gpu:
        words = words.cuda()
        chars_norm = chars_norm.cuda() if normalization else None
        word_norm_not_norm = word_norm_not_norm.cuda() if "norm_not_norm" in tasks else None
        edit = edit.cuda() if "edit_prediction" in tasks else None
        word_norm = word_norm.cuda() if normalization and word_decoder else None
        chars = chars.cuda()
      else:
        printing("WARNING : we are not adding to cuda word, chars, norm , edit 1 hot encoding", verbose=verbose, verbose_level=1)
      pos = pos.cuda()
      xpos = xpos.cuda()
      heads = heads.cuda()
      types = types.cuda()
      masks = masks.cuda()
      #single = single.cuda()
      lengths = lengths.cuda()

      if word_piece_words is not None and ("parsing" in args.tasks or "pos" in args.tasks):
        wordpieces_words = wordpieces_words.cuda()
      else:
        printing("WARNING : we are not adding to cuda word pieces of words cause tasks is {} and neither parsing or pos found",
               var=[args.tasks], verbose=verbose,
               verbose_level=1)
      if word_piece_raw_tokens is not None and ("mwe_prediction" in args.tasks\
              or "mwe_detection" in args.tasks or "n_masks_mwe" in args.tasks):
          is_mwe_label = is_mwe_label.cuda()
          wordpieces_raw_aligned_with_words = wordpieces_raw_aligned_with_words.cuda()
          n_masks_to_app_in_raw_label = n_masks_to_app_in_raw_label.cuda()
          wordpieces_inputs_raw_tokens = wordpieces_inputs_raw_tokens.cuda()
      else:
        printing("WARNING : we are not adding to cuda word pieces of words cause tasks is {}"
                 " and neither mwe_prediction,mwe_detection,n_masks_mwe found", var=[args.tasks], verbose=verbose, verbose_level=1)
    data_variable.append((all_indexes, words, word_norm,
                          wordpieces_words, wordpieces_raw_aligned_with_words, wordpieces_inputs_raw_tokens,
                          ind_wordpieces_words_alignement_index, ind_wordpieces_raw_aligned_alignement_index, ind_wordpieces_inputs_raw_tokens_alignement_index,
                          is_mwe_label, n_masks_to_app_in_raw_label,
                          chars, chars_norm, word_norm_not_norm, edit, pos, xpos, heads, types, masks, single, lengths, order_inputs, raw_word_inputs, words_normalized_str, raw_lines))
  #pdb.set_trace()
  for i in range(len(data_variable)):
    if data_variable[i][0] != 1:
      print("SIZE", i, bucket_size,_buckets,bucket_id, data_variable[i][3].size())


  return data_variable, bucket_sizes, _buckets, max_char_length_dic["n_sent"]


def get_batch_variable(data, batch_size, unk_replace=0., lattice=None,
                       normalization=False,):
  """
  Given read_data_to_variable() get a random batch
  """
  data_variable, bucket_sizes, _buckets, _ = data
  total_size = float(sum(bucket_sizes))
  # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
  # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
  # the size if i-th training bucket, as used later.
  buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]
  # Choose a bucket according to data distribution. We pick a random number
  # in [0, 1] and use the corresponding interval in train_buckets_scale.
  random_number = np.random.random_sample()
  bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
  bucket_length = _buckets[bucket_id]

  all_indexes, words, word_norm, wordpieces_words, wordpieces_raw_aligned_with_words, wordpieces_inputs_raw_tokens, \
  ind_wordpieces_words_alignement_index, ind_wordpieces_raw_aligned_alignement_index, ind_wordpieces_inputs_raw_tokens_alignement_index,\
  is_mwe_label, n_masks_to_app_in_raw_label,\
  chars, chars_norm, word_norm_not_norm, edit, pos, xpos, heads, types, masks, single, lengths, order_inputs, raw, normalized_str, raw_lines = data_variable[bucket_id]
  bucket_size = bucket_sizes[bucket_id]
  batch_size = min(bucket_size, batch_size)
  index = torch.randperm(bucket_size).long()[:batch_size]

  if words.is_cuda:
    index = index.cuda()
  words = words[index]

  if wordpieces_words is not None:
    wordpieces_words = wordpieces_words[index]
    ind_wordpieces_words_alignement_index = ind_wordpieces_words_alignement_index[index]
  if wordpieces_raw_aligned_with_words is not None:
    wordpieces_raw_aligned_with_words = wordpieces_raw_aligned_with_words[index]
    ind_wordpieces_raw_aligned_alignement_index = ind_wordpieces_raw_aligned_alignement_index[index]
  if wordpieces_inputs_raw_tokens is not None:
    wordpieces_inputs_raw_tokens = wordpieces_inputs_raw_tokens[index]
    ind_wordpieces_inputs_raw_tokens_alignement_index = ind_wordpieces_inputs_raw_tokens_alignement_index[index]
  if is_mwe_label is not None:
    is_mwe_label = is_mwe_label[index]
  if n_masks_to_app_in_raw_label is not None:
    n_masks_to_app_in_raw_label = n_masks_to_app_in_raw_label[index]
  if all_indexes is not None:
    all_indexes = all_indexes[index]
  # discarding singleton
  if unk_replace:
    ones = Variable(single.data.new(batch_size, bucket_length).fill_(1))
    noise = Variable(masks.data.new(batch_size, bucket_length).bernoulli_(unk_replace).long())
    words = words * (ones - single[index] * noise)
  if normalization:
    chars_norm = chars_norm[index]
    if word_norm is not None:
      word_norm = word_norm[index]
    if word_norm_not_norm is not None:
      word_norm_not_norm = word_norm_not_norm[index]
    if edit is not None:
      edit = edit[index]
  raw = [raw[i.cpu().item()] for i in index]
  normalized_str = [normalized_str[i.cpu().item()] for i in index]

  return all_indexes, words, word_norm, wordpieces_words, wordpieces_raw_aligned_with_words, wordpieces_inputs_raw_tokens, \
         ind_wordpieces_words_alignement_index, ind_wordpieces_raw_aligned_alignement_index, ind_wordpieces_inputs_raw_tokens_alignement_index, \
         is_mwe_label, n_masks_to_app_in_raw_label, \
         chars[index], chars_norm, word_norm_not_norm, edit, pos[index], xpos[index], heads[index], types[index],\
         masks[index], lengths[index], order_inputs[index.cpu()], raw, normalized_str, raw_lines


def iterate_batch_variable(data, batch_size, unk_replace=0.,
                           word_decoding=False,
                           lattice=None, normalization=False, verbose=1):
  """
  Iterate over the dataset based on read_data_to_variable() object (used a evaluation)
  """

  data_variable, bucket_sizes, _buckets, _ = data
  bucket_indices = np.arange(len(_buckets))

  for bucket_id in bucket_indices:
    bucket_size = bucket_sizes[bucket_id]
    bucket_length = _buckets[bucket_id]
    if bucket_size == 0:
      continue
    all_indexes, words, word_norm, wordpieces_words, wordpieces_raw_aligned_with_words, wordpieces_inputs_raw_tokens, \
    ind_wordpieces_words_alignement_index, ind_wordpieces_raw_aligned_alignement_index, ind_wordpieces_inputs_raw_tokens_alignement_index, \
    is_mwe_label, n_masks_to_app_in_raw_label, \
      chars, chars_norm, word_norm_not_norm, edit, pos, xpos, heads, types, masks, single, lengths, order_ids, \
      raw_word_inputs, normalized_str, raw_lines = data_variable[bucket_id]

    if unk_replace:
      ones = Variable(single.data.new(bucket_size, bucket_length).fill_(1))
      noise = Variable(masks.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long())
      words = words * (ones - single * noise)
    _word_norm = None
    _edit = None
    for start_idx in range(0, bucket_size, batch_size):
      excerpt = slice(start_idx, start_idx + batch_size)
      if normalization:
        chars_norm_ = chars_norm[excerpt]
        if word_norm is not None:
          _word_norm = word_norm[excerpt]
        if edit is not None:
          _edit = edit[excerpt]
        if word_norm_not_norm is not None:
          _word_norm_not_norm = word_norm_not_norm[excerpt]
        else:
            _word_norm_not_norm = None
      else:
        chars_norm_ = None
        #TODO : should make _word_norm_not_norm  and char norm independant !!
        _word_norm_not_norm = None
      if chars[excerpt].size(0) <= 1:
        printing("WARNING : batch_size 1 {} char ".format(chars[excerpt].size()),
                 verbose=verbose, verbose_level=2)
        #continue
      if normalization:
        if chars_norm_.size(0) <= 1:
          printing("WARNING : batch_size 1  {} for char_nor  ".format(chars_norm_.size()),
                   verbose=verbose, verbose_level=2)
          #continue

      _wordpieces_words = wordpieces_words[excerpt] if wordpieces_words is not None else None
      _ind_wordpieces_words_alignement_index = ind_wordpieces_words_alignement_index[excerpt] if ind_wordpieces_words_alignement_index is not None else None
      _all_indexes = all_indexes[excerpt] if all_indexes is not None else None
      _wordpieces_raw_aligned_with_words = wordpieces_raw_aligned_with_words[excerpt] if wordpieces_raw_aligned_with_words is not None else None
      _ind_wordpieces_raw_aligned_alignement_index = ind_wordpieces_raw_aligned_alignement_index[excerpt] if ind_wordpieces_raw_aligned_alignement_index is not None else None

      _wordpieces_inputs_raw_tokens = wordpieces_inputs_raw_tokens[excerpt] if wordpieces_inputs_raw_tokens is not None else wordpieces_inputs_raw_tokens
      _ind_wordpieces_inputs_raw_tokens_alignement_index = ind_wordpieces_inputs_raw_tokens_alignement_index[excerpt] if ind_wordpieces_inputs_raw_tokens_alignement_index is not None else None
      _is_mwe_label = is_mwe_label[excerpt] if is_mwe_label is not None else None
      _n_masks_to_app_in_raw_label = n_masks_to_app_in_raw_label[excerpt] if n_masks_to_app_in_raw_label is not None else None

      if word_norm is not None:
        if word_norm.size(0) <= 0:
          printing("WARNING : We are skipping a batch because word_norm {} {}".format(word_norm.size(), word_norm),
                   verbose=verbose, verbose_level=2)
          continue

      yield _all_indexes, words[excerpt], _word_norm, _wordpieces_words, _wordpieces_raw_aligned_with_words, _wordpieces_inputs_raw_tokens, \
            _ind_wordpieces_words_alignement_index, _ind_wordpieces_raw_aligned_alignement_index, _ind_wordpieces_inputs_raw_tokens_alignement_index, \
            _is_mwe_label, _n_masks_to_app_in_raw_label, \
            chars[excerpt], chars_norm_, _word_norm_not_norm, _edit, \
            pos[excerpt], xpos[excerpt], heads[excerpt], \
            types[excerpt],  \
            masks[excerpt], lengths[excerpt], order_ids[excerpt], \
            raw_word_inputs[excerpt], normalized_str[excerpt], raw_lines[excerpt]
