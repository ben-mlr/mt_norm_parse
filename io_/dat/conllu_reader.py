from .ioutils import DependencyInstance, Sentence
from .constants import DIGIT_RE, MAX_CHAR_LENGTH, NUM_CHAR_PAD, ROOT, ROOT_CHAR, ROOT_POS, ROOT_TYPE, PAD, END_CHAR, END_POS, END_TYPE, END
from io_.info_print import printing
from io_.dat.conllu_get_normalization import get_normalized_token
from env.project_variables import AVAILABLE_TASKS, TASKS_PARAMETER
from env.importing import *
from env.project_variables import PROJECT_PATH


class CoNLLReader(object):

  def __init__(self, file_path, word_dictionary,
               char_dictionary, pos_dictionary, type_dictionary, xpos_dictionary,

               lemma_dictionary, word_norm_dictionary=None,
               case=None,
               max_char_len=MAX_CHAR_LENGTH):
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
      heads.append(0)

    for tokens in lines:

      if '-' in tokens[0] or '.' in tokens[0]:
        continue
      if len(tokens)<10:
        sys.stderr.write("Sentence broken for unkwown reasons {} \n {} ".format(tokens, lines))
        if os.environ.get("EXPERIENCE") is not None:
          print("WARNING : WRITING corrupted gold data in {} ".format(os.path.join(os.environ["EXPERIENCE"], "logs/catching_errors.txt")))
          open(os.path.join(os.environ["EXPERIENCE"], "logs/catching_errors.txt"), "a").write("Line broken {} because of tokens {} from {} file \n ".format(lines, tokens,self.__file_path))
        continue

      n_exception = 0
      if normalization:
        # includes sequence level and word level
        normalized_token, n_exception = get_normalized_token(norm_field=tokens[9], n_exception=n_exception,
                                                             predict_mode_only=not must_get_norm,
                                                             verbose=verbose)
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
                 "encoded as {} ".format(normalized_token, normalized_token_id, char_norm_str_seq , char_norm_id_seqs),
                                         verbose_level=6, verbose=verbose)
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
      heads.append(0)
    return DependencyInstance(Sentence(words, word_ids, char_seqs,
                                       char_id_seqs, [lines, raw_text],
                                       word_norm=norm_words,
                                       word_norm_ids=norm_word_ids,
                                       char_norm_ids_seq=char_norm_id_seqs,
                                       char_norm_seq=char_norm_str_seq),
                              postags, pos_ids, xpostags, xpos_ids, lemmas, lemma_ids, heads, types, type_ids)

# TODO : add end begin symbol both for character sequence and normalized character sequence



# hid is as ['0', '3', '1'] ? why '' ??