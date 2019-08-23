from env.importing import *
from .constants import UNK_ID, DIGIT_RE


def construct_word_embedding_table(word_dim, word_dictionary, word_embed, random_init=False):
  scale = np.sqrt(3.0 / word_dim)
  table = np.empty([word_dictionary.size(), word_dim], dtype=np.float32)
  table[UNK_ID, :] = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
  oov = 0
  for word, index in word_dictionary.items():
    if word in word_embed and not random_init:
      embedding = word_embed[word]
    elif word.lower() in word_embed and not random_init:
      embedding = word_embed[word.lower()]
    else:
      embedding = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
      oov += 1
    table[index, :] = embedding
  print('word OOV: %d/%d' % (oov, word_dictionary.size()))
  return torch.from_numpy(table)


def load_word_embeddings(path, dry_run, content_arr, useful_words=None):
  if not useful_words:
    useful_words = getWordsToBeLoaded(content_arr)
  embed_dim = -1
  embed_dict = dict()
  pbar = None
  with codecs.open(path, 'r', 'utf-8', errors='ignore') as file:
    li = 0
    for line in file:
      line = line.strip()
      if len(line) == 0:
        continue
      tokens = line.split()
      if len(tokens) < 3:
        pbar=tqdm(total=int(tokens[0]) if not dry_run else 100)
        embed_dim = int(tokens[1])
        continue
      ## --
      if len(tokens)-1==embed_dim:
        word = DIGIT_RE.sub(b"0", str.encode(tokens[0])).decode()
        if word in useful_words:
          embed = np.empty([1, embed_dim], dtype=np.float32)
          embed[:] = tokens[1:]
          embed_dict[word] = embed
      ## --- 
      li = li + 1
      if li%5==0:
        pbar.update(5)
      if dry_run and li==100:
        break
  pbar.close()
  return embed_dict, embed_dim


def getWordsToBeLoaded(content_arr):
  words = {}
  for file in content_arr:
    if os.path.exists(file):
      with codecs.open(file, 'r', 'utf-8', errors='ignore') as f:
        for line in f:
          line = line.strip()
          if len(line) == 0 or line[0]=='#':
            continue
          tokens = line.split('\t')
          if '-' in tokens[0] or '.' in tokens[0]:
            continue
          word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
          if word not in words:
            words[word] = True
            words[word.lower()] = True
  return words



def getOOVWords(word_dictionary, test_path):
  oov_words = {}
  with codecs.open(test_path, 'r', 'utf-8', errors='ignore') as f:
    for line in f:
      line = line.strip()
      if len(line) == 0 or line[0]=='#':
        continue
      tokens = line.split('\t')
      if '-' in tokens[0] or '.' in tokens[0]:
        continue
      word = DIGIT_RE.sub(b"0", str.encode(tokens[1])).decode()
      if word not in oov_words:
        for cand_word in [word, word.lower()]:
          if word_dictionary.get_index(cand_word)==0:
            oov_words[cand_word] = True
  return oov_words


class SentenceWordPieced(object):
  def __init__(self, word_piece_raw_tokens_aligned=None, word_piece_raw_tokens=None,
               word_piece_normalization=None,
               word_piece_words=None, word_piece_lemmas=None,
               is_mwe=None,
               word_piece_raw_tokens_aligned_index=None, word_piece_words_index=None, word_piece_raw_tokens_index=None,
               n_masks_to_add_in_raw_label=None,
               is_first_bpe_of_token=None, is_first_bpe_of_norm=None, is_first_bpe_of_words=None):
    # bpe indexes
    self.word_piece_raw_tokens_aligned = word_piece_raw_tokens_aligned if len(word_piece_raw_tokens_aligned) > 0 else None
    self.word_piece_raw_tokens = word_piece_raw_tokens if len(word_piece_raw_tokens) > 0 else None
    self.word_piece_words = word_piece_words if len(word_piece_words) > 0 else None
    self.word_piece_lemmas = word_piece_lemmas if len(word_piece_lemmas) > 0 else None
    self.word_piece_normalization = word_piece_normalization if len(word_piece_normalization) else None
    # is MWE per first bpe tokens
    self.is_mwe = is_mwe if len(is_mwe) > 0 else None
    self.n_masks_to_add_in_raw_label = n_masks_to_add_in_raw_label if len(n_masks_to_add_in_raw_label) > 0 else None
    # first token of each sequence indicator
    self.is_first_bpe_of_token = is_first_bpe_of_token
    self.is_first_bpe_of_norm = is_first_bpe_of_norm
    self.is_first_bpe_of_words = is_first_bpe_of_words
    # alignement with UD token
    self.word_piece_raw_tokens_aligned_index = word_piece_raw_tokens_aligned_index
    self.word_piece_raw_tokens_index = word_piece_raw_tokens_index
    self.word_piece_words_index = word_piece_words_index

  def sanity_check_len(self, normalization, n_words):

    assert len(self.is_first_bpe_of_token) == len(self.word_piece_raw_tokens), "ERROR : {} not same len as {}".format(self.is_first_bpe_of_token, self.word_piece_raw_tokens)

    if normalization:
      assert len(self.is_first_bpe_of_norm) == len(self.word_piece_normalization)

    assert len(self.is_first_bpe_of_words) == len(self.word_piece_words), "ERROR : {} not same len as {}".format(self.is_first_bpe_of_words, self.word_piece_words)
    ## as many words (syntactic word) as lemmas
    #assert len(self.word_piece_lemmas) == len(self.word_piece_w), "ERROR : {} not same len as {}".format(self.word_piece_lemmas, self.word_piece_words)
    # as many words as aligned tokens (MASK were added)
    assert len(self.word_piece_raw_tokens_aligned) == len(self.word_piece_words), "ERROR : {} not same len as {}".format(len(self.word_piece_raw_tokens_aligned), len(self.word_piece_words))
    # as many raw tokens as is_mwe
    assert len(self.word_piece_raw_tokens) == len(self.is_mwe), "ERROR : {} not same len as {}".format(self.word_piece_raw_tokens, self.is_mwe)
    # check alignement index with 1-hot encoded index
    assert len(self.word_piece_raw_tokens_aligned_index) == len(self.word_piece_raw_tokens_aligned), "ERROR : {} not same len as {}".format(self.word_piece_raw_tokens_aligned_index, self.word_piece_raw_tokens_aligned)
    assert len(self.word_piece_raw_tokens_index) == len(self.word_piece_raw_tokens)
    assert len(self.word_piece_words) == len(self.word_piece_words_index)

    assert eval(n_words) == len([first_word for first_word in self.is_first_bpe_of_words if first_word == 1]), \
      "ERROR n_words {} while {} first bpe of words".format(n_words, len([first_word for first_word in self.is_first_bpe_of_words if first_word == 1]))

    # could do more checks on n_raw tokens based on data


class Sentence(object):
  def __init__(self, words, word_ids,
               char_seqs, char_id_seqs, lines,
               word_norm=None, word_norm_ids=None,
               char_norm_seq=None, char_norm_ids_seq=None,
               ):
    self.words = words
    self.word_ids = word_ids
    self.char_seqs = char_seqs
    self.char_id_seqs = char_id_seqs
    self.raw_lines = lines
    self.char_norm_seq = char_norm_seq
    self.char_norm_ids_seq = char_norm_ids_seq
    self.word_norm = word_norm
    self.word_norm_ids = word_norm_ids

  def length(self):
    return len(self.words)


class DependencyInstance(object):
  def __init__(self, sentence, postags, pos_ids, xpostags, xpos_ids, lemmas, lemma_ids, heads, types, type_ids,
               sentence_word_piece=None):
    self.sentence = sentence
    self.sentence_word_piece = sentence_word_piece
    self.postags = postags
    self.pos_ids = pos_ids
    self.xpostags = xpostags
    self.xpos_ids = xpos_ids
    self.heads = heads
    self.types = types
    self.type_ids = type_ids
    self.lemmas = lemmas
    self.lemma_ids = lemma_ids

  def length(self):
    return self.sentence.length()


  