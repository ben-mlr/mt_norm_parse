MAX_CHAR_LENGTH = 20


NUM_CHAR_PAD = 4


PAD = u"_PAD"
PAD_POS = u"_PAD_POS"
PAD_TYPE = u"_<PAD>"
PAD_CHAR = u"_PAD_CHAR"
ROOT = u"_ROOT"
ROOT_POS = u"_ROOT_POS"
ROOT_TYPE = u"_<ROOT>"
ROOT_CHAR = u"_ROOT_CHAR"
END = u"_END"
END_POS = u"_END_POS"
END_TYPE = u"_<END>"
END_CHAR = u"_END_CHAR"
_START_VOCAB = [PAD, ROOT, END]
CHAR_START = u"_START"

UNK_ID = 0
# we add this for normalization (no change if add_char_start==0 in create_dict + read_data
CHAR_START_ID = 2
CHAR_END_ID = 4 # related to END_CHAR
# END is both signal of sentence end for word vec and char vec
PAD_ID_CHAR = 1

PAD_ID_NORM_NOT_NORM = 2
PAD_ID_WORD = 1
PAD_ID_TAG = 1
PAD_ID_MORPH = 1
ROOT_ID_MORPH = 2

MEAN_RAND_W2V = 0
SCALE_RAND_W2V = 0.2


import re
DIGIT_RE = re.compile(br"\d")

NUM_SYMBOLIC_TAGS = 3