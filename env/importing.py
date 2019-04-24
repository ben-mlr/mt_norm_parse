
# IMPORTS

# python basics
import numpy as np
import os
from uuid import uuid4
import argparse
from sys import platform
import pdb as pdb
import git
import sys
from tqdm import tqdm
import io
import codecs
import time
import json
import re
import random
from collections import OrderedDict, Iterable

# visualization / report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorboardX import SummaryWriter

# measuring errors
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from scipy.stats import hmean
from nltk import edit_distance
# statistics
from scipy.stats import hmean


# torch related
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils.rnn import pack_sequence
import torch.nn.functional as F


# google sheet
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Bert
exception = ""
try:
	from pytorch_pretrained_bert import BertForTokenClassification, BertConfig
except Exception as e :
	print("IMPORT ERROR {}".format(e))
	exception += " | " +str(e)


# SEED INITIALIZATION
print("...initializing seeds")
sys.path.insert(0,".")
from env.project_variables import SEED_NP, SEED_TORCH
# SEED_TORCH used for any model related randomness + batch picking, dropouts, ..
# SEED_NP used for picking the bucket, for generating word embedding when loading embedding matrix and maybe other stuff
torch.manual_seed(SEED_TORCH)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED_NP)

print("ALL IMPORTS SUCESSFULLY LOADED with exception : {}  ".format(exception if exception!="" else "NONE"))