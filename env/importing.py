

import os
from uuid import uuid4
import argparse
from sys import platform
import pdb as pdb
import git
import sys
from tqdm import tqdm



import torch
import numpy as np
import time
from collections import OrderedDict, Iterable
from tensorboardX import SummaryWriter

from torch.autograd import Variable
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import json
import re

# measuring erros
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from scipy.stats import hmean
from nltk import edit_distance

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence


from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from toolbox.deep_learning_toolbox import get_cumulated_list
from torch.nn.utils.rnn import pack_sequence

# generation
import random

import gspread
from oauth2client.service_account import ServiceAccountCredentials
