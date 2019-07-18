from env.importing import *
from env.project_variables import *

assert os.path.isdir(BERT_MODELS_DIRECTORY), \
    "ERROR : {} does not exist : it should host the bert models tar.gz and vocabulary ".format(BERT_MODELS_DIRECTORY)

BERT_MODEL_DIC = {"bert-cased-multitask": {"vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-cased-vocab.txt"),
                                           "model": os.path.join(BERT_MODELS_DIRECTORY, "bert-cased-multitask.tar.gz"),
                                           "vocab_size": 28996
                                           },
                  "bert-cased": {"vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-cased-vocab.txt"),
                                 "model": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-cased.tar.gz"),
                                 "vocab_size": 28996,
                                },
                  "random":  {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-cased-vocab.txt"),
                      "model": None,
                      "vocab_size": 28996,
                              },
                  "bert_base_multilingual_cased": {
                      "vocab": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased-vocab.txt"),
                      "model": os.path.join(BERT_MODELS_DIRECTORY, "bert-base-multilingual-cased.tar.gz"),
                      "vocab_size": 119547
                  }
                  }


