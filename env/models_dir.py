from env.importing import *
from env.project_variables import *


BERT_MODEL_DIC = {"bert-cased": {"vocab": os.path.join(LM_PROJECT, "bert_models/bert-base-cased-vocab.txt"),
                                 "model": os.path.join(LM_PROJECT, "bert_models/bert-base-cased.tar.gz"),
                                 "vocab_size": 28996,
                                },
                  "random":  {
                      "vocab": os.path.join(LM_PROJECT, "bert_models/bert-base-cased-vocab.txt"),
                      "model": None,
                      "vocab_size": 28996,
                              },
                  }


