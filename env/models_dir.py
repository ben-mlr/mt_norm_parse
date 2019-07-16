from env.importing import *
from env.project_variables import *


BERT_MODEL_DIC = {"bert-cased-multitask": {"vocab": os.path.join(LM_PROJECT, "bert_models/bert-base-cased-vocab.txt"),
                                 "model": os.path.join(LM_PROJECT, "bert_models/bert-cased-multitask.tar.gz"),
                                 "vocab_size": 28996,
                                },

                 "bert-cased": {"vocab": os.path.join(LM_PROJECT, "bert_models/bert-base-cased-vocab.txt"),
                                 "model": os.path.join(LM_PROJECT, "bert_models/bert-base-cased.tar.gz"),
                                 "vocab_size": 28996,
                                },
                  "random":  {
                      "vocab": os.path.join(LM_PROJECT, "bert_models/bert-base-cased-vocab.txt"),
                      "model": None,
                      "vocab_size": 28996,
                              },
                  "bert_base_multilingual_cased": {
                      "vocab": os.path.join(LM_PROJECT, "bert_models/bert-base-multilingual-cased-vocab.txt"),
                      "model": os.path.join(LM_PROJECT, "bert_models/bert-base-multilingual-cased.tar.gz"),
                      "vocab_size": 119547
                  }

                  }


