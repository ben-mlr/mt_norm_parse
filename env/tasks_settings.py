from env.importing import CrossEntropyLoss
from model.bert_tools_from_core_code.modeling import BertGraphHead


TASKS_PARAMETER = {"normalize": {"normalization": True, "default_metric": "exact_match",
                                 "head": None, "loss": CrossEntropyLoss(),
                                "prediction_level": "word",
                                 "predicted_classes": ["NORMED", "NEED_NORM"],
                                 "predicted_classes_pred_field": ["PRED_NORMED", "PRED_NEED_NORM"]},
                   "norm_not_norm": {"normalization": True,
                                     "head": None,
                                     "loss": CrossEntropyLoss(),
                                     "prediction_level": "word",
                                     },
                   "edit_prediction": {"normalization": True,
                                       "head": None,
                                       "prediction_level": "word",
                                       "loss": CrossEntropyLoss()},
                   "pos": {"normalization": False, "default_metric": "accuracy-pos",
                           "pred": ["pos_pred"],
                           "label": ["pos"],
                           "head": None,
                           "prediction_level": "word",
                           "loss": CrossEntropyLoss()},
                   "parsing": {
                       "normalization": False,
                       "default_metric": None,
                       "pred": None, #["arc_pred", "label_pred"],
                       "head": BertGraphHead,
                       "label": ["types", "heads"],
                       "prediction_level": "word",
                       "loss": CrossEntropyLoss()
                               },
                   "all": {"normalization": True,
                           "head": None,
                           "loss": CrossEntropyLoss()}}
