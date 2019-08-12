from env.importing import CrossEntropyLoss
#from model.bert_tools_from_core_code.modeling import BertGraphHead


# CONVENTIONS :
#  for single task prediction (tagging) : the task itself batch.label , label name, logits, ... have same name
#  for double task prediction (parsing) : the labels  logits, follow the template : task_label_name_1 :
#    eg parsing_heads, parsing_types

TASKS_PARAMETER = {"normalize": {"normalization": True, "default_metric": "exact_match",
                                 "head": None, "loss": CrossEntropyLoss(ignore_index=-1),
                                 "prediction_level": "bpe",
                                 "label": ["normalize"],
                                 "eval_metrics": [["accuracy-exact-normalize"]],
                                 "predicted_classes": ["NORMED", "NEED_NORM"],
                                 "predicted_classes_pred_field": ["PRED_NORMED", "PRED_NEED_NORM"]
                                 },
                    "append_masks": {"normalization": True, "default_metric": "exact_match",
                                 "head": None, "loss": CrossEntropyLoss(),
                                 "prediction_level": "bpe",
                                 "label": ["append_masks"],
                                 "eval_metrics": [["accuracy-exact-append_masks"]],
                                 #"predicted_classes": ["NORMED", "NEED_NORM"],
                                 #"predicted_classes_pred_field": ["PRED_NORMED", "PRED_NEED_NORM"]
                                 },
                   "norm_not_norm": {"normalization": True,
                                     "head": None,
                                     "prediction_level": "bpe",
                                     "loss": CrossEntropyLoss(),
                                     },
                   "edit_prediction": {"normalization": True,
                                       "head": None,
                                       "prediction_level": "word",
                                       "loss": CrossEntropyLoss()
                                       },
                   "pos": {"normalization": False,
                           "default_metric": "accuracy-pos",
                           "pred": ["pos_pred"],
                           # a list per prediction
                           "eval_metrics": [["accuracy-exact-pos"]],
                           "label": ["pos"],
                           "head": "BertTokenHead",
                           "prediction_level": "word",
                           "loss": CrossEntropyLoss(ignore_index=-1,reduce="sum")
                           },
                   "parsing": {
                       "normalization": False,
                       "default_metric": None,
                       "eval_metrics": [["LAS"], ["UAS"]],
                       "head": "BertGraphHead",
                       "label": ["parsing_heads", "parsing_types"],
                       "prediction_level": "word",
                       "loss": CrossEntropyLoss(ignore_index=-1, reduction="mean")
                   },
                   "parsing_attention": {
                       "head": "BertGraphHeadKyungTae",
                       "loss": CrossEntropyLoss(ignore_index=-1, reduction="sum")

                   },
                   "all": {"normalization": True,
                           "head": None,
                           "loss": CrossEntropyLoss()}}
