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
                                 "eval_metrics": [["accuracy-exact-normalize", "accuracy-normalize", "npv-normalize", "recall-normalize", "precision-normalize","tnr-normalize", "f1-normalize", "accuracy-exact-n_masks_pred","accuracy-exact-normalize_pred"]],
                                 "predicted_classes": ["NORMED", "NEED_NORM"],
                                 "predicted_classes_pred_field": ["PRED_NORMED", "PRED_NEED_NORM"]
                                 },
                   # (not a real task but needed)
                   "normalize_pred": {"default_metric": "exact_match",
                                      "predicted_classes": ["NORMED", "NEED_NORM"],
                                      "predicted_classes_pred_field": ["PRED_NORMED", "PRED_NEED_NORM"]
                                 },
                   # TOOD : could add input for full flexibility
                   "mwe_detection": {"normalization": False,
                                     "head": "BertTokenHead",
                                     "loss": CrossEntropyLoss(ignore_index=-1, reduce="mean"),
                                     "prediction_level": "word",
                                     "num_labels_mandatory":True,
                                     "input": "wordpieces_inputs_raw_tokens",
                                     "label": ["mwe_detection"],
                                     "eval_metrics": [["accuracy-exact-is_mwe"]],
                                     },
                   "n_masks_mwe":
                                    {"normalization": False,
                                     "head": "BertTokenHead",
                                     "loss": CrossEntropyLoss(ignore_index=-1, reduce="mean"),
                                     "num_labels_mandatory":True,
                                     "prediction_level": "word",
                                     "input": "wordpieces_inputs_raw_tokens",
                                     "label": ["n_masks_mwe"],
                                     "eval_metrics": [["accuracy-exact-n_masks_mwe"]],
                                     },
                   "mwe_prediction":
                                    {"normalization": False,
                                     "head": "BertOnlyMLMHead",
                                    "num_labels_mandatory":False,
                                     "loss": CrossEntropyLoss(ignore_index=-1, reduce="mean"),
                                     "prediction_level": "bpe",
                                     "input": "wordpieces_raw_aligned_with_words",
                                     "label": ["mwe_prediction"],
                                     "eval_metrics": [["accuracy-exact-mwe_pred"]],
                                     },

                   "append_masks": {"normalization": True,
                                    "default_metric": "exact_match",
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
                            "num_labels_mandatory":True,
                           # a list per prediction
                           "eval_metrics": [["accuracy-exact-pos"]],
                           "label": ["pos"],
                           # because its the label of mwe prediction
                           "input": "mwe_prediction",
                           "head": "BertTokenHead",
                           "prediction_level": "word",
                           "loss": CrossEntropyLoss(ignore_index=-1,reduce="sum")
                           },
                   "mlm": {"normalization": False,
                           "default_metric": "accuracy-mlm",
                           "num_labels_mandatory": False,
                           # a list per prediction
                           "eval_metrics": [["accuracy-exact-mlm"]],
                           "label": ["mwe_prediction"],
                           # because its the label of mwe prediction
                           "input": "mwe_prediction",
                           "head": "BertOnlyMLMHead",
                           "prediction_level": "bpe",
                           "loss": CrossEntropyLoss(ignore_index=-1, reduce="sum")
                           },
                   "parsing": {
                       "normalization": False,
                       "default_metric": None,
                       "num_labels_mandatory": True,
                       "num_labels_mandatory_to_check": ["parsing_types"],
                       "eval_metrics": [["accuracy-exact-parsing_heads"], ["accuracy-exact-parsing_types"]],
                       "head": "BertGraphHead",
                        # because its the label of mwe prediction
                        "input": "mwe_prediction",
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
