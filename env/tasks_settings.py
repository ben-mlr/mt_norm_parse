from env.importing import CrossEntropyLoss
#from model.bert_tools_from_core_code.modeling import BertGraphHead


# CONVENTIONS :
#  for single task prediction (tagging) : the task itself batch.label , label name, logits, ... have same name
#  for double task prediction (parsing) : the labels  logits, follow the template : task_label_name_1 :
#    eg parsing_heads, parsing_types

TASKS_PARAMETER = {

                    "append_masks": {"normalization": True,
                                     "default_metric": "exact_match",
                                     "head": None, "loss": CrossEntropyLoss(),
                                     "prediction_level": "bpe",
                                     "label": ["append_masks"],
                                     "eval_metrics": [["accuracy-append_masks"]],
                                     # "predicted_classes": ["NORMED", "NEED_NORM"],
                                     # "predicted_classes_pred_field": ["PRED_NORMED", "PRED_NEED_NORM"]
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


                    "normalize": {"normalization": True, "default_metric": "exact_match",
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
                                     "alignement": "wordpieces_inputs_raw_tokens_alignement",
                                     "label": ["mwe_detection"],
                                     "eval_metrics": [["accuracy-is_mwe-is_mwe"]],
                                     "subsample-allowed": ["all", "InV", "OOV", "MWE"],
                                     },
                   "n_masks_mwe":
                                    {"normalization": False,
                                     "head": "BertTokenHead",
                                     "loss": CrossEntropyLoss(ignore_index=-1, reduce="mean"),
                                     "num_labels_mandatory":True,
                                     "prediction_level": "word",
                                     "input": "wordpieces_inputs_raw_tokens",
                                     "alignement": "wordpieces_inputs_raw_tokens_alignement",
                                     "label": ["n_masks_mwe"],
                                     "eval_metrics": [["accuracy-n_masks_mwe-n_masks_mwe"]],
                                    "subsample-allowed": ["all", "InV", "OOV", "MWE"],
                                     },
                   "mwe_prediction":
                                    {"normalization": False,
                                     "head": "BertOnlyMLMHead",
                                     "num_labels_mandatory":False,
                                     "loss": CrossEntropyLoss(ignore_index=-1, reduce="mean"),
                                     "prediction_level": "bpe",
                                     "subsample-allowed": ["all", "InV", "OOV", "MWE"],
                                     "input": "wordpieces_raw_aligned_with_words",
                                     "alignement": "wordpieces_raw_aligned_with_words_alignement",

                                     "label": ["mwe_prediction"],
                                     "eval_metrics": [["accuracy-mwe_pred-mwe_pred"]],
                                     },


                   "pos": {"normalization": False,
                           "default_metric": "accuracy-pos",
                           "pred": ["pos_pred"],
                            "num_labels_mandatory":True,
                           # a list per prediction
                           "eval_metrics": [["accuracy-pos-pos"]],
                           "subsample-allowed": ["all", "NEED_NORM", "NORMED", "PRED_NEED_NORM", "PRED_NORMED", "InV", "OOV"],
                           "label": ["pos"],
                           # because its the label of mwe prediction
                           "input": "mwe_prediction",
                           "alignement": "mwe_prediction_alignement",
                           "head": "BertTokenHead",
                           "prediction_level": "word",
                           "loss": CrossEntropyLoss(ignore_index=-1,reduce="sum")
                           },
                   "mlm": {"normalization": False,
                           "mask_input": True,# means the sequence input is always masked following mlm (train and test!)
                           "default_metric": "accuracy-mlm",
                           "default-subsample": "mlm",
                           "subsample-allowed": ["all", "InV", "OOV", "mlm"],
                           "num_labels_mandatory": False,
                           # a list per prediction
                           "eval_metrics": [["accuracy-mlm-mwe_prediction"]],
                           "label": ["mwe_prediction"],
                           # because its the label of mwe prediction
                           "input": "input_masked",
                           "alignement": "mwe_prediction_alignement",
                           "original": "mwe_prediction",
                           "head": "BertOnlyMLMHead",
                           "prediction_level": "bpe",
                           "loss": CrossEntropyLoss(ignore_index=-1, reduce="sum")
                           },
                   "parsing": {
                       "normalization": False,
                       "default_metric": None,
                       "num_labels_mandatory": True,
                       "num_labels_mandatory_to_check": ["types"],
                       "eval_metrics": [["accuracy-parsing-heads"], ["accuracy-parsing-types"]],
                       "head": "BertGraphHead",
                        "subsample-allowed":  ["all", "InV", "OOV"],
                        # because its the label of mwe prediction
                       "input": "mwe_prediction",
                       "alignement": "mwe_prediction_alignement",
                       "label": ["heads", "types"],
                       "prediction_level": "word",
                       "loss": CrossEntropyLoss(ignore_index=-1, reduction="mean")
                   },

                   "all": {"normalization": True,
                           "head": None,
                           "loss": CrossEntropyLoss()}}
