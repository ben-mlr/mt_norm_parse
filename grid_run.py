from training.train import train
from io_.info_print import printing
import os
from evaluate.evaluate_epoch import evaluate
from env.project_variables import PROJECT_PATH, TRAINING, DEV, TEST, CHECKPOINT_DIR, DEMO, DEMO2, REPO_DATASET
#DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def train_eval(train_path, test_path, model_id_pref,n_epochs=11,
               warmup=False, args={},use_gpu=None,debug=False,
               print_raw=False,
               verbose=0):
    hidden_size_encoder = args.get("hidden_size_encoder", 10)
    output_dim = args.get("output_dim",10)
    char_embedding_dim = args.get("char_embedding_dim",10)
    hidden_size_sent_encoder = args.get("hidden_size_sent_encoder", 10)
    hidden_size_decoder = args.get("hidden_size_decoder", 10)
    batch_size = args.get("batch_size", 2)

    n_epochs = 1 if warmup else n_epochs

    if warmup:
        printing("Warm up : running 1 epoch ", verbose=verbose, verbose_level=0)

    model_full_name = train(train_path, test_path, n_epochs=n_epochs, normalization=True,
                            batch_size=batch_size, model_specific_dictionary=True,
                            dict_path=None, model_dir=None, add_start_char=1,
                            add_end_char=1, use_gpu=use_gpu,
                            label_train=REPO_DATASET[train_path], label_dev=REPO_DATASET[test_path],
                            freq_checkpointing=10, reload=False, model_id_pref=model_id_pref,
                            hidden_size_encoder=hidden_size_encoder, output_dim=output_dim, char_embedding_dim=char_embedding_dim,
                            hidden_size_sent_encoder=hidden_size_sent_encoder, hidden_size_decoder=hidden_size_decoder,
                            print_raw=print_raw, debug=debug,
                            checkpointing=True)

    model_dir = os.path.join(CHECKPOINT_DIR, model_full_name+"-folder")
    dict_path = os.path.join(CHECKPOINT_DIR, model_full_name+"-folder", "dictionaries")

    for eval_data, eval_label in zip([train_path, test_path], ["owputi_train", "lexnorm_test"]):
            evaluate(model_full_name=model_full_name, data_path=eval_data,
                     dict_path=dict_path,use_gpu=use_gpu,
                     label_report=eval_label, debug=debug,
                     normalization=True,print_raw=print_raw,
                     model_specific_dictionary=True,
                     batch_size=batch_size,
                     dir_report=model_dir, verbose=verbose)


if __name__ == "__main__":

      train_path = DEMO
      test_path = DEMO
      params = []
      model_id_pref_list = []
      all = False
      if all:
          model_id_pref_list.extend( ["comparison_ablation-big","comparison_ablation-big","comparison_ablation-big"])
          params.append({"hidden_size_encoder": 250, "output_dim": 300, "char_embedding_dim": 300,
                         "hidden_size_sent_encoder": 250, "hidden_size_decoder": 300, "batch_size": 50})
          params.append({"hidden_size_encoder": 250, "output_dim": 300, "char_embedding_dim": 300,
                         "hidden_size_sent_encoder": 250, "hidden_size_decoder": 300, "batch_size": 20})
          params.append({"hidden_size_encoder": 250, "output_dim": 300, "char_embedding_dim": 300,
                         "hidden_size_sent_encoder": 250, "hidden_size_decoder": 300, "batch_size": 2})
          # new pparam
      model_id_pref_list.append("comparison_ablation-medium")
      model_id_pref_list.append("comparison_ablation-medium")
      model_id_pref_list.append("comparison_ablation-medium")

      params.append({"hidden_size_encoder": 51, "output_dim": 50, "char_embedding_dim": 20,
                     "hidden_size_sent_encoder": 50, "hidden_size_decoder": 50, "batch_size": 50})
      params.append({"hidden_size_encoder": 51, "output_dim": 50, "char_embedding_dim": 20,
                     "hidden_size_sent_encoder": 50, "hidden_size_decoder": 50, "batch_size": 20})
      params.append({"hidden_size_encoder": 51, "output_dim": 50, "char_embedding_dim": 20,
                     "hidden_size_sent_encoder": 50, "hidden_size_decoder": 50, "batch_size": 2})
      #
      model_id_pref_list.append("comparison_ablation-small")
      model_id_pref_list.append("comparison_ablation-small")
      model_id_pref_list.append("comparison_ablation-small")
      params.append({"hidden_size_encoder": 20, "output_dim": 20, "char_embedding_dim": 10,
                     "hidden_size_sent_encoder": 20, "hidden_size_decoder": 20, "batch_size": 2})
      params.append({"hidden_size_encoder": 20, "output_dim": 20, "char_embedding_dim": 10,
                     "hidden_size_sent_encoder": 20, "hidden_size_decoder": 20, "batch_size": 2})
      params.append({"hidden_size_encoder": 20, "output_dim": 20, "char_embedding_dim": 10,
                     "hidden_size_sent_encoder": 20, "hidden_size_decoder": 20, "batch_size": 2})
      #
      model_id_pref_list.append("comparison_ablation-various")
      model_id_pref_list.append("comparison_ablation-various")
      model_id_pref_list.append("comparison_ablation-various")
      params.append({"hidden_size_encoder": 50, "output_dim": 20, "char_embedding_dim": 20,
                     "hidden_size_sent_encoder": 50, "hidden_size_decoder": 100, "batch_size": 2})
      params.append({"hidden_size_encoder": 50, "output_dim": 20, "char_embedding_dim": 20,
                     "hidden_size_sent_encoder": 50, "hidden_size_decoder": 100, "batch_size": 20})
      params.append({"hidden_size_encoder": 50, "output_dim": 20, "char_embedding_dim": 20,
                     "hidden_size_sent_encoder": 50, "hidden_size_decoder": 100, "batch_size": 50})

      for param, model_id_pref in zip(params,model_id_pref_list):
          param["batch_size"] = 2
          model_id_pref = "TEST2"
          print("STARTING MODEL {} with param {} ".format(model_id_pref, param))
          train_eval(train_path, test_path, model_id_pref, warmup=True, args=param,
                     print_raw=False,debug=False,
                     use_gpu=None, verbose=0)
          print("DONE MODEL {} with param {} ".format(model_id_pref, param))
          break


