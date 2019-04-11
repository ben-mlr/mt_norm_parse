from env.importing import *

sys.path.insert(0,"..")
sys.path.insert(0,"/scratch/bemuller/mt_norm_parse")
#TODO  why is it necessary fo rioc ?
from predict.prediction_batch import greedy_decode_batch
#from predict.prediction_string import decode_seq_str, decode_interacively
from model.seq2seq import LexNormalizer
from model.generator import Generator
from io_.data_iterator import data_gen_conllu
from io_.dat import conllu_data
from io_.info_print import printing
from env.project_variables import PROJECT_PATH, TRAINING, DEV, TEST, DEMO, DEMO2, LIU, LEX_TEST, REPO_DATASET, CHECKPOINT_DIR, SEED_TORCH, SEED_NP, LEX_TRAIN
from toolbox.gpu_related import use_gpu_
sys.path.insert(0, os.path.join(PROJECT_PATH, "..", "experimental_pipe"))
from env.project_variables import SCORE_AUX


def predict(batch_size, data_path,
            dict_path, model_full_name,
            bucket=False, model_specific_dictionary=True,
            print_raw=False, dir_normalized=None, dir_original=None,
            get_batch_mode=False,
            normalization=True, debug=False, use_gpu=None, verbose=0):

    assert model_specific_dictionary, "ERROR : only model_specific_dictionary = True supported now"
    # NB : now : you have to load dictionary when evaluating (cannot recompute) (could add in the LexNormalizer ability)
    use_gpu = use_gpu_(use_gpu)
    hardware_choosen = "GPU" if use_gpu else "CPU"
    printing("{} mode ", var=([hardware_choosen]), verbose_level=0, verbose=verbose)

    if not debug:
        pdb.set_trace = lambda: 1

    model = LexNormalizer(generator=Generator, load=True, model_full_name=model_full_name,
                          voc_size=None, use_gpu=use_gpu, dict_path=dict_path, model_specific_dictionary=True,
                          dir_model=os.path.join(PROJECT_PATH, "checkpoints",
                                                 model_full_name + "-folder"),
                          char_decoding=True, word_decoding=False,
                          verbose=verbose
                          )

    data_read = conllu_data.read_data_to_variable(data_path, model.word_dictionary, model.char_dictionary,
                                                  model.pos_dictionary,
                                                  model.xpos_dictionary, model.type_dictionary,
                                                  use_gpu=use_gpu,
                                                  norm_not_norm=model.auxilliary_task_norm_not_norm,
                                                  symbolic_end=True, symbolic_root=True,
                                                  dry_run=0, lattice=False, verbose=verbose,
                                                  normalization=normalization,
                                                  bucket=bucket,
                                                  add_start_char=1, add_end_char=1)

    batchIter = data_gen_conllu(data_read, model.word_dictionary, model.char_dictionary,
                                batch_size=batch_size,
                                get_batch_mode=False,
                                normalization=normalization,
                                print_raw=print_raw,  verbose=verbose)
    model.eval()
    greedy_decode_batch(char_dictionary=model.char_dictionary, verbose=verbose,
                        gold_output=False,
                        use_gpu=use_gpu,
                        write_output=True,
                        label_data=REPO_DATASET[data_path],
                        batchIter=batchIter, model=model, dir_normalized=dir_normalized, dir_original=dir_original,
                        batch_size=batch_size)


if __name__ == "__main__":
    list_all_dir = os.listdir(os.path.join(PROJECT_PATH, "checkpoints"))

    ablation_id = "21cc8-fixed_all_context-aux_dense1dir_word-200_aux-0do_char_dec-False_char_src-model_12_4d49"
    list_ = [dir_ for dir_ in list_all_dir if
             dir_.startswith(ablation_id) and not dir_.endswith("log") and not dir_.endswith(
                 ".json") and not dir_.endswith("summary")]
    folder_name = list_[0]
    print(folder_name, DEV)
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", help="", required=True )
    parser.add_argument("--data_dir", help="", required=True)
    parser.add_argument("--dir_original", help="", required=True)
    parser.add_argument("--dir_normalized", help="", required=True )
    args = parser.parse_args()

    folder_name = args.folder_name
    data_dir = args.data_dir

    model_full_name = folder_name[:-7]

    predict(model_full_name=model_full_name, data_path=data_dir,
            use_gpu=None, dict_path=os.path.join(PROJECT_PATH, "checkpoints", folder_name, "dictionaries"),
            normalization=True, model_specific_dictionary=True,
            batch_size=26, dir_original=args.dir_original, dir_normalized=args.dir_normalized,
            debug=False, bucket=True,
            verbose=1)

    "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/.././data/copy_paste_real_word_test.conll"

    # run  : python predict.py --folder_name "21cc8-fixed_all_context-aux_dense1dir_word-200_aux-0do_char_dec-False_\
    # char_src-model_12_4d49-folder" --data_dir  /Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../../parsing/normpar/data/owoputi.integrated_fixed \
    # --dir_normalized /Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/predictions/normalized.conll  \
    # --dir_original  /Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/predictions/or.conll


#python predict.py --folder_name "97942_rioc--DEBUG_NO_LOSS_PADDING-0-model_1-model_1_767d-folder" --data_dir  "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/env/../../parsing/normpar/data/owoputi.integrated_fixed" --dir_normalized "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/predictions/normalized.conll"  --dir_original  "/Users/bemuller/Documents/Work/INRIA/dev/mt_norm_parse/predictions/or.conll"