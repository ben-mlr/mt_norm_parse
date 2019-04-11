from env.importing import *
from training.args_tool import args_train, parse_argument_dictionary
from io_.info_print import printing
from training.train_eval import train_eval
from env.project_variables import DEFAULT_SCORING_FUNCTION, MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE , AVAILABLE_TASKS, DIC_ARGS


import os


if __name__ == "__main__":

    args = args_train()
    params = vars(args)

    params["multi_task_loss_ponderation"] = parse_argument_dictionary(params["multi_task_loss_ponderation"])
    print("DISTRIBUTED : TRAINING EVALUATE STARTING thread on GPU {} ".format(os.environ.get("CUDA_VISIBLE_DEVICES", "NO GPU FOUND")))
    model_full_name, model_dir = train_eval(args=params,
                                            model_id_pref=args.model_id_pref,
                                            checkpointing_metric=args.checkpointing_metric[0],
                                            train_path=args.train_path,
                                            n_epochs=args.epochs,
                                            dev_path=args.dev_path,
                                            test_path=args.test_paths,
                                            expand_vocab_dev_test=args.expand_vocab_dev_test,
                                            pos_specific_path=args.pos_specific_path,
                                            overall_label=args.overall_label, overall_report_dir=args.overall_report_dir,
                                            extend_n_batch=2,
                                            get_batch_mode_all=True, compute_mean_score_per_sent=False, bucketing_train=True, freq_checkpointing=1,
                                            symbolic_end=True, symbolic_root=True, freq_writer=1, compute_scoring_curve=True,
                                            gpu=args.gpu, 
                                            verbose=args.verbose,
                                            warmup=args.warmup, max_char_len=20,
                                            scoring_func_sequence_pred=params["scoring_func"], #DEFAULT_SCORING_FUNCTION,
                                            debug=args.debug)
    print("DISTRIBUTED : TRAINING EVALUATE DONE train_eval thread on done of model {} dir {} ".format(model_full_name, model_dir))
