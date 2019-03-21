from training.args_tool import args_train
import re
from io_.info_print import printing
from training.train_eval import train_eval
from env.project_variables import DEFAULT_SCORING_FUNCTION, MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE , AVAILABLE_TASKS, DIC_ARGS


import os


def parse_argument_dictionary(argument_as_string, hyperparameter="multi_task_loss_ponderation", verbose=1):
    assert hyperparameter in DIC_ARGS , "ERROR only supported"
    if argument_as_string in MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE:
        return argument_as_string
    else:
        dic = {}
        for task in AVAILABLE_TASKS:
            if task!="all":
                pattern = "{}=([^=]*),".format(task)
                match = re.search(pattern, argument_as_string)
                assert match is not None, "ERROR : pattern {} not found for task {} in argument_as_string {}  ".format(pattern, task, argument_as_string)
                print("FOUND", match)
                dic[task] = eval(match.group(1))
        printing("SANITY CHECK : multi_task_loss_ponderation {} ", var=[params["multi_task_loss_ponderation"]],
                 verbose_level=1, verbose=verbose)
        return dic



if __name__ == "__main__":

    args = args_train()
    params = vars(args)
    params["multi_task_loss_ponderation"] = parse_argument_dictionary(params["multi_task_loss_ponderation"])
    print("DISTRIBUTED : TRAINING EVALUATE STARTING thread on GPU {} ".format(os.environ.get("CUDA_VISIBLE_DEVICES","NO GPU FOUND")))
    model_full_name, model_dir = train_eval(args=params,
                                            model_id_pref=args.model_id_pref,
                                            train_path=args.train_path,
                                            n_epochs=args.epochs,
                                            dev_path=args.dev_path,
                                            test_path=args.test_paths,
                                            expand_vocab_dev_test=args.expand_vocab_dev_test,
                                            pos_specific_path=args.pos_specific_path,
                                            overall_label=args.overall_label, overall_report_dir=args.overall_report_dir,
                                            extend_n_batch=2,
                                            get_batch_mode_all=True, compute_mean_score_per_sent=False, bucketing_train=True, freq_checkpointing=1,
                                            symbolic_end=True, symbolic_root=True, freq_writer=1, compute_scoring_curve=False,
                                            gpu=args.gpu,
                                            verbose=args.verbose,
                                            warmup=args.warmup,
                                            scoring_func_sequence_pred="BLUE",#DEFAULT_SCORING_FUNCTION,
                                            debug=args.debug)
    print("DISTRIBUTED : TRAINING EVALUATE DONE train_eval thread on done of model {} dir {} ".format(model_full_name, model_dir))
