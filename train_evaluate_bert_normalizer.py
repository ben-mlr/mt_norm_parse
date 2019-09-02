from env.importing import *
from training.args_tool import args_train, parse_argument_dictionary
from io_.info_print import printing
from training.train_eval import train_eval
from env.project_variables import DEFAULT_SCORING_FUNCTION, MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE , AVAILABLE_TASKS, DIC_ARGS

from training.bert_normalize.train_eval_bert_normalize import train_eval_bert_normalize


if __name__ == "__main__":

    args = args_train(script="train_evaluate_bert_normalizer")

    if args.test_paths is not None:
        args.test_paths = [test_path_task.split(",") for test_path_task in args.test_paths]
    if args.dev_path is not None:
        args.dev_path = [dev_path_task.split(",") for dev_path_task in args.dev_path]
    params = vars(args)

    args.lr = parse_argument_dictionary(params["lr"], hyperparameter="lr")

    #if args.multitask:
    args.tasks = [task_simul.split(",") for task_simul in args.tasks]
    #args.multi_task_loss_ponderation = OrderedDict([("pos", 1), ("loss_task_2", 1),
    #                                                ("loss_task_n_mask_prediction", 1),
    #                                                ("parsing_types", 1), ("parsing_heads", 1)])
    # TODO FACTORIZE MULTITASK
    train_eval_bert_normalize(args)
