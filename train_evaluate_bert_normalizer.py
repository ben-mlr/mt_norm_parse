from env.importing import *
from training.args_tool import args_train, parse_argument_dictionary
from io_.info_print import printing
from training.train_eval import train_eval
from env.project_variables import DEFAULT_SCORING_FUNCTION, MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE , AVAILABLE_TASKS, DIC_ARGS

from training.bert_normalize.train_eval_bert_normalize import train_eval_bert_normalize


if __name__ == "__main__":

    args = args_train(script="train_evaluate_bert_normalizer")
    params = vars(args)

    args.lr = parse_argument_dictionary(params["lr"], hyperparameter="lr")

    #if args.multitask:
    args.tasks = [task_simul.split(",") for task_simul in args.tasks]
        #args.multi_task_loss_ponderation = OrderedDict([("pos", 1), ("loss_task_2", 1),
        #                                                ("loss_task_n_mask_prediction", 1),
        #                                                ("parsing_types", 1), ("parsing_heads", 1)])

    if args.multi_task_loss_ponderation is not None:
        args.multi_task_loss_ponderation = parse_argument_dictionary(argument_as_string=params["multi_task_loss_ponderation"],
                                                                     tasks=[task for tasks in args.tasks for task in tasks],
                                                                     hyperparameter="multi_task_loss_ponderation")

        # TODO FACTORIZE MULTITASK
    train_eval_bert_normalize(args)
