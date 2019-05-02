from env.importing import *
from training.args_tool import args_train, parse_argument_dictionary
from io_.info_print import printing
from training.train_eval import train_eval
from env.project_variables import DEFAULT_SCORING_FUNCTION, MULTI_TASK_LOSS_PONDERATION_PREDEFINED_MODE , AVAILABLE_TASKS, DIC_ARGS

from training.bert_normalize.train_eval_bert_normalize import train_eval_bert_normalize


if __name__ == "__main__":

    args = args_train(script="train_evaluate_bert_normalizer")
    params = vars(args)

    train_eval_bert_normalize(args)
