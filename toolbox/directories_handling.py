from env.importing import *
from io_.info_print import printing

def setup_repoting_location(root_dir_checkpoints, model_suffix="" , verbose=1):
    """
    create an id for a model and locations for checkpoints, dictionaries, tensorboard logs, data
    :param model_suffix:
    :param verbose:
    :return:
    """
    model_local_id = str(uuid4())[:5]
    if model_suffix != "":
        model_local_id += "-"+model_suffix
    model_location = os.path.join(root_dir_checkpoints, model_local_id)
    dictionaries = os.path.join(root_dir_checkpoints, model_local_id, "dictionaries")
    tensorboard_log = os.path.join(root_dir_checkpoints, model_local_id, "tensorboard")
    end_predictions = os.path.join(root_dir_checkpoints, model_local_id, "predictions")
    os.mkdir(model_location)
    printing("CHECKPOINTING model ID:{}", var=[model_local_id], verbose=verbose, verbose_level=1)
    os.mkdir(dictionaries)
    os.mkdir(tensorboard_log)
    os.mkdir(end_predictions)
    printing("CHECKPOINTING \n- {} for checkpoints \n- {} for dictionaries created \n- {} predictions",
             var=[model_location, dictionaries, end_predictions], verbose_level=1, verbose=verbose)
    return model_local_id, model_location, dictionaries, tensorboard_log, end_predictions


