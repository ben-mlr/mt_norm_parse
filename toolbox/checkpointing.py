from io_.info_print import printing
import torch

def checkpoint(loss_former, loss, model, model_dir, epoch, epochs, info_checkpoint, verbose):
    if loss < loss_former:
        printing('Checkpoint info : Loss decreased so saving model', verbose=verbose, verbose_level=1)
        model.save(model_dir, model, info_checkpoint=info_checkpoint, suffix_name="Xep-outof{}ep".format(epochs), verbose=verbose)
        loss_former = loss
    else:
        #printing('Checkpoint info : Loss decreased so saving model', verbose=verbose, verbose_level=1)
        #model.load_state_dict(torch.load(checkpoint_dir))
        # TODO : load former checkpoint : and do change loss append IF error suddendly pick
        printing("Checkpoint info : Loss did not decrease", verbose=verbose, verbose_level=1)

    return model, loss_former