from io_.info_print import printing


def checkpoint(loss_former, loss, model, model_dir, epoch, epochs, info_checkpoint, verbose):
    if loss < loss_former:
        printing('Checkpoint info : Loss decreased so saving model', verbose=verbose, verbose_level=1)
        model.save(model_dir, model, info_checkpoint=info_checkpoint, suffix_name="{}ep-outof{}ep".format(epoch, epochs), verbose=verbose)
        loss_former = loss
    else:
        # TODO : load former checkpoint : and do change loss append
        print("Checkpoint info : Loss did not decrease")
        pass
    return model, loss_former