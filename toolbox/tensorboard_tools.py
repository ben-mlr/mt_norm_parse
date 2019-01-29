
from io_.info_print import printing


def writer_weights_and_grad(model, freq_writer, epoch, writer, verbose, report_grad=True, report_weight=True):
    if epoch % freq_writer == 0:
        for name, param in model.named_parameters():
            if report_weight:
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            if param.requires_grad and param.grad is not None and report_grad:
                writer.add_histogram("grad" + name + "-grad", param.grad.clone().cpu().data.numpy(), epoch)
        printing("REPORTING : storing weights and grad in writer ", verbose=verbose, verbose_level=0)

