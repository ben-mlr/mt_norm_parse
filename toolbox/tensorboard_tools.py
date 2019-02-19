
from io_.info_print import printing
TEST_CLIPPING, CLIP = False, 5


def writer_weights_and_grad(model, freq_writer, epoch, writer, verbose, report_grad=True, report_weight=True):
    if epoch % freq_writer == 0:
        # # TODO : make this a unit test in test/
        if TEST_CLIPPING:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    norm = param.grad.norm()
                    print("grad_norm writer_weights_and_grad", norm)
                    assert norm < CLIP
        for name, param in model.named_parameters():
            print("HERE ; parameters name ", name)
            if report_weight:
                try:
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                except Exception as e:
                    print("ERROR unable to report histogram ")
                    print(e)
            if param.requires_grad and param.grad is not None and report_grad:
                print("HERE ; parameters name iwth grad", name)
                writer.add_histogram("grad" + name + "-grad", param.grad.clone().cpu().data.numpy(), epoch)
        printing("REPORTING : storing weights and grad in writer ", verbose=verbose, verbose_level=0)

