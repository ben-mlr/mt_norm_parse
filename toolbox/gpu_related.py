import torch


def use_gpu_(use_gpu):
    if use_gpu is not None and use_gpu:
      assert torch.cuda.is_available() , "ERROR : use_gpu was set to True but cuda not available "
    use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
    return use_gpu
