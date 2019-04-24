from env.importing import *
from io_.info_print import printing

def use_gpu_(use_gpu, verbose=0):
    if use_gpu is not None and use_gpu:
      assert torch.cuda.is_available() , "ERROR : use_gpu was set to True but cuda not available "
    use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
    printing("HARDWARE : use_gpu set to {} ", var=[use_gpu], verbose=verbose, verbose_level=1)
    return use_gpu
