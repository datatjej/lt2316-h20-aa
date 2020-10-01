
import torch

def ensure_tensor(item):
    if not torch.is_tensor(item):
        raise TypeError(f"torch.tensor was expected but {type(item)}")


def ensure_gpu(item):
    nr_gpus = torch.cuda.device_count()
    used_device = item.get_device()
    if not (used_device >= 0 and used_device <= nr_gpus):
        raise RuntimeError(f"Given data is not on GPU. {used_device} is not a GPU device.")
    

def check_output(items_to_check):

    for item in items_to_check:
        ensure_tensor(item)
        ensure_gpu(item)
    
    return items_to_check
