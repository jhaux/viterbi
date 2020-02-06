import numpy as np
import torch


def prepare_input(log_dict: dict):
    for key, value in log_dict.items():
        if isinstance(value, list):
            log_dict[key] = [numpy_to_torch(single_item) for single_item in value]
        else:
            log_dict[key] = numpy_to_torch(value)


def prepare_logs(log_dict: dict):
    for k, v in log_dict.items():
        if isinstance(v, list):
            log_dict[k] = [torch_to_numpy(single_v) for single_v in v]
        elif isinstance(v, torch.Tensor):
            log_dict[k] = torch_to_numpy(v)
        else:
            log_dict[k] = v

def torch_to_numpy(log_val: torch.Tensor):
    return log_val.mean().cpu().detach().numpy()


def numpy_to_torch(numpy_input: np.ndarray):
    return torch.tensor(numpy_input).cuda().float()
