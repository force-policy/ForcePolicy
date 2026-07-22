import torch
import torch.distributed as dist


def sync_loss(loss, device):
    t = [loss]
    t = torch.tensor(t, dtype = torch.float32, device = device)
    dist.barrier()
    dist.all_reduce(t, op = dist.ReduceOp.AVG)
    return t[0]
