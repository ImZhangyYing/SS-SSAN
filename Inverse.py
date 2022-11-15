import torch
from torch.linalg import det



def invmat(M):
    if det(M) == 0:
        a = torch.pinverse(M)
    else:
        a = torch.inverse(M)
    return a
