import numpy as np
import pandas as pd
import os
import random
import torch


def set_seeds(seed_value=42, cuda_deterministic=False):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def merge_duplicated_rows(df, idsep=';'):
    """
    Accepts a pandas DataFrame and merges duplicated rows and concatenate their indices.
    """
    dupdf = df[df.duplicated(keep=False)]

    if dupdf.shape[0] == 0:
        print('No redundancy among rows found, returning input.')
        
        return df

    dupdfix = dupdf.index.values

    merged = []
    for _, val in dupdf.groupby(dupdf.columns.tolist()):
        inx = val.index.tolist()
        i = idsep.join(inx)
        v = val.max(0)
        v.name = i
        merged.append(v)

    df = df.drop(dupdfix)
    df = pd.concat([df, pd.concat(merged, axis=1).T])
    
    return df

