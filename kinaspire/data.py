import numpy as np
import warnings
import torch
from torch.utils.data import TensorDataset, DataLoader

from ._utils import set_seeds


class DataSplitter(object):
    """
    Class for splitting phos-MS data.

    :param data: phos-MS data, dataframe, sample * site
    """
    def __init__(self, data, data_val_size=0.5, data_test_size=0):
        super().__init__()

        self.data = data
        self.data_val_size = data_val_size
        self.data_test_size = data_test_size

    def split(self, random_state=42):
        total_samples = self.data.shape[0]
        num_val_samples = int(self.data_val_size * total_samples)
        num_test_samples = int(self.data_test_size * total_samples)
        num_train_samples = total_samples - num_val_samples - num_test_samples
        
        # Randomly shuffle the data
        set_seeds(random_state)
        shuffled_indices = torch.randperm(total_samples).tolist()
        train_indices = shuffled_indices[:num_train_samples]
        val_indices = shuffled_indices[num_train_samples : num_train_samples + num_val_samples]
        test_indices = shuffled_indices[num_train_samples + num_val_samples:]
        
        # split phos-MS data
        self.train_data = self.data.iloc[train_indices,:]
        self.val_data = self.data.iloc[val_indices,:]
        self.test_data = self.data.iloc[test_indices]

        # print(f"train_data size: {self.train_data.shape[0]}, val_data size: {self.val_data.shape[0]}, test_data size: {self.test_data.shape[0]}\n")

        return self.train_data, self.val_data, self.test_data


def split_prior(prior, rng=np.random.default_rng(42), fraction_gs=0.1):
    """
    Function that splits a matrix by row subsets while preserving at least 1 nonzero element per columns.
    """
    if (prior.sum(1) == 0).sum() != 0:
        print('Prior have 0s in Kinase, can not split.\nReturning without change')
        return prior, None

    if fraction_gs == 0:
        return prior, prior

    use_prior = prior.copy()
    use_prior = use_prior.loc[:, use_prior.astype(bool).sum(0) != 0].copy()
    use_gs = use_prior.copy()
    
    _, cols = use_prior.shape

    numzero = 1
    counter = 0
    while numzero > 0:
        colsids = np.arange(cols)

        rng.shuffle(colsids)
        colsgs = sorted(colsids[:int(cols * fraction_gs)])
        colspri = sorted(colsids[int(cols * fraction_gs):])
        
        tmp = use_prior.copy()
        tmp.iloc[:, colsgs] = 0
        numzero = (tmp.astype(bool).sum(1) == 0).sum()

        counter += 1

        if not (counter % 100):
            print(f'Attempts to create full prior: {counter}', end='\r')

    use_prior.iloc[:, colsgs] = 0
    use_prior = use_prior.reindex(columns=prior.columns, fill_value=0)

    use_gs.iloc[:, colspri] = 0
    use_gs = use_gs.loc[:, use_gs.astype(bool).sum(0) != 0].copy()

    if use_prior[use_gs.columns].astype(bool).sum().sum() > 0:
        warnings.warn('Overlap between prior and gold standard exist.')

    if (use_prior.sum(1) == 0).sum():
        warnings.warn('There appear to be kinases in prior that do not have phosphosite connections. Something is wrong!')
    
    use_gs = use_gs[use_gs.sum(1) > 0]

    return use_prior, use_gs


def setup_dataloader_from_df(df, batch_size, shuffle=True):
    if df.shape[0] != 0:
        dataset = TensorDataset(torch.tensor(np.array(df), dtype=torch.float), torch.tensor(np.array(df), dtype=torch.float))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = None

    return data_loader

