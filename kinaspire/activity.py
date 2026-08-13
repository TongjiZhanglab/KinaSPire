import numpy as np
import pandas as pd
import torch

from .data import setup_dataloader_from_df


def cal_kin_act(data, prior, model, batch_size, meta_kin=False, device='cpu'):
    data_loader = setup_dataloader_from_df(df=data, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    
    kin_act = []
    for inputs, _ in data_loader:
        inputs = inputs.to(device)
        
        model = model.eval()
        with torch.no_grad():
            outputs = model.prior(inputs)
            if meta_kin:
                outputs = model.kki(outputs)
                
        kin_act.append(outputs)

    kin_act = torch.vstack(kin_act).detach().cpu().numpy()
    
    if not meta_kin:
        kin_act_df = pd.DataFrame(kin_act, index=data.index, columns=prior.index)
    else:
        kin_act_df = pd.DataFrame(kin_act, index=data.index, columns=[str(col) for col in np.arange(prior.shape[0])])

    return kin_act_df


def scale_kin_act(kin_act_df, prior):
    prior_split = split_cols(prior.T).T
    scaled_num = (prior_split > 0).sum(axis=1)
    scaled_num = scaled_num[scaled_num >= 1]
    
    kin_act_df_scaled = kin_act_df.copy().T
    common_cols = kin_act_df_scaled.columns.intersection(scaled_num.index)
    kin_act_df_scaled = (kin_act_df_scaled[common_cols] / scaled_num[common_cols]).T
    
    kin_act_df_scaled = kin_act_df_scaled.loc[~(kin_act_df_scaled == 0).all(axis=1), ~(kin_act_df_scaled == 0).all()]
    kin_act_df_scaled = ((kin_act_df_scaled.T - kin_act_df_scaled.T.min()) / (kin_act_df_scaled.T.max() - kin_act_df_scaled.T.min())).T

    return kin_act_df_scaled


def split_cols(df):
    for col in df.columns:
        if ';' in col:
            new_cols = col.split(';')
            for new_col in new_cols:
                df[new_col] = df[col].to_list()
    
    df = df.loc[:, ~df.columns.str.contains(';')]

    return df

