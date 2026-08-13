import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import zscore


def generate_kinase_gs_aggregated(phos_ms_df, act_sites_df, prior_df=None, z_threshold=1.645, min_obs=30, min_var=0.1):
    """
    Generates Gold Standard (GS) sets, excluding kinase-site pairs found in the Prior.
    
    Parameters:
    -----------
    phos_ms_df : pd.DataFrame
        Phosphoproteomics data.
    act_sites_df : pd.DataFrame
        Metadata with columns ['ACC_ID', 'MOD_RSD', 'rsd', 'pos'].
    prior_df : pd.DataFrame, optional
        Prior knowledge matrix (Index=Kinase, Cols=Site). 
        If provided, activating sites with non-zero entries in this matrix 
        (for the corresponding kinase) will be excluded from GS generation.
    z_threshold : float
        Z-score cutoff (default 1.645).
    
    Returns:
    --------
    pd.DataFrame
        GS set with columns ['Kinase', 'Sample', 'Z_Score', 'Class'].
    """
    
    # 1. Prepare Metadata copy
    # We work on a copy to ensure the external act_sites_df is not modified
    work_act_df = act_sites_df.copy()
    
    # Construct Site ID (e.g., P33981:T46)
    work_act_df['Site_ID'] = (
        work_act_df['ACC_ID'].astype(str) + ':' + 
        work_act_df['rsd'].astype(str) + 
        work_act_df['pos'].astype(str)
    )
    
    # --- NEW LOGIC: Filter based on Prior ---
    if prior_df is not None:
        print(f"Filtering activating sites against Prior matrix...")
        initial_count = len(work_act_df)
        
        # Define a helper to check if a (Kinase, Site) pair exists in Prior with non-zero value
        # We assume Prior Index = Kinase ID, Prior Columns = Site ID
        def is_in_prior(row):
            kin = row['ACC_ID']
            site = row['Site_ID']
            
            # Check if dimensions exist in Prior
            if kin in prior_df.index and site in prior_df.columns:
                # Check if value is non-zero (implies connection)
                return prior_df.loc[kin, site] != 0
            return False

        # Identify rows to drop
        # applying row-wise is acceptable here as act_sites_df is typically small (<500 rows)
        mask_in_prior = work_act_df.apply(is_in_prior, axis=1)
        
        # Keep only rows NOT in prior
        work_act_df = work_act_df[~mask_in_prior]
        
        dropped_count = initial_count - len(work_act_df)
        print(f"Dropped {dropped_count} sites appearing in Prior. Remaining: {len(work_act_df)}")
        
        if work_act_df.empty:
            print("Warning: All activating sites were removed by Prior filtering.")
            return pd.DataFrame(columns=['Kinase', 'Sample', 'Z_Score', 'Class'])

    # ----------------------------------------

    # Create mapping for later
    site_to_kinase_map = dict(zip(work_act_df['Site_ID'], work_act_df['ACC_ID']))
    
    # 2. Filter Phos-MS Data
    common_sites = list(set(phos_ms_df.index) & set(work_act_df['Site_ID']))
    if not common_sites:
        print("Warning: No matching sites found in MS data after filtering.")
        return pd.DataFrame(columns=['Kinase', 'Sample', 'Z_Score', 'Class'])
    
    filtered_df = phos_ms_df.loc[common_sites]
    
    # 3. Quality Control
    obs_mask = filtered_df.notna().sum(axis=1) >= min_obs
    var_mask = filtered_df.var(axis=1, skipna=True) >= min_var
    cleaned_df = filtered_df.loc[filtered_df.index[obs_mask & var_mask]]
    
    if cleaned_df.empty:
        return pd.DataFrame(columns=['Kinase', 'Sample', 'Z_Score', 'Class'])
    
    # 4. Calculate Z-scores (Site Level)
    means = cleaned_df.mean(axis=1, skipna=True)
    stds = cleaned_df.std(axis=1, skipna=True, ddof=1)
    site_z_scores = cleaned_df.sub(means, axis=0).div(stds, axis=0)
    
    # 5. Aggregate to Kinase Level
    site_z_scores['Kinase'] = site_z_scores.index.map(site_to_kinase_map)
    # Mean of Z-scores for multiple sites of the same kinase
    kinase_level_scores = site_z_scores.groupby('Kinase').mean()
    
    # 6. Re-Standardize (Kinase Level)
    k_means = kinase_level_scores.mean(axis=1, skipna=True)
    k_stds = kinase_level_scores.std(axis=1, skipna=True, ddof=1)
    final_kinase_z = kinase_level_scores.sub(k_means, axis=0).div(k_stds, axis=0)

    # 7. Classify
    long_df = final_kinase_z.stack().reset_index()
    long_df.columns = ['Kinase', 'Sample', 'Z_Score']
    
    gs_positive = long_df[long_df['Z_Score'] > z_threshold].copy()
    gs_positive['Class'] = 1
    
    gs_negative = long_df[long_df['Z_Score'] < -z_threshold].copy()
    gs_negative['Class'] = 0
    
    final_gs = pd.concat([gs_positive, gs_negative], ignore_index=True)
    
    return final_gs.sort_values(by=['Kinase', 'Sample'])


def _get_gs_vals(score_df, gs_df, scale=False):
    """
    Extracts kinase scores for GS positive and negative samples.
    
    Parameters:
    -----------
    score_df : pd.DataFrame
        Kinase activity scores (Index: Kinase, Columns: Sample).
    gs_df : pd.DataFrame
        Gold Standard set (Columns: Kinase, Sample, Class). 
        Class 1 = Positive, 0 = Negative.
    scale : bool
        If True, row-scale (Z-score) the score_df.
        
    Returns:
    --------
    dict containing:
      - 'GSpos_values': Dict {kinase: [scores]}
      - 'GSneg_values': Dict {kinase: [scores]}
    """
    if scale:
        # Z-score normalization across samples (axis=1)
        score_df = score_df.apply(zscore, axis=1)
        
    pos_vals = {}
    neg_vals = {}
    
    # Get unique kinases present in both scores and GS
    common_kinases = sorted(set(score_df.index) & set(gs_df['Kinase']))
    
    for kinase in common_kinases:
        # Extract samples for this kinase from GS
        kin_gs = gs_df[gs_df['Kinase'] == kinase]
        pos_samples = kin_gs[kin_gs['Class'] == 1]['Sample'].values
        neg_samples = kin_gs[kin_gs['Class'] == 0]['Sample'].values
        
        # Get scores for these samples
        # Intersect with columns to ensure samples exist in score data
        valid_pos = sorted(set(pos_samples) & set(score_df.columns))
        valid_neg = sorted(set(neg_samples) & set(score_df.columns))
        
        # Extract values
        if valid_pos:
            p_scores = score_df.loc[kinase, valid_pos].dropna().values
            if len(p_scores) > 0:
                pos_vals[kinase] = p_scores
                
        if valid_neg:
            n_scores = score_df.loc[kinase, valid_neg].dropna().values
            if len(n_scores) > 0:
                neg_vals[kinase] = n_scores
                
    return {
        "GSpos_values": pos_vals,
        "GSneg_values": neg_vals
    }


def _roc_sampler(gs_pos_scores, gs_neg_scores, n_pert=1000, sub_samp_frac=0.8, seed=42):
    """
    Computes AUROC using subsampling (bootstrapping).
    
    Parameters:
    -----------
    gs_pos_scores : list/array of positive class scores
    gs_neg_scores : list/array of negative class scores
    n_pert : int, number of permutations
    sub_samp_frac : float, fraction of data to sample
    """
    np.random.seed(seed)
    aurocs = []
    
    n_pos = len(gs_pos_scores)
    n_neg = len(gs_neg_scores)
    
    # Determine sample sizes
    size_pos = int(n_pos * sub_samp_frac)
    size_neg = int(n_neg * sub_samp_frac)
    
    # Check if we have enough data to sample
    if size_pos < 1 or size_neg < 1:
        return np.array([np.nan])
        
    for _ in range(n_pert):
        # Sampling without replacement
        sample_pos = np.random.choice(gs_pos_scores, size=size_pos, replace=False)
        sample_neg = np.random.choice(gs_neg_scores, size=size_neg, replace=False)
        
        y_true = np.concatenate([np.ones(len(sample_pos)), np.zeros(len(sample_neg))])
        y_scores = np.concatenate([sample_pos, sample_neg])
        
        try:
            score = roc_auc_score(y_true, y_scores)
            aurocs.append(score)
        except ValueError:
            # Handle edge cases (e.g. only one class present)
            continue
            
    return np.array(aurocs) if len(aurocs) > 0 else np.array([np.nan])


def _pr_sampler(gs_pos_scores, gs_neg_scores, n_pert=1000, sub_samp_frac=0.8, seed=42):
    """
    Computes AUPRC (Average Precision) using subsampling (bootstrapping).
    """
    np.random.seed(seed)
    auprcs = []

    n_pos = len(gs_pos_scores)
    n_neg = len(gs_neg_scores)

    size_pos = int(n_pos * sub_samp_frac)
    size_neg = int(n_neg * sub_samp_frac)

    if size_pos < 1 or size_neg < 1:
        return np.array([np.nan])

    for _ in range(n_pert):
        sample_pos = np.random.choice(gs_pos_scores, size=size_pos, replace=False)
        sample_neg = np.random.choice(gs_neg_scores, size=size_neg, replace=False)

        y_true = np.concatenate([np.ones(len(sample_pos)), np.zeros(len(sample_neg))])
        y_scores = np.concatenate([sample_pos, sample_neg])

        try:
            score = average_precision_score(y_true, y_scores)
            auprcs.append(score)
        except ValueError:
            continue

    return np.array(auprcs) if len(auprcs) > 0 else np.array([np.nan])


def benchmark_roc(
    score_lists, gs_list, min_num=30, sub_samp_frac=0.8,
    metric=("auroc", "auprc"), n_pert=1000, seed=42,
    return_samples=True
):
    # 1) Pre-process Scores (Filter & Scale)
    processed_scores = {}
    cts = list(score_lists.keys())

    for ct in cts:
        df = score_lists[ct].copy()

        mask = df.notna().sum(axis=1) >= min_num
        df = df[mask]

        if ct in gs_list:
            common_kins = sorted(set(df.index) & set(gs_list[ct]["Kinase"]))
            df = df.loc[common_kins]

            processed_scores[ct] = df.apply(lambda x: zscore(x, nan_policy="omit"), axis=1)
        else:
            print(f"Warning: No GS data for {ct}")

    # 2) Pool GS pos/neg scores across cancer types
    pool_pos_scores = []
    pool_neg_scores = []

    for ct in sorted(processed_scores.keys()):
        gs_vals = _get_gs_vals(processed_scores[ct], gs_list[ct], scale=False)
        valid_kins = sorted(set(gs_vals["GSpos_values"]).intersection(gs_vals["GSneg_values"]))

        for k in valid_kins:
            pool_pos_scores.extend(gs_vals["GSpos_values"][k])
            pool_neg_scores.extend(gs_vals["GSneg_values"][k])

    print(f"Benchmarking with {len(pool_pos_scores)} positive pairs and {len(pool_neg_scores)} negative pairs.")

    # 3) Compute metrics via Sampling
    rows = []

    if "auroc" in metric:
        auroc_samples = _roc_sampler(
            pool_pos_scores, pool_neg_scores,
            n_pert=n_pert, sub_samp_frac=sub_samp_frac, seed=seed
        )
        for i, v in enumerate(auroc_samples):
            rows.append({"metric": "auroc", "iter": i, "value": float(v)})

    if "auprc" in metric:
        auprc_samples = _pr_sampler(
            pool_pos_scores, pool_neg_scores,
            n_pert=n_pert, sub_samp_frac=sub_samp_frac, seed=seed
        )
        for i, v in enumerate(auprc_samples):
            rows.append({"metric": "auprc", "iter": i, "value": float(v)})

    samples_df = pd.DataFrame(rows)

    # Optional
    if not return_samples:
        out = (
            samples_df.groupby("metric")["value"]
            .agg(value="mean", sd="std", n="count")
            .reset_index()
        )
        return out

    return samples_df