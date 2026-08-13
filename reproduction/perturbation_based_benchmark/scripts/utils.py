from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.metrics import average_precision_score, roc_auc_score


def run_rank(act: pd.DataFrame, meta: pd.DataFrame, average: bool = True) -> pd.DataFrame:
    """
    Calculate (after sign unification) the rank and scaled_rank of perturbed kinases in each experiment.

    Parameters
    ----------
    act : DataFrame
        Activity score matrix, row = kinase, column = experiment.
    meta : DataFrame
        Must contain at least three columns:
        - id: Experiment ID (matches act column names)
        - target: Perturbed kinase name; if multiple, use ';' to join
        - sign: Perturbation direction (+1 = activation/overexpression, -1 = inhibition/knockdown)
    average : bool, default True
        If True, calculates average rank for the same kinase across multiple experiments (consistent with R version).

    Returns
    -------
    DataFrame
        If average=True: columns are ['targets','rank','scaled_rank','sample']
        If average=False: columns are
        ['sample','targets','rank','kinases_act','all_kinases_act','scaled_rank']
    """
    # ---- Process meta, filter experiments without target ----
    obs = (
        meta.rename(columns={"target": "perturb"})
            .groupby("id", as_index=False)
            .agg(targets=("perturb", lambda x: ";".join(map(str, x))),
                 sign=("sign", lambda x: pd.unique(x)[0]))
            .rename(columns={"targets": "perturb", "id": "sample"})
    )
    obs = obs[(obs["perturb"].notna()) & (obs["perturb"] != "")]

    # Only keep experiments present in act
    samples = [c for c in act.columns if c in set(obs["sample"])]
    if not samples:
        return pd.DataFrame(columns=["sample","targets","rank","kinases_act",
                                     "all_kinases_act","scaled_rank"])

    # Wide -> long format
    method_act_long = (
        act.loc[:, samples]
           .reset_index()
           .rename(columns={"index": "kinase"})
           .melt(id_vars="kinase", var_name="sample", value_name="score")
           .dropna(subset=["score"])
    )

    rows = []
    for exp in method_act_long["sample"].unique():
        # Select experiment, sort by direction
        sub = method_act_long[method_act_long["sample"] == exp].copy()
        sub = sub.sort_values("score", ascending=False).reset_index(drop=True)
        n_kin = len(sub)

        # The target kinases for this experiment (can be multiple, separated by ;)
        if exp not in set(obs["sample"]):
            continue
        targets = (
            obs.loc[obs["sample"] == exp, "perturb"]
               .iloc[0].split(";")
        )

        # For each target, find rank (1-based, as in R)
        for t in targets:
            idx = sub.index[sub["kinase"] == t]
            if len(idx) == 0:
                continue  # Skip if not in the matrix (set NA in R, filter later)
            rank = int(idx[0]) + 1
            rows.append({
                "sample": exp,
                "targets": t,
                "rank": rank,
                "kinases_act": n_kin,
                "all_kinases_act": ";".join(sub["kinase"].tolist()),
                "scaled_rank": rank / n_kin
            })

    rank_df = pd.DataFrame(rows)
    if rank_df.empty:
        return rank_df

    if average:
        # Aggregate by kinase: mean rank & mean scaled_rank; concatenate sample names by ';'
        agg = (rank_df
               .groupby("targets", as_index=False)
               .agg(rank=("rank", "mean"),
                    scaled_rank=("scaled_rank", "mean"),
                    sample=("sample", lambda x: ";".join(map(str, x)))))
        return agg

    return rank_df


def run_phit(act: pd.DataFrame, meta: pd.DataFrame, k: int = 10, average: bool = True, metric='rank') -> float:
    """
    Calculate P_hit(k): proportion of perturbed kinases ranking in the top-k.

    Parameters
    ----------
    act : DataFrame
        Kinase × experiment activity matrix.
    meta : DataFrame
        Contains columns id/target/sign.
    k : int
        top-k threshold.
    average : bool
        Whether to pre-average rank across experiments for the same kinase (as per R code).

    Returns
    -------
    float
        Hit rate (0~1).
    """
    rank_df = run_rank(act, meta, average=average)
    if rank_df.empty:
        return np.nan
    phit = np.mean(rank_df[metric] <= k)
    return float(phit)

def compute_recall_at_k(meta: pd.DataFrame,
                        mat: pd.DataFrame,
                        k: int = 20,
                        mode: str = "macro",
                        metric='rank') -> dict:
    """
    Calculate Recall@k = (# of true targets in top k) / (total # of true targets)

    Parameters
    ----------
    meta : pd.DataFrame
        Must include columns:
        - 'id': Experiment ID (matches columns of mat)
        - 'target': Perturbed kinases (can be multiple, separated by ';' or multiple rows)
    mat : pd.DataFrame
        Kinase × experiment matrix, index: kinase, columns: experiment.
    k : int
        top-k threshold.
    mode : {'macro', 'micro'}
        - 'macro': recall per experiment, then average (default)
        - 'micro': pool all hits/targets for all experiments

    Returns
    -------
    dict
        {
          'recall_at_k': float,
          'n_total_targets': int,
          'n_hits': int,
          'n_experiments': int,
          'recall_per_experiment': dict   # recall for each experiment
        }
    """
    recalls = {}
    n_hits_global = 0
    n_total_global = 0

    for exp in mat.columns:
        if exp not in meta["id"].values:
            continue

        # Get target set for current experiment
        targets = meta.loc[meta["id"] == exp, "target"].tolist()
        targets = [t for tg in targets for t in str(tg).split(";")]
        targets = [t for t in targets if t in mat.index]
        n_total = len(targets)
        if n_total == 0:
            continue

        # Find top-k kinases
        pct = False if metric == 'rank' else 'True'
        ranks = mat[exp].rank(ascending=False, pct=pct)
        topk = set(ranks[ranks <= k].index)

        # Count hits
        hits = sum(t in topk for t in targets)
        recall = hits / n_total
        recalls[exp] = recall

        # Update global stats
        n_hits_global += hits
        n_total_global += n_total

    # Summarize
    if mode == "macro":
        recall_at_k = np.mean(list(recalls.values())) if recalls else np.nan
    elif mode == "micro":
        recall_at_k = n_hits_global / n_total_global if n_total_global else np.nan
    else:
        raise ValueError("mode must be 'macro' or 'micro'")

    return {
        "recall_at_k": recall_at_k,
        "n_total_targets": n_total_global,
        "n_hits": n_hits_global,
        "n_experiments": len(recalls),
        "recall_per_experiment": recalls
    }

def compute_mean_rank(meta: pd.DataFrame,
                      mat: pd.DataFrame,
                      missing_penalty: float = 1.0) -> dict:
    """
    Compute average normalized rank (the smaller the better) for target kinases in perturbation experiments,
    and apply penalty rank for kinases not found.

    Parameters
    ----------
    meta : pd.DataFrame
        Must include columns:
        - 'id': Experiment ID (matches columns of mat)
        - 'target': Perturbed kinases (can be multiple, separated by ';' or multiple rows)
    mat : pd.DataFrame
        Kinase × experiment matrix (rows = kinase, columns = experiment)
    missing_penalty : float, default 1.0
        Penalty quantile assigned to missing target kinases (0~1, 1=worst)

    Returns
    -------
    dict
        {
            'mean_rank': float,              # Average quantile rank of all targets
            'n_total': int,                  # Total number of targets
            'n_valid': int,                  # Number of targets found in matrix
            'rank_table': pd.DataFrame       # Detailed results table
        }
    """
    results = []

    for exp in mat.columns:
        if exp not in meta["id"].values:
            continue

        # Current experiment's score ranking (smaller = more significant)
        tmp_act = mat[exp].copy().dropna()
        tmp_rank = tmp_act.rank(pct=True, ascending=False)

        # Get all target kinases for this experiment (support multiple targets)
        tmp_target = meta.loc[meta['id'] == exp, 'target'].tolist()
        tmp_target = [t for tg in tmp_target for t in str(tg).split(";")]

        for target in tmp_target:
            if target in tmp_rank.index:
                r = tmp_rank.loc[target]
                status = "valid"
            else:
                r = missing_penalty
                status = "missing"

            results.append({
                "experiment": exp,
                "target": target,
                "rank_pct": float(r),
                "status": status
            })

    # Summarize results
    rank_df = pd.DataFrame(results)
    if rank_df.empty:
        return {"mean_rank": np.nan, "n_total": 0, "n_valid": 0, "rank_table": rank_df}

    mean_rank = rank_df["rank_pct"].mean()
    n_total = len(rank_df)
    n_valid = (rank_df["status"] == "valid").sum()

    return {
        "mean_rank": mean_rank,
        "n_total": n_total,
        "n_valid": n_valid,
        "rank_table": rank_df
    }

# ---------- Utility: Column Scaling ----------
def scale_scores(act: pd.DataFrame, scaling: str = "sd") -> pd.DataFrame:
    """
    Scale each experiment column: 'sd' for standard deviation, 'max' for max(|.|).
    """
    act = act.copy()
    if scaling == "max":
        return act.apply(lambda s: s / np.nanmax(np.abs(s.values)) if np.nanmax(np.abs(s.values)) not in (0, np.nan) else s)
    elif scaling == "sd":
        # Equivalent to R::scale(center=FALSE, scale=TRUE): divide each column by column SD
        return act.apply(lambda s: s / np.nanstd(s.values) if np.nanstd(s.values) not in (0, np.nan) else s)
    else:
        return act

# ---------- Remove background experiments ----------
def remove_bg(df_exp_rows: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """
    df_exp_rows: rows = experiment, columns = kinase; value = sign-unified score (may be NA)
    obs: index = experiment; column: perturb (list[str])
    Logic: If sum of all scores for target kinases in this experiment is 0 or NA
           (i.e., nobody had inferred activity), then remove this experiment.
    """
    keep_rows = []
    for exp, row in df_exp_rows.iterrows():
        targets: List[str] = obs.loc[exp, "perturb"]
        targets_in_mat = [t for t in targets if t in df_exp_rows.columns]
        if not targets_in_mat:
            # No target in the matrix, means no info, remove
            continue
        s = row[targets_in_mat]
        sum_act = np.nansum(s.values)
        # R logic: !is.na(sum_act) & !sum_act == 0
        if not np.isnan(sum_act) and not (sum_act == 0):
            keep_rows.append(exp)
    return df_exp_rows.loc[keep_rows]

# ---------- Preprocessing: Sign unification + reshape ----------
def prepare_bench(act: pd.DataFrame,
                  meta: pd.DataFrame,
                  rm_bg: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_exp_rows: rows = experiment, columns = kinase, value = sign-unified score (0->NA)
      obs: index = experiment, column: perturb (list[str]) (sign is already absorbed, so set to 1)
    """
    # Process meta: combine targets for same experiment, take unique sign
    target_df = (meta.rename(columns={"target": "perturb"})
                      .groupby("id", as_index=False)
                      .agg(targets=("perturb", lambda x: ";".join(map(str, x))),
                           sign=("sign", lambda x: pd.unique(x)[0]))
                      .rename(columns={"targets": "perturb", "id": "sample"}))
    # Remove empty targets
    target_df = target_df[(target_df["perturb"].notna()) & (target_df["perturb"] != "")]
    # Keep only experiments present in act
    target_df = target_df[target_df["sample"].isin(act.columns)]
    if target_df.empty:
        raise ValueError("No valid experiments remain after filtering.")

    # Sign unification: multiply each experiment's column by sign
    # Make a DataFrame, rows=experiment
    rows = []
    index_names = []
    for _, r in target_df.iterrows():
        exp = r["sample"]; sign = r["sign"]
        # Take column of this experiment
        col = act[exp].copy()
        # Sign unification
        col = col * (1 if sign == 1 else -1)
        rows.append(col.values)
        index_names.append(exp)

    df_exp_rows = pd.DataFrame(rows, index=index_names, columns=act.index)

    # 0 -> NA (as in R)
    df_exp_rows = df_exp_rows.replace(0, np.nan)

    # obs: split target into list; sign set to 1 (already absorbed)
    obs = (target_df.assign(sign=1,
                            perturb=target_df["perturb"].apply(lambda s: s.split(";")))
                    .set_index("sample")[["perturb", "sign"]])

    # Optional: remove background experiments
    if rm_bg:
        df_exp_rows = remove_bg(df_exp_rows, obs)
        obs = obs.loc[df_exp_rows.index]

    return df_exp_rows, obs

# ---------- AUROC (rank-sum, no sklearn needed) ----------
def _auroc_from_scores(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """
    AUROC = P(pos > neg). Uses average rank for ties.
    """
    pos_scores = np.asarray(pos_scores, dtype=float)
    neg_scores = np.asarray(neg_scores, dtype=float)
    pos_scores = pos_scores[~np.isnan(pos_scores)]
    neg_scores = neg_scores[~np.isnan(neg_scores)]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return np.nan

    x = np.concatenate([pos_scores, neg_scores])
    # Use pandas average rank
    ranks = pd.Series(x).rank(method="average").to_numpy()
    r_pos = ranks[:n_pos].sum()
    auroc = (r_pos - n_pos*(n_pos+1)/2.0) / (n_pos * n_neg)
    return float(auroc)

# ---------- Core: run benchmark and return median AUROC ----------
def run_perturb_bench(act: pd.DataFrame,
                      meta: pd.DataFrame,
                      scale_data: bool = True,
                      rm_bg: bool = False,
                      n_iter: int = 1000,
                      metric: str = "auroc",
                      random_state: Optional[int] = None) -> Dict[str, float]:
    """
    Equivalent to R's run_perturbBench (here implements AUROC; extend for AUPRC if needed).
    Returns {'auroc': median_AUROC}
    """
    rng = np.random.default_rng(random_state)

    # 1) Optional column-wise scaling
    if scale_data:
        act = scale_scores(act, scaling="sd")

    # 2) Sign unify & reshape
    df_exp_rows, obs = prepare_bench(act, meta, rm_bg=rm_bg)

    # 3) Build global positive/negative sample pool (across all experiments)
    pos_pool = []
    neg_pool = []
    for exp, row in df_exp_rows.iterrows():
        available_kin = row.index[row.notna()].tolist()
        if not available_kin:
            continue
        targets = [t for t in obs.loc[exp, "perturb"] if t in available_kin]
        if len(targets) == 0:
            continue
        # Positive class: target kinases
        pos_pool.extend(row[targets].values.tolist())
        # Negative class: all other kinases
        other_kin = [k for k in available_kin if k not in targets]
        neg_pool.extend(row[other_kin].values.tolist())

    pos_pool = np.array(pos_pool, dtype=float)
    neg_pool = np.array(neg_pool, dtype=float)
    pos_pool = pos_pool[~np.isnan(pos_pool)]
    neg_pool = neg_pool[~np.isnan(neg_pool)]

    if len(pos_pool) == 0 or len(neg_pool) == 0:
        return np.nan

    # 4) Downsample (balanced), repeat n_iter times to compute AUROC
    n = min(len(pos_pool), len(neg_pool))
    aurocs = []
    for _ in range(n_iter):
        # Randomly sample equal number
        pos_idx = rng.choice(len(pos_pool), size=n, replace=(n > len(pos_pool)))
        neg_idx = rng.choice(len(neg_pool), size=n, replace=(n > len(neg_pool)))
        au = _auroc_from_scores(pos_pool[pos_idx], neg_pool[neg_idx])
        if not np.isnan(au):
            aurocs.append(au)

    if len(aurocs) == 0:
        return np.nan

    return aurocs

def cal_perfs(dp1, dp2, wd, cv, refit, act_df_signed, act_df, meta):
    out_row = {}
    out_row['dp1'] = dp1; out_row['dp2'] = dp2; out_row['wd'] = wd
    out_row['cv'] = cv; out_row['refit'] = refit
    
    phit_val = run_phit(act_df_signed, meta, k=10, average=True)
    out_row['phit_paper'] = phit_val

    rank_tbl = run_rank(act_df_signed, meta, average=True)
    out_row['avg_rank_paper'] = rank_tbl['scaled_rank'].mean() if len(rank_tbl) > 0 else np.nan

    out = compute_recall_at_k(meta, act_df_signed, k=10, mode='macro')
    out_row['recall_macro'] = out['recall_at_k']

    out = compute_recall_at_k(meta, act_df_signed, k=10, mode='micro')
    out_row['recall_micro'] = out['recall_at_k']

    out = compute_mean_rank(meta, act_df_signed, missing_penalty = np.nan)
    out_row['avg_rank_modified'] = out['mean_rank']

    out = run_perturb_bench(act_df, meta, scale_data=True, rm_bg=False, n_iter=1000, random_state=42)
    out_row['median_roc'] = np.median(out) if out is not np.nan else out

    return out_row

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