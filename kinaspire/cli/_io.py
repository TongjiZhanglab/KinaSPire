from pathlib import Path

import pandas as pd


def load_table(path, index_col=0):
    """Load a table from parquet, feather, tsv, or csv."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        table = pd.read_feather(path)
        if "index" in table.columns:
            table = table.set_index("index")
        return table
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", index_col=index_col)
    if suffix == ".csv":
        return pd.read_csv(path, index_col=index_col)
    raise ValueError(f"Unsupported table format: {path}")


def save_table(table, path):
    """Save a table to parquet, tsv, or csv based on the file suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        table.to_parquet(path)
    elif suffix in {".tsv", ".txt"}:
        table.to_csv(path, sep="\t")
    elif suffix == ".csv" or suffix == "":
        table.to_csv(path)
    else:
        raise ValueError(f"Unsupported output format: {path}")
