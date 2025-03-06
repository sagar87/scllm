import pandas as pd


def _prepare_mapping(df: pd.DataFrame, identity: str, target: str):
    mapping = (
        pd.crosstab(df[identity], df[target])
        .agg(["idxmax", "max"], axis=1)
        .loc[:, "idxmax"]
        .to_dict()
    )
    return mapping


def _validate_top_genes(top_genes: int, num_genes: int):
    if top_genes <= 0:
        raise ValueError("top_genes must be greater than 0")
    if top_genes > num_genes:
        raise ValueError("top_genes must be less than or equal to the number of genes")

    return top_genes
