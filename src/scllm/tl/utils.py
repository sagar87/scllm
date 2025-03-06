import pandas as pd


def _prepare_mapping(df: pd.DataFrame, identity: str, target: str):
    mapping = (
        pd.crosstab(df[identity], df[target])
        .agg(["idxmax", "max"], axis=1)
        .loc[:, "idxmax"]
        .to_dict()
    )
    return mapping
