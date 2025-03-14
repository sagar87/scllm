from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc


def _create_factor_df(
    adata: sc.AnnData,
    varm_key: str,
    factors: list[str] | str = "0",
    sign: Literal["+", "-"] = "+",
    top_genes: int = 10,
) -> pd.DataFrame:
    """Create a DataFrame containing the topfactor loadings for specified factors."""
    # validate factors have the right format
    # factors = _validate_factors(factors, adata.varm[varm_key].shape[1])

    # create factor dataframe
    factor_df = (
        pd.DataFrame(
            adata.varm[varm_key],
            index=adata.var_names.tolist(),
            columns=[f"{i}" for i in range(adata.varm[varm_key].shape[1])],
        )
        .reset_index()
        .rename(columns={"index": "gene"})
        .melt("gene", var_name="factor", value_name="weight")
        .loc[lambda df: df["factor"].isin(factors)]
        .sort_values(by=["factor", "weight"], ascending=False if sign == "+" else True)
        .groupby("factor")
        .head(top_genes)
    )
    return factor_df


def _prepare_factor_data(
    adata: sc.AnnData,
    varm_key: str,
    factors: list[str] | str = ["0"],
    sign: Literal["+", "-"] = "+",
    top_genes: int = 10,
    num_samples: int = 1,
) -> list[dict]:
    """Prepare data for the chain."""
    # validate factors have the right format
    # factors = _validate_factors(factors, adata.varm[varm_key].shape[1])

    # extract top factor weights
    factor_weights = _create_factor_df(adata, varm_key, factors, sign, top_genes)

    data = []
    for factor in factors:
        sub = factor_weights.loc[factor_weights["factor"] == factor]
        for i in range(num_samples):
            data.append(
                {
                    "factor": factor,
                    "genes": sub["gene"].tolist(),
                    "sign": sign,
                    "init": i,
                }
            )

    return data


def _prepare_cluster_data(
    adata: sc.AnnData,
    cluster_key: str,
    top_items: int = 10,
    num_samples: int = 1,
    **kwargs,
):
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        key_added=f"{cluster_key}_rank_genes_groups",
        **kwargs,
    )

    genes = sc.get.rank_genes_groups_df(
        adata,
        group=adata.obs[cluster_key].unique(),
        key=f"{cluster_key}_rank_genes_groups",
    )

    data = pd.concat(
        [
            (
                genes.groupby("group", observed=True)
                .head(top_items)
                .groupby("group", observed=True)
                .agg({"names": list})
                .reset_index()
                .loc[:, ["group", "names"]]
                .rename(columns={"names": "data"})
                .assign(init=i)
            )
            for i in range(num_samples)
        ]
    ).to_dict("records")

    return data


def _prepare_mapping(df: pd.DataFrame, identity: str, target: str):
    mapping = (
        pd.crosstab(df[identity], df[target])
        .agg(["idxmax", "max"], axis=1)
        .loc[:, "idxmax"]
        .to_dict()
    )

    return {k: np.str_(v) for k, v in mapping.items()}


def _prepare_var_names(df: pd.DataFrame, mapping: dict) -> dict:
    var_names = (
        df.groupby("term")
        .agg({"union": list})
        .reset_index()
        .assign(
            flattened=lambda df: df["union"].apply(
                lambda xss: [x for xs in xss for x in xs], 1
            )
        )
        .drop(columns=["union"])
        .set_index("term")
        .loc[lambda df: df.index.isin(mapping.values())]
    ).to_dict()["flattened"]
    return var_names
