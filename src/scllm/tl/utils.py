import pandas as pd
import scanpy as sc


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
    return mapping


def _validate_top_genes(top_genes: int, num_genes: int):
    if top_genes <= 0:
        raise ValueError("top_genes must be greater than 0")
    if top_genes > num_genes:
        raise ValueError("top_genes must be less than or equal to the number of genes")

    return top_genes
