from typing import Literal

import scanpy as sc


# validator
def _validate_top_genes(top_genes: int, num_genes: int):
    if top_genes <= 0:
        raise ValueError("top_genes must be greater than 0")
    if top_genes > num_genes:
        raise ValueError("top_genes must be less than or equal to the number of genes")

    return top_genes


def _validate_factors(factors: list[str] | str, num_factors: int):
    """Validate factors have the right format."""
    if isinstance(factors, list):
        if len(factors) == 0:
            raise ValueError("Factors must be a non-empty list")

    if isinstance(factors, str):
        if factors == "all":
            factors = [str(i) for i in range(num_factors)]
        else:
            factors = [factors]

    return factors


def _validate_sign(sign: Literal["+", "-", "both"]):
    if sign not in ["+", "-", "both"]:
        raise ValueError("Sign must be one of: +, -, both")
    return sign


def _validate_cluster_key(adata: sc.AnnData, cluster_key: str):
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"Cluster key {cluster_key} not found in adata.obs")
    return cluster_key
