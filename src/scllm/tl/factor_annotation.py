from typing import Literal

import pandas as pd
import scanpy as sc
from langchain_core.language_models import BaseLanguageModel

from .chains import FactorAnnotationChain


def _create_factor_df(
    adata: sc.AnnData,
    varm_key: str,
    factors: list[str] | str = "0",
    sign: Literal["+", "-"] = "+",
    top_genes: int = 10,
) -> pd.DataFrame:
    """Create a DataFrame containing factor loadings for specified factors.

    This helper function processes the factor loadings from an AnnData object and
    returns a DataFrame with the top genes for each specified factor.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing factor loadings in varm
    varm_key : str
        Key in adata.varm containing the factor loadings
    factors : list[str] | str, default="0"
        Factor(s) to analyze. Can be a single factor or list of factors
    sign : Literal["+", "-"], default="+"
        Whether to consider positive or negative loadings

    Returns
    -------
    pd.DataFrame
        DataFrame containing gene names, factor IDs, and loadings for the top genes
        of each specified factor
    """
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


def annotate_factor(
    adata: sc.AnnData,
    varm_key: str,
    llm: BaseLanguageModel,
    factors: list[str] | str = "0",
    num_samples: int = 1,
    key_added: str = "scllm",
    top_genes: int = 10,
    **kwargs,
) -> sc.AnnData:
    """Annotate factors using marker genes and LLM-based analysis.

    This function analyzes factor loadings from dimensionality reduction methods
    (e.g., PCA, NMF) and uses a language model to annotate the biological meaning
    of each factor based on its top genes.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing factor loadings in varm
    varm_key : str
        Key in adata.varm containing the factor loadings
    llm : BaseLanguageModel
        Language model instance for factor annotation
    factors : list[str] | str, default="0"
        Factor(s) to analyze. Can be a single factor or list of factors
    sign : Literal["+", "-"], default="+"
        Whether to consider positive or negative loadings
    num_samples : int, default=1
        Number of times to run the annotation to assess stability
    key_added : str, default="scllm_annotation"
        Key under which to store the annotations in adata.obs
    top_genes : int, default=10
        Number of top genes to consider for each factor
    **kwargs
        Additional arguments passed to the annotation chain

    Returns
    -------
    sc.AnnData
        AnnData object with added factor annotations in adata.obs[key_added]

    Examples
    --------
    >>> import scanpy as sc
    >>> from langchain_openai import ChatOpenAI
    >>> llm = ChatOpenAI()
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> adata = annotate_factor(adata, varm_key="pca", llm=llm, factors=["0", "1"])
    """
    if isinstance(factors, str):
        factors = [factors]

    factor_pos = _create_factor_df(adata, varm_key, factors, "+", top_genes)
    factor_neg = _create_factor_df(adata, varm_key, factors, "-", top_genes)

    data = []
    for factor in factors:
        factor_pos_sub = factor_pos.loc[factor_pos["factor"] == factor]
        factor_neg_sub = factor_neg.loc[factor_neg["factor"] == factor]
        data.append(
            {"factor": factor, "genes": factor_pos_sub["gene"].tolist(), "sign": "+"}
        )
        data.append(
            {"factor": factor, "genes": factor_neg_sub["gene"].tolist(), "sign": "-"}
        )

    res = FactorAnnotationChain(llm).invoke(data)
    adata.uns[key_added] = res
