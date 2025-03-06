from typing import Literal

import pandas as pd
import scanpy as sc
from langchain_core.language_models import BaseLanguageModel

from .chains import construct_term_chain
from .utils import _prepare_mapping


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


def _prepare_chain_data(
    adata: sc.AnnData,
    varm_key: str,
    factors: list[str] | str = "0",
    sign: Literal["+", "-"] = "+",
    top_genes: int = 10,
    num_samples: int = 1,
) -> pd.DataFrame:
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


def annotate_factor(
    adata: sc.AnnData,
    varm_key: str,
    llm: BaseLanguageModel,
    factors: list[str] | str = "0",
    num_samples: int = 1,
    key_added: str = "scllm",
    top_genes: int = 10,
    sign: Literal["+", "both"] = "both",
    term: str = "cell type",
    extra: str = "",
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

    # prepare data for the chain
    data = _prepare_chain_data(adata, varm_key, factors, "+", top_genes, num_samples)
    if sign == "both":
        data_neg = _prepare_chain_data(
            adata, varm_key, factors, "-", top_genes, num_samples
        )
        data += data_neg

    chain = construct_term_chain(
        llm, term=term, extra=extra, passthrough=["factor", "sign", "init"]
    )

    out = chain.invoke(data)

    # generate the mostlikely mapping
    df = pd.DataFrame(out).assign(
        id=lambda df: df.apply(lambda row: row.factor + row.sign, 1)
    )
    mapping = _prepare_mapping(df, "id", "target")

    adata.uns[key_added] = {"raw": out, "mapping": mapping}
