import pandas as pd
import scanpy as sc
from langchain_core.language_models import BaseLanguageModel

from .chains import construct_term_chain_with_genes
from .utils import _prepare_mapping, _validate_top_genes


def _validate_cluster_key(adata: sc.AnnData, cluster_key: str):
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"Cluster key {cluster_key} not found in adata.obs")
    return cluster_key


def _prepare_chain_data(
    adata: sc.AnnData, cluster_key: str, top_genes: int = 10, num_samples: int = 1
):
    cluster_data = []
    for group in adata.obs[cluster_key].unique():
        genes = (
            sc.get.rank_genes_groups_df(
                adata,
                group=group,
                key=f"{cluster_key}_rank_genes_groups",
            )
            .head(top_genes)
            .names.tolist()
        )
        for i in range(num_samples):
            cluster_data.append({"cluster": group, "genes": genes, "init": i})
    return cluster_data


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


def annotate_cluster(
    adata: sc.AnnData,
    cluster_key: str,
    llm: BaseLanguageModel,
    num_samples: int = 1,
    key_added: str = "cluster_annotation",
    top_genes: int = 10,
    term: str = "cell type",
    extra: str = "",
    **kwargs,
) -> sc.AnnData:
    """Annotate cell clusters using marker genes and LLM-based analysis.

    This function performs differential expression analysis to identify marker genes
    for each cluster, then uses a language model to annotate the cell types based
    on these markers. It supports multiple sampling runs to assess annotation stability.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing single-cell data
    cluster_key : str
        Key in adata.obs containing cluster labels
    llm : BaseLanguageModel
        Language model instance for cell type annotation
    num_samples : int, default=1
        Number of times to run the annotation to assess stability
    key_added : str, default="cluster_annotation"
        Key under which to store the annotations in adata.obs
    top_genes : int, default=10
        Number of top marker genes to consider for each cluster
    **kwargs
        Additional arguments passed to sc.tl.rank_genes_groups

    Returns
    -------
    sc.AnnData
        AnnData object with added annotations in adata.obs[key_added] and
        detailed annotation results in adata.uns[f"scllm_{cluster_key}"]

    Examples
    --------
    >>> import scanpy as sc
    >>> from langchain_openai import ChatOpenAI
    >>> llm = ChatOpenAI()
    >>> adata = sc.read_h5ad("data.h5ad")
    >>> adata = annotate_cluster(adata, cluster_key="leiden", llm=llm)
    """
    num_genes = adata.shape[1]
    _ = _validate_top_genes(top_genes, num_genes)
    _ = _validate_cluster_key(adata, cluster_key)

    # rank genes
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        key_added=f"{cluster_key}_rank_genes_groups",
        **kwargs,
    )

    # create the input data for the input chain [{'cluster': cluster_id, 'genes': [list of top_genes]}]
    cluster_data = _prepare_chain_data(adata, cluster_key, top_genes, num_samples)

    # run the chain
    chain = construct_term_chain_with_genes(llm, term=term, extra=extra)
    out = chain.invoke(cluster_data)

    # map cluster id's to celltype
    df = pd.DataFrame(out)
    mapping = _prepare_mapping(df, "cluster", "term")
    var_names = _prepare_var_names(df, mapping)

    adata.obs[key_added] = adata.obs[cluster_key].astype(str).map(mapping)
    adata.uns[key_added] = {"raw": out, "mapping": mapping, "var_names": var_names}
