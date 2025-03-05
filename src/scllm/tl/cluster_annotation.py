import pandas as pd
import scanpy as sc
from langchain_core.language_models import BaseLanguageModel

from .chains import CellTypeAnnotationChain


def annotate_cluster(
    adata: sc.AnnData,
    cluster_key: str,
    llm: BaseLanguageModel,
    num_samples: int = 1,
    key_added: str = "scllm_annotation",
    top_genes: int = 10,
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
    key_added : str, default="scllm_annotation"
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
    # rank genes
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        key_added=f"{cluster_key}_rank_genes_groups",
        **kwargs,
    )

    # create the input data for the input chain [{'cluster': cluster_id, 'genes': [list of top_genes]}]
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

        cluster_data.append({"cluster": group, "genes": genes})

    out = [
        pd.DataFrame(CellTypeAnnotationChain(llm).invoke(cluster_data)).assign(init=1)
        for i in range(num_samples)
    ]
    df = pd.concat(out)

    # map cluster id's to celltype
    mapping = (
        pd.crosstab(df["cluster"], df["cell_type"])
        .agg(["idxmax", "max"], axis=1)
        .loc[:, "idxmax"]
        .to_dict()
    )

    adata.obs[key_added] = adata.obs[cluster_key].astype(str).map(mapping)
    adata.uns[f"scllm_{cluster_key}"] = df
