import pandas as pd
import scanpy as sc
from langchain_core.language_models import BaseLanguageModel

from .chains import CellTypeAnnotationChain

# def _analyze_genelist(gene_list: List[str], format_instructions: str):
#     prompt_template = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are an expert biologist with extensive knowledge in single cell RNA-seq analysis.",
#             ),
#             (
#                 "human",
#                 "Identify the most likely cell type given the following genes: {genes}. {format_instructions}",
#             ),
#         ]
#     )

#     return prompt_template.format_prompt(
#         genes=", ".join(gene_list), format_instructions=format_instructions
#     )


def annotate_cluster(
    adata: sc.AnnData,
    cluster_key: str,
    llm: BaseLanguageModel,
    num_samples: int = 1,
    use_raw: bool = False,
    key_added: str = "scllm_annotation",
    top_genes: int = 10,
) -> sc.AnnData:
    """
    Annotate a cluster with a cell type

    """

    # rank genes
    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        use_raw=use_raw,
        key_added=f"{cluster_key}_rank_genes_groups",
    )

    # create the input data for the input chain
    cluster_data = []
    for group in adata.obs[cluster_key].unique():
        genes = (
            sc.get.rank_genes_groups_df(
                adata, group=group, key=f"{cluster_key}_rank_genes_groups"
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

    mapping = (
        pd.crosstab(df["cluster"], df["cell_type"])
        .agg(["idxmax", "max"], axis=1)
        .loc[:, "idxmax"]
        .to_dict()
    )

    adata.obs[key_added] = adata.obs[cluster_key].astype(str).map(mapping)
    adata.uns[f"scllm_{cluster_key}"] = df
