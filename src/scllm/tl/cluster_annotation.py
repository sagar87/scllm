from typing import List

import scanpy as sc
from anndata import AnnData
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# from langchain.schema.runnable import RunnableEach
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables.base import RunnableEach, RunnableLambda, RunnableParallel
from langchain_core.runnables.branch import RunnableBranch
from pydantic import BaseModel, Field


class CellType(BaseModel):
    cell_type: str = Field(description="The most likely cell type.")
    confidence: float = Field(
        description="The confidence in the cell type. Range from 0 to 1."
    )
    marker_genes: List[str] = Field(
        description="The marker genes for the cell type. List only genes that are expressed in the cell type."
    )


def _analyze_genelist(gene_list: List[str], format_instructions: str):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert biologist with extensive knowledge in single cell RNA-seq analysis.",
            ),
            (
                "human",
                "Identify the most likely cell type given the following genes: {genes}. {format_instructions}",
            ),
        ]
    )

    return prompt_template.format_prompt(
        genes=", ".join(gene_list), format_instructions=format_instructions
    )


def annotate_cluster(
    llm, adata: sc.AnnData, cluster_key: str, num_samples:int =1, use_raw: bool = False, top_genes: int = 10, 
) -> AnnData:
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

    # create a dictionary mapping cluster keys to their top genes { str: [] }
    cluster_genes = {}
    for group in adata.obs[cluster_key].unique():
        cluster_genes[group] = (
            sc.get.rank_genes_groups_df(
                adata, group=group, key=f"{cluster_key}_rank_genes_groups"
            )
            .head(top_genes)
            .names.tolist()
        )

    output_parser = PydanticOutputParser(pydantic_object=CellType)
    cell_type_branch = (
        RunnableLambda(
            lambda x: _analyze_genelist(x, output_parser.get_format_instructions())
        )
        | llm
        | output_parser
    )
    chain = RunnableLambda(lambda x: x * num_samples) | RunnableEach(bound=cell_type_branch)
    res = chain.invoke(list(cluster_genes.values()))

    return res
