import random

import scanpy as sc
from anndata import AnnData
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from .prompts import SYSTEM_PROMPT, USER_PROMPT


def annotate_cluster(
    llm, adata: sc.AnnData, cluster_key: str, use_raw: bool = False, top_genes: int = 10
) -> AnnData:
    """
    Annotate a cluster with a cell type
    """

    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        use_raw=use_raw,
        key_added=f"{cluster_key}_rank_genes_groups",
    )

    mapping = {}

    for cluster in adata.obs[cluster_key].unique():
        df = sc.get.rank_genes_groups_df(
            adata, group=cluster, key=f"{cluster_key}_rank_genes_groups"
        ).head(top_genes)
        genes = df["names"].tolist()
        query = ", ".join(genes)
        cell_type = annotate_gene_list(llm, query)
        mapping[cluster] = cell_type["cell_type"]

    adata.obs[f"{cluster_key}_annotated"] = adata.obs[cluster_key].map(mapping)
    return adata


def annotate_gene_list(llm, genes: list) -> str:
    """
    Annotate a list of genes with a description
    """
    system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)

    user_prompt = HumanMessagePromptTemplate.from_template(
        USER_PROMPT,
        input_variables=["marker"],
    )
    prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

    chain_one = (
        {"marker": lambda x: x["marker"]}
        | prompt
        | llm
        | {"cell_type": lambda x: x.content}
    )

    return chain_one.invoke({"marker": genes})


def get_quote() -> dict:
    """
    Get random quote

    Get randomly selected quote from database our programming quotes

    :return: selected quote
    :rtype: dict
    """

    quotes = [
        {
            "quote": "A long descriptive name is better than a short "
            "enigmatic name. A long descriptive name is better "
            "than a long descriptive comment.",
            "author": "Robert C. Martin",
        },
        {
            "quote": "You should name a variable using the same "
            "care with which you name a first-born child.",
            "author": "Robert C. Martin",
        },
        {
            "quote": "Any fool can write code that a computer "
            "can understand. Good programmers write code"
            " that humans can understand.",
            "author": "Martin Fowler",
        },
    ]

    return quotes[random.randint(0, len(quotes) - 1)]
