import pandas as pd
import scanpy as sc
from langchain_core.language_models import BaseLanguageModel

from .chains import CellTypeAnnotationChain


def annotate_factor(
    adata: sc.AnnData,
    varm_key: str,
    llm: BaseLanguageModel,
    num_samples: int = 1,
    key_added: str = "scllm_annotation",
    top_genes: int = 10,
    **kwargs,
) -> sc.AnnData:
    pass
