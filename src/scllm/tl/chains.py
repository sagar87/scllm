from langchain_core.runnables.base import RunnableEach, RunnableLambda, RunnableParallel

from .parser import CellTypeParser
from .prompts import CellAnnotationPrompt, FactorAnnotationPrompt


def CellTypeAnnotationChain(llm):
    """Create a chain for annotating cell types using marker genes.

    This function creates a LangChain pipeline that processes marker genes for each cluster
    and returns predicted cell types. The chain handles the entire workflow from gene list
    processing to cell type prediction.

    Parameters
    ----------
    llm : BaseLanguageModel
        Language model instance to use for cell type prediction

    Returns
    -------
    RunnableEach
        A LangChain runnable that processes each cluster's marker genes and returns
        a dictionary containing:
        - cluster: The cluster identifier
        - cell_type: The predicted cell type

    Examples
    --------
    >>> from langchain_openai import ChatOpenAI
    >>> llm = ChatOpenAI()
    >>> chain = CellTypeAnnotationChain(llm)
    >>> result = chain.invoke([
    ...     {"cluster": "0", "genes": ["CD4", "CD8A", "IL2RA"]},
    ...     {"cluster": "1", "genes": ["MS4A1", "CD79A", "CD79B"]}
    ... ])
    """
    return RunnableEach(
        bound=RunnableParallel(
            {
                "cluster": RunnableLambda(lambda x: x["cluster"]),
                "cell_type": (
                    RunnableLambda(lambda x: x["genes"])
                    | RunnableLambda(
                        lambda x: CellAnnotationPrompt.format_prompt(
                            genes=", ".join(x),
                            format_instructions=CellTypeParser.get_format_instructions(),
                        )
                    )
                    | llm
                    | CellTypeParser
                    | RunnableLambda(lambda x: x.cell_type)
                ),
            }
        )
    )


def FactorAnnotationChain(llm):
    """Create a chain for annotating cell types using marker genes."""
    return RunnableEach(
        bound=RunnableParallel(
            {
                "factor": RunnableLambda(lambda x: x["factor"]),
                "sign": RunnableLambda(lambda x: x["sign"]),
                "cell_type": (
                    RunnableLambda(lambda x: x["genes"])
                    | RunnableLambda(
                        lambda x: FactorAnnotationPrompt.format_prompt(
                            genes=", ".join(x),
                            format_instructions=CellTypeParser.get_format_instructions(),
                        )
                    )
                    | llm
                    | CellTypeParser
                    | RunnableLambda(lambda x: x.cell_type)
                ),
            }
        )
    )
