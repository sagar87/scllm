from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables.base import RunnableEach, RunnableLambda, RunnableParallel
from langchain_core.runnables.branch import RunnableBranch

from .parser import CellTypeParser
from .prompts import CellAnnotationPrompt


def SampleChain(chain: RunnableLambda, num_samples: int):
    # runs a chain num_samples times
    inits = [i for i in range(num_samples)]

    return RunnableEach(
        bound=RunnableParallel(
            {
                "init": RunnableLambda(lambda x: x),
                "result": chain,
            }
        )
    )


def CellTypeAnnotationChain(llm):

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


# recevies a list of genes and returns a list of cell types
# def CellTypeAnnotationChain(llm):
#     return RunnableLambda(lambda x: CellAnnotationPrompt.format_prompt(
#         genes=", ".join(x), format_instructions=CellTypeParser.get_format_instructions()
#     )) | llm | CellTypeParser


# cell_type_branch = (
#     RunnableLambda(
#         lambda x: _analyze_genelist(x, output_parser.get_format_instructions())
#     )
#     | llm
#     | output_parser
# )
#     chain = RunnableLambda(lambda x: x * num_samples) | RunnableEach(
#         bound=cell_type_branch
#     )
#     res = chain.invoke(list(cluster_genes.values()))
