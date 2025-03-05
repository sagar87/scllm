from langchain_core.runnables.base import RunnableEach, RunnableLambda, RunnableParallel

from .parser import CellTypeParser
from .prompts import CellAnnotationPrompt


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
