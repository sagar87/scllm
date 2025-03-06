from functools import partial

from langchain_core.runnables.base import RunnableEach, RunnableLambda, RunnableParallel

from .parser import construct_term_parser, construct_term_comparison_parser
from .prompts import construct_term_prompt, construct_term_comparison_prompt


def construct_passthrough(passthrough: list[str]):
    def passthrough_fn(dict_in: dict, key: str):
        return dict_in[key]

    return {
        var: RunnableLambda(partial(passthrough_fn, key=var)) for var in passthrough
    }


def construct_term_chain(llm, term: str, extra: str = "", passthrough: list[str] = []):
    prompt = construct_term_prompt(term=term, extra=extra)
    parser = construct_term_parser(term=term)
    passes = construct_passthrough(passthrough)
    format_instructions = parser.get_format_instructions()

    return RunnableEach(
        bound=RunnableParallel(
            {
                # "cluster": RunnableLambda(lambda x: x["cluster"]),
                **passes,
                "target": RunnableLambda(
                    lambda x: prompt.format_prompt(
                        genes=x["genes"], format_instructions=format_instructions
                    )
                )
                | llm
                | parser
                | RunnableLambda(lambda x: x.term),
            }
        )
    )


def construct_term_comparison_chain(
    llm, term: str, extra: str = "", passthrough: list[str] = []
):
    """
    Constructs a chain for comparing two term descriptions using a language model.

    This function sets up a chain where a language model evaluates whether two
    given descriptions of a term are the same. It uses a prompt template and an
    output parser to format the input and interpret the output, respectively.
    Additional data can be passed through the chain without modification.

    Parameters
    ----------
    llm : BaseLanguageModel
        The language model to use for term comparison.
    term : str
        The type of term to compare (e.g., "cell type").
    extra : str, optional
        Extra information to include in the prompt.
    passthrough : list[str], optional
        List of keys whose values should be passed through unchanged.

    Returns
    -------
    RunnableEach
        A chained runnable that performs the term comparison.
    """

    prompt = construct_term_comparison_prompt(term=term, extra=extra)
    parser = construct_term_comparison_parser(term=term)
    passes = construct_passthrough(passthrough)
    format_instructions = parser.get_format_instructions()

    return RunnableEach(
        bound=RunnableParallel(
            {
                # "cluster": RunnableLambda(lambda x: x["cluster"]),
                **passes,
                "target": RunnableLambda(
                    lambda x: prompt.format_prompt(
                        entity1=x["entity1"],
                        entity2=x["entity2"],
                        format_instructions=format_instructions,
                    )
                )
                | llm
                | parser
                | RunnableLambda(lambda x: x.same),
            }
        )
    )
