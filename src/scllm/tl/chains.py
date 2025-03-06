from functools import partial

from langchain_core.runnables.base import RunnableEach, RunnableLambda, RunnableParallel

from .parser import construct_term_parser
from .prompts import construct_term_prompt


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
