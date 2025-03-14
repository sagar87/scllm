from collections.abc import MutableMapping
from functools import partial

from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableEach, RunnableLambda, RunnableParallel

from .parser import construct_term_parser
from .prompts import construct_term_prompt


def flatten(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def prune(dictionary):
    return {k.split("_")[-1]: v for k, v in dictionary.items()}


def _term_chain(llm, prompt, parser):
    """
    Extracts information from the data field and processes
    """
    return RunnableEach(
        bound=RunnableParallel(
            {
                "pass": RunnablePassthrough(),
                "target": RunnableLambda(
                    lambda x: prompt.format_prompt(
                        data=x["data"],
                        format_instructions=parser.get_format_instructions(),
                    )
                )
                | llm
                | parser
                | RunnableLambda(lambda x: x.model_dump()),
            }
        )
        | RunnableLambda(flatten)
        | RunnableParallel(
            {
                "pass": RunnablePassthrough(),
                "union": RunnableLambda(
                    lambda x: list(set(x["target_features"]) & set(x["pass_data"]))
                ),
            }
        )
        | RunnableLambda(flatten)
        | RunnableLambda(prune)
    )


def _terms_chain(llm, prompt, parser):
    """
    Extracts information from the data field and processes
    """
    return RunnableEach(
        bound=RunnableParallel(
            {
                "pass": RunnablePassthrough(),
                "target": RunnableLambda(
                    lambda x: prompt.format_prompt(
                        data=x["data"],
                        format_instructions=parser.get_format_instructions(),
                    )
                )
                | llm
                | parser
                | RunnableLambda(lambda x: x.model_dump()),
            }
        )
        | RunnableLambda(flatten)
        | RunnableLambda(prune)
    )


def _description_chain(llm, prompt, parser):
    """
    Extracts information from the data field and processes
    """
    return RunnableEach(
        bound=RunnableParallel(
            {
                "pass": RunnablePassthrough(),
                "target": RunnableLambda(
                    lambda x: prompt.format_prompt(
                        data=x["data"],
                    )
                )
                | llm
                | parser,
            }
        )
        | RunnableLambda(flatten)
        | RunnableLambda(prune)
    )


# Legacy code
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


def construct_term_chain_with_genes(llm, term: str, extra: str = ""):
    prompt = construct_term_prompt(term=term, extra=extra)
    parser = construct_term_parser(term=term)
    # passes = construct_passthrough(passthrough)
    format_instructions = parser.get_format_instructions()

    return RunnableEach(
        bound=RunnableParallel(
            {
                "pass": RunnablePassthrough(),
                "target": RunnableLambda(
                    lambda x: prompt.format_prompt(
                        genes=x["genes"], format_instructions=format_instructions
                    )
                )
                | llm
                | parser
                | RunnableLambda(lambda x: x.model_dump()),
            }
        )
        | RunnableLambda(flatten)
        | RunnableParallel(
            {
                "pass": RunnablePassthrough(),
                "union": RunnableLambda(
                    lambda x: list(set(x["target_genes"]) & set(x["pass_genes"]))
                ),
            }
        )
        | RunnableLambda(flatten)
        | RunnableLambda(prune)
    )
