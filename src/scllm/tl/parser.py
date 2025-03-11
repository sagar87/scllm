from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, create_model


def _term_parser(term: str, features: str):

    model = create_model(
        "TermParser",
        term=(str, Field(description=f"Extract the most likely {term}.")),
        features=(
            list,
            Field(description=f"Extract the {features} associated with {term}."),
        ),
    )

    return PydanticOutputParser(pydantic_object=model)


def _multiple_term_parser(term, features, num_terms):
    new_model = create_model(
        "DynamicModel",
        term=(
            list[str],
            Field(description=f"Extract the {num_terms} most likely {term}."),
        ),
        features=(
            list[list[str]],
            Field(description=f"Extract the {features} associated with each {term}."),
        ),
    )

    return PydanticOutputParser(pydantic_object=new_model)


# Legacy code
def construct_term_parser(term: str):
    term_description = f"Extract the most likely {term}."
    genes_description = (
        f"Extract the genes that are most likely to be associated with {term}."
    )

    class Term(BaseModel):
        term: str = Field(description=term_description)
        genes: list[str] = Field(description=genes_description)

    return PydanticOutputParser(pydantic_object=Term)
