from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# def construct_term_parser(term: str):
#     description = f"Extract the most likely {term}."

#     class Term(BaseModel):
#         term: str = Field(description=description)

#     return PydanticOutputParser(pydantic_object=Term)


def construct_term_parser(term: str):
    term_description = f"Extract the most likely {term}."
    genes_description = (
        f"Extract the genes that are most likely to be associated with {term}."
    )

    class Term(BaseModel):
        term: str = Field(description=term_description)
        genes: list[str] = Field(description=genes_description)

    return PydanticOutputParser(pydantic_object=Term)
