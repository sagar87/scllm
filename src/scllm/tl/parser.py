from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


def construct_term_parser(term: str):
    description = f"Extract the most likely {term}."

    class Term(BaseModel):
        term: str = Field(description=description)

    return PydanticOutputParser(pydantic_object=Term)
