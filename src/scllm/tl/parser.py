from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


def construct_term_parser(term: str):
    description = f"Extract the most likely {term}."

    class Term(BaseModel):
        term: str = Field(description=description)

    return PydanticOutputParser(pydantic_object=Term)



def construct_term_comparison_parser(term: str):
    description = f"Are the two {term}s the same."

    class Same(BaseModel):
        same: bool = Field(description=description)

    return PydanticOutputParser(pydantic_object=Same)


