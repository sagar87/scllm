from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class CellType(BaseModel):
    cell_type: str = Field(description="Extract the most likely cell type.")


CellTypeParser = PydanticOutputParser(pydantic_object=CellType)


def construct_term_parser(term: str):
    description = f"Extract the most likely {term}."

    class Term(BaseModel):
        term: str = Field(description=description)

    return PydanticOutputParser(pydantic_object=Term)
