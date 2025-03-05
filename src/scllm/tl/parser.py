from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class CellType(BaseModel):
    cell_type: str = Field(description="The most likely cell type.")


CellTypeParser = PydanticOutputParser(pydantic_object=CellType)
