from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

single_cell_analyst = SystemMessagePromptTemplate.from_template(
    "You are an expert biologist with extensive knowledge in single cell RNA-seq analysis."
)

identify_cell_type = HumanMessagePromptTemplate.from_template(
    "Identify the most likely cell type given the following genes: {genes}. {format_instructions}"
)

CellAnnotationPrompt = ChatPromptTemplate.from_messages(
    [single_cell_analyst, identify_cell_type]
)


identify_factor = HumanMessagePromptTemplate.from_template(
    "Identify the most likely cell type given the following genes: {genes}. {format_instructions}"
)

FactorAnnotationPrompt = ChatPromptTemplate.from_messages(
    [single_cell_analyst, identify_factor]
)
