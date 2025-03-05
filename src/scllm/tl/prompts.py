from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from typing import Optional

single_cell_analyst = SystemMessagePromptTemplate.from_template(
    "You are an expert biologist with extensive knowledge in single cell RNA-seq analysis."
)

identify_cell_type = HumanMessagePromptTemplate.from_template(
    "Identify the most likely cell type given the following genes: {genes}. {format_instructions}"
)

CellAnnotationPrompt = ChatPromptTemplate.from_messages(
    [single_cell_analyst, identify_cell_type]
)


identify_factor_cell_type = HumanMessagePromptTemplate.from_template(
    "Identify the most likely cell type given the following genes: {genes}. {format_instructions}"
)

FactorAnnotationPrompt = ChatPromptTemplate.from_messages(
    [single_cell_analyst, identify_factor_cell_type]
)


identify_factor_process = HumanMessagePromptTemplate.from_template(
    "Identify the most likely biological process given the following genes: {genes}. {format_instructions}"
)

FactorProcessAnnotationPrompt = ChatPromptTemplate.from_messages(
    [single_cell_analyst, identify_factor_process]
)


def identify_term_from_genes(
    term: str = "cell type",
    extra: Optional[str] = None,
    system_prompt: str = "You are an expert biologist with extensive knowledge in single cell RNA-seq analysis.",
    human_prompt: str = "Identify the most likely {term} given the following genes: {genes}.\n{extra}",
    format_instructions: bool = True
    ) -> ChatPromptTemplate:
    
    human_prompt = human_prompt.format(term=term, extra=extra)
    
    if format_instructions: 
        human_prompt = human_prompt + "\n{format_instructions}"
    
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_prompt),
        ]
    )
