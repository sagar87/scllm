from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


def construct_term_prompt(
    term: str = "cell type",
    extra: str = "",
    system_prompt: str = "You are an expert biologist with extensive knowledge in single cell RNA-seq analysis.",
    human_prompt: str = "Identify the most likely {term} given the following genes: {{genes}}.\n{extra}",
    format_instructions: bool = True,
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
