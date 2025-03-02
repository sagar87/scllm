SYSTEM_PROMPT = "You are an AI assistant that helps biologists with annotating cells."

USER_PROMPT = """
Identify cell type using the following markers.
Only provide the cell type name. Do not show numbers before the name.
Some can be a mixture of multiple cell types.

{marker}
"""
