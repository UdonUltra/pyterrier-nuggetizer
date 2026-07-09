from jinja2 import Template

ASSIGNER_GRADE_2_PROMPT_STRING = Template(
    """Based on the query and passage, label each of the {{ nuggets|length }} nuggets either as support or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support; otherwise, label them as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

Search Query: {{ query }}
Passage: {{ context }}
Nugget List: {{ nuggets }}

Only return the list of labels (List[str]). Do not explain.
Labels:"""
)

ASSIGNER_GRADE_3_PROMPT_STRING = Template(
    """Based on the query and passage, label each of the {{ nuggets|length }} nuggets either as support, partial_support, or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support. A nugget that is partially captured in the passage should be labeled as partial_support. If the nugget is not captured at all, label it as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

Search Query: {{ query }}
Passage: {{ context }}
Nugget List: {{ nuggets }}

Only return the list of labels (List[str]). Do not explain.
Labels:"""
)


__all__ = ["ASSIGNER_GRADE_2_PROMPT_STRING", "ASSIGNER_GRADE_3_PROMPT_STRING"]
