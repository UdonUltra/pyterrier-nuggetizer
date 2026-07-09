from jinja2 import Template


SCORER_PROMPT_STRING = Template(
    """Based on the query, label each of the {{nuggets|length}} nuggets either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a “good” answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

Search Query: {{ query }}
Nugget List: {{ nuggets }}

Only return the list of labels (List[str]). Do not explain.
Labels:"""
)

__all__ = ["SCORER_PROMPT_STRING"]
