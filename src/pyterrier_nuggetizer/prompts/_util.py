from jinja2 import Template
from typing import Dict, Any

def render_prompt(template: Template, context: Dict[str, Any]) -> str:
    """
    Render a Jinja2 template with the given context dictionary.

    Args:
        template_name: Filename (e.g. 'creator.txt') inside /prompts
        context: Dictionary of variables to pass into the template

    Returns:
        Rendered prompt string
    """
    return template.render(**context)

def make_callable_template(template: Template):
    """
    Create a callable function that renders a Jinja2 template with the given context.
    Args:
        template: Jinja2 Template object
    Returns:
        A callable function that takes keyword arguments and returns the rendered template
    """
    def template_call(**kwargs):
        return template.render(**kwargs)

    return template_call


__all__ = ["render_prompt", "make_callable_template"]
