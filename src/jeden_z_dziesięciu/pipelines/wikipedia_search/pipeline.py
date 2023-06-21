"""
This is a boilerplate pipeline 'wikipedia_search'
generated using Kedro 0.18.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
               func=wikipedia_test,
               inputs=["tokenizer", "binary_questions", "binary_answers"],
               outputs="wikipedia_results",
            ),
        ]
    )
