"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.18.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import test_performance_fs, test_performance_os
from functools import partial


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=partial(test_performance_fs, trained = True),
                inputs=[
                    "trained_model",
                    "tokenizer",
                    "test_binary_questions",
                    "test_binary_answers",
                ],
                outputs="finetuned_accuracy_fs",
            ),
            node(
                func=partial(test_performance_os, trained = True),
                inputs=[
                    "trained_model",
                    "tokenizer",
                    "test_binary_questions",
                    "test_binary_answers",
                ],
                outputs="finetuned_accuracy_os",
            )])
