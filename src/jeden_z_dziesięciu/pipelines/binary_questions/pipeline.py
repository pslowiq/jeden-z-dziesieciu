from kedro.pipeline import Pipeline, node, pipeline
from functools import partial
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_model,
                inputs=["params:model_name"],
                outputs="model",
            ),
            node(
                func=train,
                inputs=["params:model_params", "model", "tokenizer", "params:filepath", "test_binary_questions", "test_binary_answers"],
                outputs="trained_model",
            ),
        ]
    )
