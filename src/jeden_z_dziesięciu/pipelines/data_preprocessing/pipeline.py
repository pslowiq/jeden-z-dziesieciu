"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=split_sentences,
                inputs=["train_questions_file", "train_answers_file"],
                outputs=["train_questions", "train_answers", "train_merged_file"],
            ),
            node(
                func=split_sentences,
                inputs=["test_questions_file", "test_answers_file"],
                outputs=["test_questions", "test_answers", "test_merged_file"],
            ),
            
            node(
                func=filter_sentences_with_binary_answer,
                inputs=["train_questions", "train_answers"],
                outputs=["train_binary_questions", "train_binary_answers"],
            ),
            node(
                func=filter_sentences_with_binary_answer,
                inputs=["test_questions", "test_answers"],
                outputs=["test_binary_questions", "test_binary_answers"],
            ),
            node(
                func=make_tokenizer,
                inputs=["params:model_name"],
                outputs="tokenizer",
            )])
