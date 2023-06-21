"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = pipelines['data_preprocessing'] + pipelines['binary_questions']
    pipelines["evaluation"] = pipelines['data_preprocessing'] + pipelines['model_evaluation']
    #pipelines["wiki_search"] = sum(pipelines.values())
    return pipelines
