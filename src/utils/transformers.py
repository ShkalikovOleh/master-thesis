"""This module contains utilities for Transformers package"""

import gc
from contextlib import contextmanager
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer, pipeline


@contextmanager
def use_hf_pipeline(
    pipeline_name: str,
    model_name: str,
    model_kwargs: dict[str, Any] = {},
    tokenizer_kwargs: dict[str, Any] = {},
    pipeline_kwargs: dict[str, Any] = {},
    AutoModelClass: Any | None = None,
):
    """Almost every XLNER pipeline use several models during run.
    Since models occupy limited GPU memory we need release all memory
    used by model after the transformation step completion.
    This function wraps creation and deletion of any HF pipeline.
    After leaving with scope all underlying for model memory will be
    cleaned.

    Args:
        pipeline_name (str): name of the pipeline
        model_name (str): name of the model on the HF Hub of path
        model_kwargs (dict[str, Any]): params for model
        tokenizer_kwargs (dict[str, Any]): params for tokenizer
        pipeline_kwargs (dict[str, Any]): params for pipeline

    Yields:
        Desired initialized pipeline with specified model and tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if AutoModelClass is not None:
        model = AutoModelClass.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModel.from_pretrained(model_name, **model_kwargs)
    pipe = pipeline(pipeline_name, model=model, tokenizer=tokenizer, **pipeline_kwargs)

    try:
        yield pipe
    finally:
        del pipe
        del tokenizer
        del model

        gc.collect()
        torch.cuda.empty_cache()
