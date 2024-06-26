"""This module implements a translation step of the pipeline"""

from transformers import AutoModelForSeq2SeqLM
from datasets import Dataset

from src.utils.transformers import use_hf_pipeline


class TranslationTransform:
    """Translation transfrom which use the given HF model to translate
    text from src_lang to tgt_lang"""

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        batch_size: int,
        in_column: str,
        out_column: str,
        model_path: str,
        device: int = 0,
        src_lang_code: str | None = None,
        tgt_lang_code: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.in_column = in_column
        self.out_column = out_column

        self.batch_size = batch_size
        self.device = device

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._src_code = src_lang_code if src_lang_code else src_lang
        self._tgt_code = tgt_lang_code if tgt_lang_code else tgt_lang

    def __call__(self, ds: Dataset) -> Dataset:
        task = f"translation_{self.src_lang}_to_{self.tgt_lang}"
        with use_hf_pipeline(
            task,
            self.model_path,
            pipeline_kwargs={
                "device": self.device,
                "batch_size": self.batch_size,
            },
            AutoModelClass=AutoModelForSeq2SeqLM,
        ) as pipe:

            def translate_impl(sentences: list[str]) -> list[str]:
                translation_iter = pipe(
                    sentences, src_lang=self._src_code, tgt_lang=self._tgt_code
                )
                translations = [out["translation_text"] for out in translation_iter]
                return {self.out_column: translations}

            ds = ds.map(
                translate_impl,
                input_columns=self.in_column,
                batched=True,
                batch_size=self.batch_size,
            )

        return ds
