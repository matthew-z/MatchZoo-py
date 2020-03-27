"""Bert Preprocessor."""

from functools import partial

from transformers import BertTokenizer

from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from matchzoo.preprocessors import units
from .chain_transform import ChainTransform


class BertPreprocessor(BasePreprocessor):
    """
    Basic preprocessor helper.

    :param mode: String, supported mode can be referred
        https://huggingface.co/pytorch-transformers/pretrained_models.html.

    """

    def __init__(self, mode: str = 'bert-base-uncased',
                 multiprocessing: bool = True,
                 truncated_mode: str = 'pre',
                 truncated_length_left: int = 20,
                 truncated_length_right: int = 492):

        """Initialization."""
        super().__init__(multiprocessing)
        self._tokenizer = BertTokenizer.from_pretrained(mode)
        self._truncated_length_left = truncated_length_left
        self._truncated_length_right = truncated_length_right
        self._truncated_mode = truncated_mode
        if self._truncated_length_left:
            self._left_truncated_length_unit = units.TruncatedLength(
                self._truncated_length_left, self._truncated_mode
            )
        if self._truncated_length_right:
            self._right_truncated_length_unit = units.TruncatedLength(
                self._truncated_length_right, self._truncated_mode
            )

    def fit(self, data_pack: DataPack, verbose: int = 1):
        """Tokenizer is all BertPreprocessor's need."""
        return self

    def bert_encode(self, text):
        encode = partial(self._tokenizer.encode,
                         add_special_tokens=False,
                         max_length=max(self._truncated_length_right,
                                        self._truncated_length_left))
        encode.__name__ = "BertEncode"
        return encode(text[:10000])

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        data_pack.apply_on_text(self.bert_encode,
                                mode='both', inplace=True,
                                multiprocessing=self.multiprocessing,
                                verbose=verbose)

        if self._truncated_length_left:
            data_pack.apply_on_text(ChainTransform(self._left_truncated_length_unit),
                                    mode='left', inplace=True,
                                    verbose=verbose)
        if self._truncated_length_right:
            data_pack.apply_on_text(ChainTransform(self._right_truncated_length_unit),
                                    mode='right', inplace=True,
                                    verbose=verbose)

        data_pack.append_text_length(inplace=True, verbose=verbose,
                                     multiprocessing=self.multiprocessing)
        data_pack.drop_empty(inplace=True)
        return data_pack
