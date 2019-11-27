"""Bert Preprocessor."""

from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from pytorch_transformers import BertTokenizer


class BertPreprocessor(BasePreprocessor):
    """
    Baisc preprocessor helper.

    :param mode: String, supported mode can be referred
        https://huggingface.co/pytorch-transformers/pretrained_models.html.

    """

    def __init__(self, mode: str = 'bert-base-uncased',
                 multiprocessing: bool = False):
        """Initialization."""
        super().__init__(multiprocessing)
        self._tokenizer = BertTokenizer.from_pretrained(mode)

    def fit(self, data_pack: DataPack, verbose: int = 1):
        """Tokenizer is all BertPreprocessor's need."""
        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        data_pack.apply_on_text(self._tokenizer.encode,
                                mode='both', inplace=True,
                                multiprocessing=self.multiprocessing,
                                verbose=verbose)
        data_pack.append_text_length(inplace=True, verbose=verbose,
                                     multiprocessing=self.multiprocessing)
        data_pack.drop_empty(inplace=True)
        return data_pack
