"""Naive Preprocessor."""

from tqdm import tqdm

from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from . import units
from .chain_transform import ChainTransform

tqdm.pandas()


class NumeralizePreprocessor(BasePreprocessor):
    """
    A simple preprocessor that only for tokenization and truncation.
    Its vocab is built from pre-trained embeddings instead of training data.
    """

    def __init__(self,
                 truncated_mode: str = 'pre',
                 truncated_length_left: int = None,
                 truncated_length_right: int = None,
                 remove_stop_words: bool = False,
                 lowercase: bool = False,
                 multiprocessing: bool = False,
                 vocab_max_size=None,
                 terms=None):
        super().__init__(multiprocessing)

        self._units = [units.Tokenize()]

        if remove_stop_words:
            self._units.append(units.stop_removal.StopRemoval())

        if lowercase:
            self._units.append(units.Lowercase())

        self._context['vocab_unit'] = units.Vocabulary()

        self._truncated_mode = truncated_mode
        self._truncated_length_left = truncated_length_left
        self._truncated_length_right = truncated_length_right

        if self._truncated_length_left:
            self._left_truncated_length_unit = units.TruncatedLength(
                self._truncated_length_left, self._truncated_mode
            )
        if self._truncated_length_right:
            self._right_truncated_length_unit = units.TruncatedLength(
                self._truncated_length_right, self._truncated_mode
            )

        if terms:
            self._build_vocab_from_embedding(terms, vocab_max_size)


    def _build_vocab_from_embedding(self, terms, max_size: int = None):

        if max_size:
            self._context['vocab_unit'].fit(list(terms)[:max_size])
        else:
            self._context['vocab_unit'].fit(terms)

        vocab_size = len(self._context['vocab_unit'].state['term_index'])
        self._context['vocab_size'] = vocab_size
        self._context['embedding_input_dim'] = vocab_size


    def fit(self, data_pack: DataPack, verbose: int = 1):
        """
        This preprocessor does not need fitting.
        """
        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create truncated length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()

        func = ChainTransform(self._units)

        data_pack.apply_on_text(func, inplace=True,
                                multiprocessing=self.multiprocessing,
                                verbose=verbose)

        if self._truncated_length_left:
            data_pack.apply_on_text(
                ChainTransform(self._left_truncated_length_unit),
                mode='left', inplace=True,
                verbose=verbose)

        if self._truncated_length_right:
            data_pack.apply_on_text(
                ChainTransform(self._right_truncated_length_unit),
                mode='right', inplace=True,
                verbose=verbose)

        data_pack.apply_on_text(ChainTransform(self._context['vocab_unit']),
                                mode='both', inplace=True,
                                verbose=verbose)

        data_pack.append_text_length(
            inplace=True,
            verbose=verbose)

        data_pack.drop_empty(inplace=True)

        return data_pack
