"""Naive Preprocessor."""

from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from tqdm import tqdm

from . import units
from .build_vocab_unit import build_vocab_unit
from .chain_transform import ChainTransform

tqdm.pandas()


class NaivePreprocessor(BasePreprocessor):
    """
    Naive preprocessor.

    Example:
        >>> import matchzoo as mz
        >>> train_data = mz.datasets.toy.load_data()
        >>> test_data = mz.datasets.toy.load_data(stage='test')
        >>> preprocessor = mz.preprocessors.NaivePreprocessor()
        >>> train_data_processed = preprocessor.fit_transform(train_data,
        ...                                                   verbose=0)
        >>> type(train_data_processed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>
        >>> test_data_transformed = preprocessor.transform(test_data,
        ...                                                verbose=0)
        >>> type(test_data_transformed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>

    """

    def fit(self, data_pack: DataPack, verbose: int = 1):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :param verbose: Verbosity.
        :return: class:`NaivePreprocessor` instance.
        """
        func = ChainTransform(self._default_units())
        data_pack = data_pack.apply_on_text(
            func,
            multiprocessing=self.multiprocessing,
            verbose=verbose)
        vocab_unit = build_vocab_unit(data_pack, verbose=verbose)
        self._context['vocab_unit'] = vocab_unit
        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create truncated length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()
        units_ = self._default_units()
        units_.append(self._context['vocab_unit'])
        units_.append(
            units.TruncatedLength(text_length=30, truncate_mode='post'))
        func = ChainTransform(units_)
        data_pack.apply_on_text(func, inplace=True,
                                multiprocessing=self.multiprocessing,
                                verbose=verbose)
        data_pack.append_text_length(inplace=True,
                                     verbose=verbose)
        data_pack.drop_empty(inplace=True)
        return data_pack
