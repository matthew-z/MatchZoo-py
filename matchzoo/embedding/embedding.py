"""Matchzoo toolkit for token embedding."""

import typing

import numpy as np

import matchzoo as mz
from .embedding_helper import _GloVe, _FastText, _Word2Vec
from ..preprocessors.units.vocabulary import Vocabulary


class EmbeddingBase(object):
    """Base Class for embedding."""

    def build_matrix(self,
                     term_index: typing.Union[
                         dict, Vocabulary.TermIndex]):
        """Build embedding matrix for given tokens."""
        raise NotImplementedError


class EmebeddingV2(EmbeddingBase):
    """Embedding Class V2."""

    def __init__(self, data):
        """Init Embedding Class."""
        self._data = data
        self._output_dim = self._data.dim

    def build_matrix(
            self,
            term_index: typing.Union[
                dict, Vocabulary.TermIndex],
            lower_case_backup=False
    ) -> np.ndarray:
        """Build embedding matrix for given tokens."""
        return self._data.build_matrix(term_index).numpy()

    @property
    def term_index(self):
        return self._data.stoi
    @property
    def index_term(self):
        return self._data.itos


class GloVe(EmebeddingV2):
    """Glove Embedding."""

    def __init__(self, name: str = "840B", dim: int = 300):
        """Init for Glove Embedding."""
        data = _GloVe(name=name, dim=dim)
        super().__init__(data)


class FastText(EmebeddingV2):
    """FastText Embedding."""

    def __init__(self, language: str = "en"):
        """Init for FastText Embedding."""
        data = _FastText(language=language)
        super().__init__(data)


class Word2Vec(EmebeddingV2):
    """Word2Vec Embedding."""

    def __init__(self):
        """Init for Word2Vec Embedding."""
        data = _Word2Vec()
        super().__init__(data)


class Embedding(EmbeddingBase):
    """
    Embedding class.

    Examples::
        >>> import matchzoo as mz
        >>> train_raw = mz.datasets.toy.load_data()
        >>> pp = mz.preprocessors.NaivePreprocessor()
        >>> train = pp.fit_transform(train_raw, verbose=0)
        >>> vocab_unit = mz.build_vocab_unit(train, verbose=0)
        >>> term_index = vocab_unit.state['term_index']
        >>> embed_path = mz.datasets.embeddings.EMBED_RANK

    To load from a file:
        >>> embedding = mz.embedding.load_from_file(embed_path)
        >>> matrix = embedding.build_matrix(term_index)
        >>> matrix.shape[0] == len(term_index)
        True

    To build your own:
        >>> data = {'A':[0, 1], 'B':[2, 3]}
        >>> embedding = mz.Embedding(data, 2)
        >>> matrix = embedding.build_matrix({'A': 2, 'B': 1, '_PAD': 0})
        >>> matrix.shape == (3, 2)
        True

    """

    def __init__(self, data: dict, output_dim: int):
        """
        Embedding.

        :param data: Dictionary to use as term to vector mapping.
        :param output_dim: The dimension of embedding.
        """
        self._data = data
        self._output_dim = output_dim

    def build_matrix(
            self,
            term_index: typing.Union[
                dict, Vocabulary.TermIndex]
    ) -> np.ndarray:
        """
        Build a matrix using `term_index`.

        :param term_index: A `dict` or `TermIndex` to build with.
        :param initializer: A callable that returns a default value for missing
            terms in data. (default: a random uniform distribution in range)
            `(-0.2, 0.2)`).
        :return: A matrix.
        """
        input_dim = len(term_index)
        matrix = np.empty((input_dim, self._output_dim))

        valid_keys = self._data.keys()
        for term, index in term_index.items():
            if term in valid_keys:
                matrix[index] = self._data[term]
            else:
                matrix[index] = np.random.uniform(-0.2, 0.2,
                                                  size=self._output_dim)
        return matrix


def load_from_file(file_path: str, mode: str = 'word2vec') -> Embedding:
    """Load embedding from `file_path`.

    :param file_path: Path to file.
    :param mode: Embedding file format mode, one of 'word2vec', 'fasttext'
        or 'glove'.(default: 'word2vec')
    :return: An :class:`matchzoo.embedding.Embedding` instance.
    """
    embedding_data = {}
    output_dim = 0
    if mode == 'word2vec' or mode == 'fasttext':
        with open(file_path, 'r') as f:
            output_dim = int(f.readline().strip().split(' ')[-1])
            for line in f:
                current_line = line.rstrip().split(' ')
                embedding_data[current_line[0]] = current_line[1:]
    elif mode == 'glove':
        with open(file_path, 'r') as f:
            output_dim = len(f.readline().rstrip().split(' ')) - 1
            f.seek(0)
            for line in f:
                current_line = line.rstrip().split(' ')
                embedding_data[current_line[0]] = current_line[1:]
    else:
        raise TypeError(f"{mode} is not a supported embedding type."
                        f"`word2vec`, `fasttext` or `glove` expected.")
    return Embedding(embedding_data, output_dim)
