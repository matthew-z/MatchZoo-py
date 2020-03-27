"""Forked from https://github.com/pytorch/text.

BSD 3-Clause License

Copyright (c) James Bradbury and Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import gzip
import logging
import os
import tarfile
import typing
import zipfile

import six
import torch
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
import matchzoo as mz
from ..preprocessors.units.vocabulary import Vocabulary

logger = logging.getLogger(__name__)
UNK = "UNK####UNK####UNK"


def _infer_shape(f):
    num_lines, vector_dim = 0, None
    for line in f:
        if vector_dim is None:
            row = line.rstrip().split(b" ")
            vector = row[1:]
            # Assuming word, [vector] format
            if len(vector) > 2:
                # The header present in some (w2v) formats contains two elements.
                vector_dim = len(vector)
                num_lines += 1  # First element read
        else:
            num_lines += 1
    f.seek(0)
    return num_lines, vector_dim


def reporthook(t):
    """https://github.com/tqdm/tqdm."""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


class Vectors(object):
    """Vector-based Embedding Base Class."""

    def __init__(self, name, cache=None,
                 url=None, unk_init=None, max_vectors=None):
        """Init Vectors.

        Arguments:
           name: name of the file that contains the vectors
           cache: directory for cached vectors
           url: url for download if vectors not found in cache
           unk_init (callback): by default, initialize out-of-vocabulary word vectors
               to zero vectors; can be any function that takes in a Tensor and
               returns a Tensor of the same size
           max_vectors (int): this can be used to limit the number of
               pre-trained vectors loaded.
               Most pre-trained vector sets are sorted
               in the descending order of word frequency.
               Thus, in situations where the entire set doesn't fit in memory,
               or is not needed for another reason, passing `max_vectors`
               can limit the size of the loaded set.
        """
        cache = cache or mz.USER_DATA_DIR.joinpath(name)
        self.itos = None
        self.stoi = None
        self.vectors = None
        self.dim = None
        self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init
        self.unk_idx = None
        self._cache(name, cache, url=url, max_vectors=max_vectors)
        self._add_unk()

    def _add_unk(self):
        if UNK in self.stoi:
            self.unk_idx = self.stoi[UNK]
            return
        self.unk_idx = len(self.stoi)
        self.stoi[UNK] = self.unk_idx
        self.itos.append(UNK)
        unk_vec = self.unk_init(torch.Tensor(self.dim)).unsqueeze(0)
        self.vectors = torch.cat([self.vectors, unk_vec], dim=0)

    def __getitem__(self, token):
        """Return the embedding vector for a given token.

        The embedding of unknown will be returned when the token
        is not in the vocabulary
        """
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.vectors[self.unk_idx]

    def _cache(self, name, cache, url=None, max_vectors=None):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        if os.path.isfile(name):
            path = name
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = os.path.join(cache, os.path.basename(name)) + file_suffix
        else:
            path = os.path.join(cache, name)
            if max_vectors:
                file_suffix = '_{}.pt'.format(max_vectors)
            else:
                file_suffix = '.pt'
            path_pt = path + file_suffix

        if not os.path.isfile(path_pt):
            if not os.path.isfile(path) and url:
                logger.info('Downloading vectors from {}'.format(url))
                if not os.path.exists(cache):
                    os.makedirs(cache)
                dest = os.path.join(cache, os.path.basename(url))
                if not os.path.isfile(dest):
                    with tqdm(unit='B', unit_scale=True, miniters=1,
                              desc=dest) as t:
                        try:
                            urlretrieve(url, dest, reporthook=reporthook(t))
                        except KeyboardInterrupt as e:  # remove the partial zip file
                            os.remove(dest)
                            raise e
                logger.info('Extracting vectors into {}'.format(cache))
                ext = os.path.splitext(dest)[1][1:]
                if ext == 'zip':
                    with zipfile.ZipFile(dest, "r") as zf:
                        zf.extractall(cache)
                elif ext == 'gz':
                    if dest.endswith('.tar.gz'):
                        with tarfile.open(dest, 'r:gz') as tar:
                            tar.extractall(path=cache)
            if not os.path.isfile(path):
                raise RuntimeError('no vectors found at {}'.format(path))

            logger.info("Loading vectors from {}".format(path))
            ext = os.path.splitext(path)[1][1:]
            if ext == 'gz':
                open_file = gzip.open
            else:
                open_file = open

            vectors_loaded = 0
            with open_file(path, 'rb') as f:
                num_lines, dim = _infer_shape(f)
                if not max_vectors or max_vectors > num_lines:
                    max_vectors = num_lines

                itos, vectors, dim = [], torch.zeros((max_vectors, dim)), None

                for line in tqdm(f, total=max_vectors,
                                 desc="Reading raw txt"):
                    # Explicitly splitting on " " is important, so we don't
                    # get rid of Unicode non-breaking spaces in the vectors.
                    entries = line.rstrip().split(b" ")

                    word, entries = entries[0], entries[1:]
                    if dim is None and len(entries) > 1:
                        dim = len(entries)
                    elif len(entries) == 1:
                        logger.warning(
                            "Skipping token {} with 1-dimensional "
                            "vector {}; likely a header".format(word, entries))
                        continue
                    elif dim != len(entries):
                        raise RuntimeError(
                            "Vector for token {} has {} dimensions, but previously "
                            "read vectors have {} dimensions. All vectors must have "
                            "the same number of dimensions.".format(word, len(
                                entries), dim))

                    try:
                        if isinstance(word, six.binary_type):
                            word = word.decode('utf-8')
                    except UnicodeDecodeError:
                        logger.info(
                            "Skipping non-UTF8 token {}".format(repr(word)))
                        continue

                    vectors[vectors_loaded] = torch.tensor(
                        [float(x) for x in entries])
                    vectors_loaded += 1
                    itos.append(word)

                    if vectors_loaded == max_vectors:
                        break

            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            logger.info('Saving vectors to {}'.format(path_pt))
            if not os.path.exists(cache):
                os.makedirs(cache)
            torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
        else:
            logger.info('Loading vectors from {}'.format(path_pt))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)

    def __len__(self):
        """Return the number of tokens."""
        return len(self.vectors)

    def build_matrix(
            self,
            term_index: typing.Union[
                dict, Vocabulary.TermIndex],
            lower_case_backup=False
    ) -> torch.Tensor:
        """Build embedding matrix for given tokens."""
        tokens = term_index.keys()
        return self._get_vecs_by_tokens(tokens, lower_case_backup)

    def _get_vecs_by_tokens(self, tokens, lower_case_backup=False):
        """Look up embedding vectors of tokens.

        Arguments:
            tokens: a token or a list of tokens. if `tokens` is a string,
                returns a 1-D tensor of shape `self.dim`; if `tokens` is a
                list of strings, returns a 2-D tensor of shape=(len(tokens),
                self.dim).
            lower_case_backup : Whether to look up the token in the lower case.
                If False, each token in the original case will be looked up;
                if True, each token in the original case will be looked up first,
                if not found in the keys of the property `stoi`, the token in the
                lower case will be looked up. Default: False.
        """
        to_reduce = False

        if isinstance(tokens, str):
            tokens = [tokens]
            to_reduce = True

        token_ids = []
        for token in tokens:
            if token in self.stoi:
                token_id = self.stoi[token]
            elif lower_case_backup and token.lower() in self.stoi:
                token_id = self.stoi[token.lower()]
            else:
                token_id = self.unk_idx
            token_ids.append(token_id)

        vecs = self.vectors[token_ids]
        return vecs[0] if to_reduce else vecs


class _GloVe(Vectors):
    url = {
        '42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        '840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        '6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
    }

    def __init__(self, name='840B', dim=300, **kwargs):
        url = self.url[name]
        name = 'glove.{}.{}d.txt'.format(name, str(dim))
        cache = mz.USER_DATA_DIR.joinpath("glove")

        super(_GloVe, self).__init__(name, url=url, cache=cache, **kwargs)


class _FastText(Vectors):
    url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'

    def __init__(self, language="en", **kwargs):
        url = self.url_base.format(language)
        name = os.path.basename(url)
        cache = mz.USER_DATA_DIR.joinpath("fasttext")
        super(_FastText, self).__init__(name, url=url, cache=cache, **kwargs)


class _Word2Vec(Vectors):
    def __init__(self, **kwargs):
        url = 'https://allennlp.s3.amazonaws.com/datasets/word2vec' \
              '/GoogleNews-vectors-negative300.txt.gz '
        name = os.path.basename(url)
        cache = mz.USER_DATA_DIR.joinpath("word2vec")
        super(_Word2Vec, self).__init__(name, url=url, cache=cache, **kwargs)
