"""
These tests are simplied because the original verion takes too much time to
run, making CI fails as it reaches the time limit.
"""
from pathlib import Path

import pytest

import matchzoo as mz
from matchzoo import preprocessors


@pytest.fixture(scope='module',
                params=preprocessors.list_available())
def preprocessor_cls(request):
    return request.param


@pytest.fixture(scope='module', params=[
    mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=2)),
    mz.tasks.Classification(num_classes=2),
])
def task(request):
    return request.param


@pytest.fixture(scope='module')
def train_raw(task):
    return mz.datasets.toy.load_data('train', task)


def test_fit_transform(train_raw, preprocessor_cls):
    p1 = preprocessor_cls(multiprocessing=False)
    p2 = preprocessor_cls(multiprocessing=True)

    processed = p1.fit_transform(train_raw.copy())
    assert processed

    processed_parallel = p2.fit_transform(train_raw.copy())
    assert processed_parallel

    assert all(processed.left == processed_parallel.left)
    assert all(processed.right == processed_parallel.right)
    assert all(processed.relation == processed_parallel.relation)
