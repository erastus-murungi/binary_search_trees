from random import randint
from sys import maxsize
from typing import Type

import pytest

from avl import AVLTreeIterative
from bst import BinarySearchTree, BinarySearchTreeIterative, Leaf


@pytest.mark.parametrize(
    "bst_class", [BinarySearchTreeIterative, BinarySearchTree, AVLTreeIterative]
)
def test_insertion(bst_class: Type[BinarySearchTree]):
    for _ in range(50):
        values = frozenset([randint(-10000, 10000) for _ in range(100)])
        bst = bst_class()
        for val in values:
            bst.insert(val, None)
            assert val in bst
            bst.check_invariants(-maxsize, maxsize)

        for val in values:
            del bst[val]
            assert val not in bst
            bst.check_invariants(-maxsize, maxsize)

        assert not bst
        assert len(bst) == 0
        assert isinstance(bst.root, Leaf)


@pytest.mark.parametrize(
    "bst_class", [BinarySearchTreeIterative, BinarySearchTree, AVLTreeIterative]
)
def test_sorted(bst_class: Type[BinarySearchTree]):
    for _ in range(100):
        bst = bst_class()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        assert list(bst) == sorted(values)


@pytest.mark.parametrize(
    "bst_class", [BinarySearchTreeIterative, BinarySearchTree, AVLTreeIterative]
)
def test_successor(bst_class: Type[BinarySearchTree]):
    for _ in range(100):
        bst = bst_class()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        sorted_values = tuple(sorted(values))
        for i in range(len(sorted_values) - 1):
            assert bst.successor(sorted_values[i]).key == sorted_values[i + 1]

        with pytest.raises(KeyError):
            _ = bst.successor(sorted_values[-1])


@pytest.mark.parametrize(
    "bst_class", [BinarySearchTreeIterative, BinarySearchTree, AVLTreeIterative]
)
def test_predecessor(bst_class: Type[BinarySearchTree]):
    for _ in range(100):
        bst = bst_class()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        sorted_values = tuple(sorted(values))
        for i in range(1, len(sorted_values)):
            assert (
                bst.predecessor(sorted_values[i]).key == sorted_values[i - 1]
            ), sorted_values

        with pytest.raises(KeyError):
            _ = bst.predecessor(sorted_values[0])


@pytest.mark.parametrize(
    "bst_class", [BinarySearchTreeIterative, BinarySearchTree, AVLTreeIterative]
)
def test_minimum(bst_class: Type[BinarySearchTree]):
    for _ in range(100):
        bst = bst_class()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        assert bst.minimum().key == min(values)

    bst_empty = bst_class()
    with pytest.raises(ValueError):
        _ = bst_empty.minimum()


@pytest.mark.parametrize(
    "bst_class", [BinarySearchTreeIterative, BinarySearchTree, AVLTreeIterative]
)
def test_maximum(bst_class: Type[BinarySearchTree]):
    for _ in range(100):
        bst = bst_class()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        assert bst.maximum().key == max(values)

    bst_empty = bst_class()
    with pytest.raises(ValueError):
        _ = bst_empty.maximum()


@pytest.mark.parametrize(
    "bst_class", [BinarySearchTreeIterative, BinarySearchTree, AVLTreeIterative]
)
def test_extract_min(bst_class: Type[BinarySearchTree]):
    for _ in range(100):
        bst = bst_class()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)
            assert value in bst

        for expected_key in sorted(values):
            (key, _), _ = bst.extract_min()
            assert key == expected_key
            assert expected_key not in bst

        assert not bst