from random import randint
from sys import maxsize
from typing import Type

import pytest

from bst import BST
from avl import AVL


@pytest.mark.parametrize("bst_class", [BST, AVL])
def test_insertion(bst_class: Type[BST]):
    for _ in range(50):
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        bst = bst_class[int, None]()
        for val in values:
            bst.insert(val, None)
            assert val in bst
            bst.check_invariants(-maxsize, maxsize)

        for val in values:
            del bst[val]
            bst.check_invariants(-maxsize, maxsize)

        assert not bst
        assert len(bst) == 0
        assert bst.root is None


@pytest.mark.parametrize("bst_class", [BST, AVL])
def test_sorted(bst_class: Type[BST]):
    for _ in range(100):
        bst = bst_class[int, None]()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        assert list(bst) == sorted(values)


@pytest.mark.parametrize("bst_class", [BST, AVL])
def test_successor(bst_class: Type[BST]):
    for _ in range(100):
        bst = bst_class[int, None]()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        sorted_values = tuple(sorted(values))
        for i in range(len(sorted_values) - 1):
            assert bst.successor(sorted_values[i]).key == sorted_values[i + 1]

        assert bst.successor(sorted_values[-1]) is None


@pytest.mark.parametrize("bst_class", [BST, AVL])
def test_predecessor(bst_class: Type[BST]):
    for _ in range(100):
        bst = bst_class[int, None]()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        sorted_values = tuple(sorted(values))
        for i in range(1, len(sorted_values)):
            assert bst.predecessor(sorted_values[i]).key == sorted_values[i - 1]

        assert bst.predecessor(sorted_values[0]) is None


@pytest.mark.parametrize("bst_class", [BST, AVL])
def test_minimum(bst_class: Type[BST]):
    for _ in range(100):
        bst = bst_class[int, None]()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        assert bst.minimum().key == min(values)

    bst_empty = bst_class[int, None]()
    assert bst_empty.minimum() is None


@pytest.mark.parametrize("bst_class", [BST, AVL])
def test_maximum(bst_class: Type[BST]):
    for _ in range(100):
        bst = bst_class[int, None]()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        assert bst.maximum().key == max(values)

    bst_empty = bst_class[int, None]()
    assert bst_empty.maximum() is None
