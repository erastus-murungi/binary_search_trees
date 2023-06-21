from random import randint
from sys import maxsize
from typing import Type

import pytest

from avl import AVLTreeIterative
from bst import (
    AbstractBinarySearchTree,
    BinarySearchTreeIterative,
    BinarySearchTreeRecursive,
    BSTWithParentIterative,
)
from rbt import RedBlackTree
from scapegoat import ScapeGoatTree
from splaytree import SplayTree
from ziptree import ZipTree


@pytest.mark.parametrize(
    "bst_class",
    [
        BinarySearchTreeRecursive,
        BinarySearchTreeIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ScapeGoatTree,
    ],
)
def test_insertion(bst_class: Type[AbstractBinarySearchTree]):
    for _ in range(50):
        values = frozenset([randint(-10000, 10000) for _ in range(100)])
        bst = bst_class()
        for val in values:
            bst.insert(val, None)
            assert val in bst
            bst.validate(-maxsize, maxsize)

        for val in values:
            deleted = bst.delete(val)
            assert deleted.key == val
            assert val not in bst
            bst.validate(-maxsize, maxsize)

        assert not bst
        assert len(bst) == 0


@pytest.mark.parametrize(
    "bst_class",
    [
        BinarySearchTreeRecursive,
        BinarySearchTreeIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ScapeGoatTree,
    ],
)
def test_sorted(bst_class: Type[AbstractBinarySearchTree]):
    for _ in range(100):
        bst = bst_class()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        assert list(bst) == sorted(values)


@pytest.mark.parametrize(
    "bst_class",
    [
        BinarySearchTreeRecursive,
        BinarySearchTreeIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ScapeGoatTree,
    ],
)
def test_successor(bst_class: Type[AbstractBinarySearchTree]):
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
    "bst_class",
    [
        BinarySearchTreeRecursive,
        BinarySearchTreeIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ScapeGoatTree,
    ],
)
def test_predecessor(bst_class: Type[AbstractBinarySearchTree]):
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
    "bst_class",
    [
        BinarySearchTreeRecursive,
        BinarySearchTreeIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ScapeGoatTree,
    ],
)
def test_minimum(bst_class: Type[AbstractBinarySearchTree]):
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
    "bst_class",
    [
        BinarySearchTreeRecursive,
        BinarySearchTreeIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ScapeGoatTree,
    ],
)
def test_maximum(bst_class: Type[AbstractBinarySearchTree]):
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
    "bst_class",
    [
        BinarySearchTreeRecursive,
        BinarySearchTreeIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ScapeGoatTree,
    ],
)
def test_extract_min(bst_class: Type[AbstractBinarySearchTree]):
    for _ in range(100):
        bst = bst_class()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)
            assert value in bst

        for expected_key in sorted(values):
            key, _ = bst.extract_min()
            assert key == expected_key
            bst.validate(-maxsize, maxsize)
            assert expected_key not in bst

        assert not bst


@pytest.mark.parametrize(
    "bst_class",
    [
        BinarySearchTreeRecursive,
        BinarySearchTreeIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ScapeGoatTree,
    ],
)
def test_extract_max(bst_class: Type[AbstractBinarySearchTree]):
    for _ in range(100):
        bst = bst_class()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)
            assert value in bst

        for expected_key in sorted(values, reverse=True):
            key, _ = bst.extract_max()
            assert key == expected_key
            bst.validate(-maxsize, maxsize)
            assert expected_key not in bst

        assert not bst
