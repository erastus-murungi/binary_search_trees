from random import randint
from sys import maxsize
from typing import Iterator, Type, cast
from hypothesis.strategies import lists, integers

import pytest

from avl import AVLTreeIterative
from bst import BST, AbstractBST, BSTIterative, BSTWithParentIterative
from rbt import RedBlackTree
from scapegoat import ScapeGoatTree
from splaytree import SplayTree
from treap import Treap, TreapNode, TreapSplitMerge
from tree_23 import Tree23
from ziptree import ZipTree, ZipTreeRecursive


def gen_random_string(count: int) -> Iterator[str]:
    for _ in range(count):
        yield "".join([chr(randint(97, 122)) for _ in range(randint(1, 10))])


@pytest.mark.parametrize(
    "bst_class",
    [
        BST,
        BSTIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ZipTreeRecursive,
        ScapeGoatTree,
        Treap,
        TreapSplitMerge,
        Tree23
    ],
)
def test_insertion_and_deletion(bst_class: Type[AbstractBST]):
    for _ in range(50):
        keys = list({randint(-10000, 10000) for _ in range(100)})
        values = list(gen_random_string(len(keys)))
        bst = bst_class()
        for key, value in zip(keys, values):
            bst.insert(key, value)
            assert key in bst
            bst.validate(-maxsize, maxsize)

        for key, value in zip(keys, values):
            deleted_value = bst.delete(key)
            assert deleted_value == value
            assert key not in bst
            bst.validate(-maxsize, maxsize)

        assert not bst
        assert len(bst) == 0


@pytest.mark.parametrize(
    "bst_class",
    [
        BST,
        BSTIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ZipTreeRecursive,
        ScapeGoatTree,
        Treap,
        TreapSplitMerge,
        Tree23,
    ],
)
def test_sorted(bst_class: Type[AbstractBST]):
    for _ in range(100):
        bst = bst_class()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        assert list(bst) == sorted(values)


@pytest.mark.parametrize(
    "bst_class",
    [
        BST,
        BSTIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ZipTreeRecursive,
        ScapeGoatTree,
        Treap,
        TreapSplitMerge,
    ],
)
def test_successor(bst_class: Type[AbstractBST]):
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
        BST,
        BSTIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ZipTreeRecursive,
        ScapeGoatTree,
        Treap,
        TreapSplitMerge,
    ],
)
def test_predecessor(bst_class: Type[AbstractBST]):
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
        BST,
        BSTIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ZipTreeRecursive,
        ScapeGoatTree,
        Treap,
        TreapSplitMerge,
        Tree23
    ],
)
def test_minimum(bst_class: Type[AbstractBST]):
    for _ in range(100):
        bst = bst_class()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        assert bst.minimum()[0] == min(values)

    bst_empty = bst_class()
    with pytest.raises(ValueError):
        _ = bst_empty.minimum_node()


@pytest.mark.parametrize(
    "bst_class",
    [
        BST,
        BSTIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ZipTreeRecursive,
        ScapeGoatTree,
        Treap,
        TreapSplitMerge,
    ],
)
def test_maximum(bst_class: Type[AbstractBST]):
    for _ in range(100):
        bst = bst_class()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        assert bst.maximum()[0] == max(values)

    bst_empty = bst_class()
    with pytest.raises(ValueError):
        _ = bst_empty.maximum_node()


@pytest.mark.parametrize(
    "bst_class",
    [
        BST,
        BSTIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ZipTreeRecursive,
        ScapeGoatTree,
        Treap,
        TreapSplitMerge,
    ],
)
def test_extract_min(bst_class: Type[AbstractBST]):
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
        BST,
        BSTIterative,
        BSTWithParentIterative,
        AVLTreeIterative,
        RedBlackTree,
        SplayTree,
        ZipTree,
        ZipTreeRecursive,
        ScapeGoatTree,
        Treap,
        TreapSplitMerge,
    ],
)
def test_extract_max(bst_class: Type[AbstractBST]):
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


def test_treap_split_merge():
    for _ in range(10):
        values = list({randint(-10000, 10000) for _ in range(100)})
        sorted_values = list(sorted(values))
        for expected_key in sorted_values[:-1]:
            bst = TreapSplitMerge()
            for value in values:
                bst.insert(value, None)
                assert value in bst

            bst.validate(-maxsize, maxsize)
            left, right = bst.split(expected_key, bst.root)
            assert expected_key in left
            assert expected_key not in right
            assert all(key <= expected_key for key in cast(TreapNode, left))
            assert all(key > expected_key for key in cast(TreapNode, right))

            assert sorted_values == list(bst.merge(left, right))

        bst = TreapSplitMerge()
        for value in values:
            bst.insert(value, None)
            assert value in bst
        left, _ = bst.split(max(values), bst.root)
        assert list(left) == list(bst)
        _, right = bst.split(min(values) - 1, bst.root)
        assert list(right) == list(bst)
