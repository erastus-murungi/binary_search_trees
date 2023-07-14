from math import inf
from operator import itemgetter
from random import randint
from typing import Iterator, cast

import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists, text, tuples
from more_itertools import unique_everseen

from avl import AVLTreeIterative
from bst import BST, BSTIterative, BSTWithParentIterative
from rbt import RedBlackTree
from scapegoat import ScapeGoatTree
from splaytree import SplayTree
from treap import Treap, TreapNode, TreapSplitMerge
from tree_23 import Tree23
from ziptree import ZipTree, ZipTreeRecursive


def gen_random_string(count: int) -> Iterator[str]:
    for _ in range(count):
        yield "".join([chr(randint(97, 122)) for _ in range(randint(1, 10))])


ALL_CLASSES = [
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
]


@given(lists(tuples(integers(), text())))
def test_insertion_and_deletion(key_value_pairs):
    for bst_class in ALL_CLASSES:
        bst = bst_class()
        for key, value in key_value_pairs:
            bst.insert(key, value)
            assert key in bst
            bst.validate(-inf, inf)

        key_value_pairs = list(unique_everseen(key_value_pairs, key=itemgetter(0)))
        for key, value in key_value_pairs:
            _ = bst.delete(key)
            # assert deleted_value == value
            assert key not in bst
            bst.validate(-inf, inf)

        assert not bst
        assert len(bst) == 0


@given(lists(integers(), unique=True))
def test_sorted(keys):
    for bst_class in ALL_CLASSES:
        bst = bst_class()
        for key in keys:
            bst.insert(key, None)

        assert list(bst) == sorted(set(keys))


@given(lists(integers(), unique=True, min_size=1))
def test_successor(keys):
    for bst_class in ALL_CLASSES:
        bst = bst_class()
        for key in keys:
            bst.insert(key, None)

        sorted_keys = tuple(sorted(keys))
        for i in range(len(sorted_keys) - 1):
            assert bst.successor(sorted_keys[i])[0] == sorted_keys[i + 1]

        with pytest.raises(KeyError):
            _ = bst.successor(sorted_keys[-1])[0]


@given(lists(integers(), unique=True, min_size=1))
def test_predecessor(keys):
    for bst_class in ALL_CLASSES:
        bst = bst_class()
        for key in keys:
            bst.insert(key, None)

        sorted_keys = tuple(sorted(keys))
        for i in range(1, len(sorted_keys)):
            assert bst.predecessor(sorted_keys[i])[0] == sorted_keys[i - 1]

        with pytest.raises(KeyError):
            _ = bst.predecessor(sorted_keys[0])


@given(lists(integers(), unique=True, min_size=1))
def test_minimum(keys):
    for bst_class in ALL_CLASSES:
        bst = bst_class()
        for key in keys:
            bst.insert(key, None)

        assert bst.minimum()[0] == min(keys)

        bst_empty = bst_class()
        with pytest.raises(ValueError):
            _ = bst_empty.minimum()


@given(lists(integers(), unique=True, min_size=1))
def test_maximum(keys):
    for bst_class in ALL_CLASSES:
        bst = bst_class()
        for key in keys:
            bst.insert(key, None)

        assert bst.maximum()[0] == max(keys)

        bst_empty = bst_class()
        with pytest.raises(ValueError):
            _ = bst_empty.maximum()


@given(lists(integers(), unique=True, min_size=1))
def test_extract_min(keys):
    for bst_class in ALL_CLASSES:
        bst = bst_class()
        for key in keys:
            bst.insert(key, None)

        for expected_key in sorted(keys):
            key, _ = bst.extract_min()
            assert key == expected_key
            bst.validate(-inf, inf)
            assert expected_key not in bst

        assert not bst


@given(lists(integers(), unique=True, min_size=1))
def test_extract_max(keys):
    for bst_class in ALL_CLASSES:
        bst = bst_class()
        for key in keys:
            bst.insert(key, None)

        for expected_key in sorted(keys, reverse=True):
            key, _ = bst.extract_max()
            assert key == expected_key
            bst.validate(-inf, inf)
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

            bst.validate(-inf, inf)
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
