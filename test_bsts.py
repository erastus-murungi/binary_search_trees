from math import inf, nan
from random import randint
from typing import Iterator, cast

import pytest
from hypothesis import given
from hypothesis.strategies import floats, integers, lists, text
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


@given(lists(integers()))
def test_insertion_and_deletion_integers(keys):
    for bst_class in ALL_CLASSES:
        bst = bst_class()
        for key in keys:
            bst.insert(key, None)
            assert key in bst
            bst.validate(-inf, inf)

        keys = list(unique_everseen(keys))
        for key in keys:
            _ = bst.delete(key)
            # assert deleted_value == value
            assert key not in bst
            bst.validate(-inf, inf)

        assert not bst
        assert len(bst) == 0


@given(lists(floats(0, 1e100)))  # ignore nan and inf for now
def test_insertion_and_deletion_randoms(keys):
    for bst_class in ALL_CLASSES:
        bst = bst_class()
        for key in keys:
            bst.insert(key, None)
            assert key in bst
            bst.validate(-inf, inf)

        keys = list(unique_everseen(keys))
        for key in keys:
            _ = bst.delete(key)
            # assert deleted_value == value
            assert key not in bst
            bst.validate(-inf, inf)

        assert not bst
        assert len(bst) == 0


@given(lists(text()))
def test_insertion_and_deletion_text(keys):
    for bst_class in ALL_CLASSES:
        bst = bst_class()
        for key in keys:
            bst.insert(key, None)
            assert key in bst
            # bst.validate('', max(keys))

        keys = list(unique_everseen(keys))
        for key in keys:
            _ = bst.delete(key)
            # assert deleted_value == value
            assert key not in bst
            # bst.validate('', max(keys))

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


@given(lists(integers(), min_size=1, unique=True))
def test_treap_split_merge(keys):
    sorted_keys = list(sorted(keys))
    for expected_key in sorted_keys[:-1]:
        bst = TreapSplitMerge()
        for key in keys:
            bst.insert(key, None)
            assert key in bst

        bst.validate(-inf, inf)
        left, right = bst.split(expected_key, bst.root)
        assert expected_key in left
        assert expected_key not in right
        assert all(key <= expected_key for key in cast(TreapNode, left))
        assert all(key > expected_key for key in cast(TreapNode, right))

        assert sorted_keys == list(bst.merge(left, right))

    bst = TreapSplitMerge()
    for key in keys:
        bst.insert(key, None)
        assert key in bst
    left, _ = bst.split(max(keys), bst.root)
    assert list(left) == list(bst)
    _, right = bst.split(min(keys) - 1, bst.root)
    assert list(right) == list(bst)
