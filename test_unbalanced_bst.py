from random import randint

from bst import BST


def test_insertion():
    for _ in range(100):
        values = frozenset([randint(-10000, 10000) for _ in range(500)])
        bst = BST[int, None]()
        for val in values:
            bst.insert(val, None)
            assert val in bst

        for val in values:
            del bst[val]

        assert not bst
        assert len(bst) == 0
        assert bst.root is None


def test_sorted():
    for _ in range(100):
        bst = BST[int, None]()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        assert list(bst) == sorted(values)


def test_successor():
    for _ in range(100):
        bst = BST[int, None]()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        sorted_values = tuple(sorted(values))
        for i in range(len(sorted_values) - 1):
            assert bst.successor(sorted_values[i]).key == sorted_values[i + 1]

        assert bst.successor(sorted_values[-1]) is None


def test_predecessor():
    for _ in range(100):
        bst = BST[int, None]()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        sorted_values = tuple(sorted(values))
        for i in range(1, len(sorted_values)):
            assert bst.predecessor(sorted_values[i]).key == sorted_values[i - 1]

        assert bst.predecessor(sorted_values[0]) is None


def test_minimum():
    for _ in range(100):
        bst = BST[int, None]()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        assert bst.minimum().key == min(values)

    bst_empty = BST[int, None]()
    assert bst_empty.minimum() is None


def test_maximum():
    for _ in range(100):
        bst = BST[int, None]()
        values = frozenset([randint(-10000, 10000) for _ in range(50)])
        for value in values:
            bst.insert(value, None)

        assert bst.maximum().key == max(values)

    bst_empty = BST[int, None]()
    assert bst_empty.maximum() is None
