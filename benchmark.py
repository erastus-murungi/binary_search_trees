import statistics
from sys import setrecursionlimit
from timeit import timeit
from typing import Final, Type

import matplotlib.pyplot as plt  # type: ignore

from avl import AVLTreeIterative
from bst import BST, BSTIterative, BSTWithParentIterative, Tree
from rbt import RedBlackTree
from scapegoat import ScapeGoatTree
from splaytree import SplayTree
from treap import Treap, TreapSplitMerge
from tree_23 import Tree23
from ziptree import ZipTree, ZipTreeRecursive

ALL_CLASSES: Final[tuple[Type[Tree], ...]] = (
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
)

setrecursionlimit(10_000_000)


def perform_insertions(constructor: Type[Tree], n_insertions: int) -> None:
    bst = constructor()
    for i in range(n_insertions):
        bst.insert(i, i)


def benchmark_many_consecutive_insertions() -> None:
    times = []
    for constructor in ALL_CLASSES:
        T = []
        for n_insertions in range(1, 1000, 100):
            T.append(
                timeit(
                    lambda: perform_insertions(constructor, n_insertions),
                    number=2,
                ),
            )
        times.append(statistics.mean(T))

    labels = [const.__name__ for const in ALL_CLASSES]

    plt.bar(labels, times)
    plt.xticks(rotation=45)
    plt.xlabel("Data structure")
    plt.grid()
    plt.ylabel("Time (s)")
    plt.show()


if __name__ == "__main__":
    benchmark_many_consecutive_insertions()
