from __future__ import annotations
from abc import ABC
from collections import Counter
from dataclasses import dataclass, field
from random import randint, random
from typing import Generic, Union, TypeGuard

from bst import AbstractBinarySearchTreeIterative
from nodes import AbstractBinarySearchTreeNode, Sentinel
from core import Comparable, Value, SentinelReferenceError, SentinelType, NodeType


@dataclass(slots=True)
class ZipNode(
    AbstractBinarySearchTreeNode[
        Comparable, Value, "ZipNode[Comparable, Value]", Sentinel[Comparable]
    ],
):
    key: Comparable
    value: Value
    left: Union[ZipNode[Comparable, Value], Sentinel[Comparable]] = field(
        default_factory=Sentinel[Comparable], repr=False
    )
    right: Union[ZipNode[Comparable, Value], Sentinel[Comparable]] = field(
        default_factory=Sentinel[Comparable], repr=False
    )
    rank: int = field(default_factory=int)


def random_rank():
    rank = 0
    while random() > 0.5:
        rank += 1
    return rank


class ZipTree(
    AbstractBinarySearchTreeIterative[
        Comparable, Value, ZipNode[Comparable, Value], Sentinel[Comparable]
    ]
):
    def insert(self, key: Comparable, value: Value, allow_overwrite: bool = False):
        """

        Inserting x into a zip tree works as follows:

            1. Assign a random rank to x by flipping coins and counting
                how many flips were needed to get heads.
            2. Do a search for x. Stop the search once you reach a node where
                the node's left child has a lower rank than x,
                the node's right child has a rank less than or equal to x, and
                the node's right child, if it has the same rank as x, has a larger key than x.
            3. Perform an unzip. Specifically:
                Continue the search for x as before, recording when we move left and when we move right.
                Chain all the nodes together where we went left by
                    making each the left child of the previously-visited left-moving node.
                Chain all the nodes together where we went right by
                    making each the right child of the previously-visited right-moving node.
                Make those two chains the children of the node x.
        """
        rank = random_rank()
        node = self.node(key=key, value=value, rank=rank)
        current = self.root
        prev = self.sentinel()
        while self.is_node(current) and (
            node.rank < current.rank or rank == current.rank and node.key > current.key
        ):
            prev = current
            if node.key < current.key:
                current = current.left
            else:
                current = current.right
        if current is self.root:
            self.root = node
        else:
            assert self.is_node(prev)
            if key < prev.key:
                prev.left = node
            elif key > prev.key:
                prev.right = node
            else:
                if allow_overwrite:
                    prev.value = value
                    return prev
                else:
                    raise KeyError(f"Key {key} already exists.")

        if self.is_sentinel(current):
            node.left = self.sentinel()
            node.right = self.sentinel()
            self.size += 1
            return node
        assert self.is_node(current)
        if key < current.key:
            node.right = current
        else:
            node.left = current

        prev = node

        while self.is_node(current):
            fix = prev
            if current.key < key:
                while True:
                    prev = current
                    current = current.right
                    if self.is_sentinel(current) or current.key > key:
                        break
            else:
                while True:
                    prev = current
                    current = current.left
                    if self.is_sentinel(current) or current.key < key:
                        break
            if fix.key > key or (fix == node and prev.key > key):
                fix.left = current
            else:
                fix.right = current
        self.size += 1
        return node

    def delete(self, x: Comparable):
        prev: Union[ZipNode[Comparable, Value], Sentinel[Comparable]] = self.sentinel()
        current: Union[
            ZipNode[Comparable, Value], Sentinel[Comparable]
        ] = self.nonnull_root
        while self.is_node(current) and x != current.key:  # find the x in the tree
            prev = current
            if x < current.key:
                current = current.nonnull_left
            else:
                current = current.nonnull_right
        left = current.left  # get the left and right subtree of x
        right = current.right
        y = current
        if self.is_sentinel(left):
            current = right
        elif self.is_sentinel(right):
            current = left
        elif left.rank >= right.rank:
            current = left
        else:
            current = right

        if self.root.key == x:
            self.root = current
        elif x < prev.key:
            prev.left = current
        else:
            prev.right = current

        while self.is_node(left) and self.is_node(right):
            if left.rank >= right.rank:
                while True:
                    prev = left
                    left = left.right
                    if self.is_sentinel(left) or left.rank < right.rank:
                        break
                prev.right = right
            else:
                while True:
                    prev = right
                    right = right.left
                    if self.is_sentinel(right) or left.rank >= right.rank:
                        break
                prev.left = left

        self.size -= 1
        return y

    def is_node(
        self, node: Union[ZipNode[Comparable, Value], Sentinel[Comparable]]
    ) -> TypeGuard[ZipNode[Comparable, Value]]:
        return isinstance(node, ZipNode)

    def is_sentinel(
        self, node: Union[ZipNode[Comparable, Value], Sentinel[Comparable]]
    ) -> TypeGuard[Sentinel[Comparable]]:
        return isinstance(node, Sentinel)

    def node(
        self, key: Comparable, value: Value, *args, **kwargs
    ) -> ZipNode[Comparable, Value]:
        return ZipNode(key=key, value=value)

    def sentinel(self) -> Sentinel[Comparable]:
        return Sentinel[Comparable]()


if __name__ == "__main__":
    from random import seed

    seed(0)

    for _ in range(1000):
        bst: ZipTree[int, None] = ZipTree[int, None]()
        num_values = 3
        # values = list({randint(0, 10000) for _ in range(num_values)})
        # values = [35, 70, 51]
        values = [7401, 66, 9236]

        for i, val in enumerate(values):
            bst.insert(val, None, allow_overwrite=True)
            # print(bst.pretty_str())
            bst.validate(0, 1000000)
            assert val in bst

        assert bst.size == len(values)

        # print(bst.pretty_str())

        for val in values:
            deleted = bst.delete(val)
            assert deleted.key == val
            bst.validate(0, 1000000)
            print(bst.pretty_str())
            assert val not in bst, values
