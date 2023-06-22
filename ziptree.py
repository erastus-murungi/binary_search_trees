from __future__ import annotations

from dataclasses import dataclass, field
from random import random
from typing import Any, TypeGuard, Union

from bst import AbstractBST, AbstractBSTIterative
from core import Key, Value
from nodes import AbstractBSTNode, Sentinel


@dataclass(slots=True)
class ZipNode(
    AbstractBSTNode[Key, Value, "ZipNode[Key, Value]", Sentinel],
):
    key: Key
    value: Value
    left: Union[ZipNode[Key, Value], Sentinel] = field(
        default_factory=Sentinel, repr=False
    )
    right: Union[ZipNode[Key, Value], Sentinel] = field(
        default_factory=Sentinel, repr=False
    )
    rank: int = field(default_factory=int)


def random_rank():
    rank = 0
    while random() > 0.5:
        rank += 1
    return rank


class ZipTree(AbstractBSTIterative[Key, Value, ZipNode[Key, Value], Sentinel]):
    def insert_node(self, node: ZipNode[Key, Value], allow_overwrite: bool = False):
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
        current: Union[ZipNode[Key, Value], Sentinel] = self.root
        prev: Union[ZipNode[Key, Value], Sentinel] = self.sentinel()

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
            if node.key < prev.key:
                prev.left = node
            elif node.key > prev.key:
                prev.right = node
            else:
                if allow_overwrite:
                    prev.value = node.value
                    return prev
                else:
                    raise KeyError(f"Key {node.key} already exists.")

        if self.is_sentinel(current):
            node.left = self.sentinel()
            node.right = self.sentinel()
            self.size += 1
            return node
        assert self.is_node(current)
        if node.key < current.key:
            node.right = current
        else:
            node.left = current

        prev = node

        while self.is_node(current):
            assert self.is_node(prev)
            fix = prev
            if current.key < node.key:
                while True:
                    prev = current
                    current = current.right
                    if self.is_sentinel(current):
                        break
                    assert self.is_node(current)
                    if current.key > node.key:
                        break
            else:
                while True:
                    prev = current
                    current = current.left
                    if self.is_sentinel(current):
                        break
                    assert self.is_node(current)
                    if current.key < node.key:
                        break
            if fix.key > node.key or (fix == node and prev.key > node.key):
                fix.left = current
            else:
                fix.right = current
        self.size += 1
        return node

    def delete(self, target_key: Key) -> ZipNode[Key, Value]:
        target_node, prev = self.access_with_parent(target_key)

        current: Union[ZipNode[Key, Value] | Sentinel]

        left, right = target_node.left, target_node.right
        if self.is_sentinel(left):
            current = right
        elif self.is_sentinel(right):
            current = left
        else:
            assert self.is_node(left) and self.is_node(right)
            if left.rank >= right.rank:
                current = left
            else:
                current = right
        if self.is_node(prev):
            if target_key < prev.key:
                prev.left = current
            else:
                prev.right = current
        else:
            assert self.nonnull_root.key == target_key
            self.root = current

        while self.is_node(left) and self.is_node(right):
            if left.rank >= right.rank:
                while True:
                    prev = left
                    left = left.right
                    if self.is_sentinel(left):
                        break
                    assert self.is_node(left)
                    if left.rank < right.rank:
                        break
                prev.right = right
            else:
                while True:
                    prev = right
                    right = right.left
                    if self.is_sentinel(right):
                        break
                    assert self.is_node(right)
                    if left.rank >= right.rank:
                        break

                prev.left = left

        self.size -= 1
        return target_node

    @staticmethod
    def is_node(node: Any) -> TypeGuard[ZipNode[Key, Value]]:
        return isinstance(node, ZipNode)

    @staticmethod
    def node(key: Key, value: Value, *args, **kwargs) -> ZipNode[Key, Value]:
        return ZipNode(key=key, value=value)

    @classmethod
    def sentinel_class(cls):
        return Sentinel


class ZipTreeRecursive(AbstractBST[Key, Value, ZipNode[Key, Value], Sentinel]):
    def zip(
        self,
        x: Union[ZipNode[Key, Value], Sentinel],
        y: Union[ZipNode[Key, Value], Sentinel],
    ) -> Union[ZipNode[Key, Value], Sentinel]:
        if self.is_sentinel(x):
            return y
        if self.is_sentinel(y):
            return x
        assert self.is_node(x) and self.is_node(y)
        if x.rank < y.rank:
            y.left = self.zip(x, y.left)
            return y
        else:
            x.right = self.zip(x.right, y)
            return x

    def insert_node(
        self, node: ZipNode[Key, Value], allow_overwrite: bool = True
    ) -> bool:
        old_size = self.size

        def insert(
            root: Union[ZipNode[Key, Value], Sentinel], x: ZipNode[Key, Value]
        ) -> ZipNode[Key, Value]:
            if self.is_sentinel(root):
                return x
            assert self.is_node(root)
            if x.key == root.key:
                if allow_overwrite:
                    root.value = x.value
                    self.size -= 1
                else:
                    raise KeyError(f"Key {x.key} already exists.")
            elif x.key < root.key:
                if insert(root.left, x) == x:
                    if x.rank < root.rank:
                        root.left = x
                    else:
                        root.left = x.right
                        x.right = root
                        return x
            else:
                if insert(root.right, x) == x:
                    if x.rank <= root.rank:
                        root.right = x
                    else:
                        root.right = x.left
                        x.left = root
                        return x
            return root

        self.root = insert(self.root, node)
        self.size += 1
        return self.size > old_size

    def delete(self, key: Key) -> ZipNode[Key, Value]:
        def delete(root: ZipNode[Key, Value], x: ZipNode[Key, Value]):
            if x.key == root.key:
                return self.zip(root.left, root.right)
            elif self.is_node(root.left) and x.key < root.key:
                if x.key == root.left.key:
                    root.left = self.zip(root.left.left, root.left.right)
                else:
                    delete(root.left, x)
            else:
                assert self.is_node(root.right) and x.key > root.key
                if x.key == root.right.key:
                    root.right = self.zip(root.right.left, root.right.right)
                else:
                    delete(root.right, x)
            return root

        if node := self.access(key):
            self.root = delete(self.nonnull_root, node)
            self.size -= 1
            return node
        else:
            raise KeyError(f"Key {key} does not exist.")

    @staticmethod
    def is_node(node: Any) -> TypeGuard[ZipNode[Key, Value]]:
        return isinstance(node, ZipNode)

    @staticmethod
    def node(key: Key, value: Value, *args, **kwargs) -> ZipNode[Key, Value]:
        return ZipNode(key=key, value=value)

    @classmethod
    def sentinel_class(cls):
        return Sentinel
