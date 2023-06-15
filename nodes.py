from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Self, TypeGuard, Union, cast

from typeguard import typechecked

from core import (AbstractNode, AbstractSentinel, Comparable, NodeType,
                  SentinelReached, SentinelType, Value)


class Sentinel(
    AbstractSentinel[Comparable, "BinarySearchTreeInternalNode", "Sentinel"]
):
    @classmethod
    def default(cls) -> "Sentinel":
        return Sentinel()

    def pretty_str(self) -> str:
        return "∅"

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----'∅'\n"

    def validate(self, *arg, **kwargs) -> bool:
        return True

    def __len__(self) -> int:
        return 0

    def __contains__(self, key: object) -> bool:
        return False

    def access(self, key: Comparable) -> Self:
        return self


class BinarySearchTreeInternalNode(
    AbstractNode[Comparable, Value, "BinarySearchTreeInternalNode", Sentinel]
):
    left: Union[BinarySearchTreeInternalNode[Comparable, Value], Sentinel[Comparable]]
    right: Union[BinarySearchTreeInternalNode[Comparable, Value], Sentinel[Comparable]]

    def __init__(self, key: Comparable, value: Value):
        super().__init__(key, value)
        self.left = self.leaf()
        self.right = self.leaf()

    def leaf(self) -> Sentinel[Comparable]:
        return Sentinel()

    def is_leaf(self, node: Any) -> TypeGuard[Sentinel[Comparable]]:
        return isinstance(node, Sentinel)

    def pretty_str(self) -> str:
        return "".join(self.yield_line("", "R"))

    def validate(self, lower_limit: Comparable, upper_limit: Comparable) -> bool:
        if not (lower_limit < self.key < upper_limit):
            return False
        return self.left.validate(lower_limit, self.key) and self.right.validate(
            self.key, upper_limit
        )

    def __len__(self) -> int:
        return 1 + len(self.left) + len(self.right)

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----{self}\n"
        indent += "     " if prefix == "R" else "|    "
        yield from self.left.yield_line(indent, "L")
        yield from self.right.yield_line(indent, "R")

    def __contains__(self, key: Any) -> bool:
        node = self.access(key)
        return isinstance(node, type(self)) and node.key == key

    def access(
        self, key: Comparable
    ) -> Union[BinarySearchTreeInternalNode[Comparable, Value], Sentinel[Comparable]]:
        if self.key == key:
            return self
        elif key < self.key:
            return self.left.access(key)
        else:
            return self.right.access(key)

    def minimum(self) -> BinarySearchTreeInternalNode[Comparable, Value]:
        if isinstance(self.left, Sentinel):
            return self
        return self.left.minimum()

    def maximum(self) -> BinarySearchTreeInternalNode[Comparable, Value]:
        if isinstance(self.right, Sentinel):
            return self
        return self.right.maximum()

    @property
    def nonnull_right(self) -> BinarySearchTreeInternalNode[Comparable, Value]:
        if isinstance(self.right, BinarySearchTreeInternalNode):
            return self.right
        raise SentinelReached(f"Node {self.right} is not a BinarySearchTreeNode")

    @property
    def nonnull_left(self) -> BinarySearchTreeInternalNode[Comparable, Value]:
        if isinstance(self.left, BinarySearchTreeInternalNode):
            return self.left
        raise SentinelReached(f"Node {self.left} is not a BinarySearchTreeNode")

    def successor(
        self, key: Comparable
    ) -> BinarySearchTreeInternalNode[Comparable, Value]:
        if key > self.key:
            return self.nonnull_right.successor(key)
        elif key < self.key:
            try:
                return self.nonnull_left.successor(key)
            except ValueError:
                return self
        else:
            return self.nonnull_right.minimum()

    def predecessor(
        self, key: Comparable
    ) -> BinarySearchTreeInternalNode[Comparable, Value]:
        if key < self.key:
            return self.nonnull_left.predecessor(key)
        elif key > self.key:
            try:
                return self.nonnull_right.predecessor(key)
            except ValueError:
                return self
        else:
            return self.nonnull_left.maximum()

    def insert(
        self, key: Comparable, value: Value, allow_overwrite: bool = False
    ) -> BinarySearchTreeInternalNode[Comparable, Value]:
        if self.key == key:
            if allow_overwrite:
                self.value = value
            else:
                raise ValueError(f"Overwrites of {key} not allowed")
        elif key < self.key:
            if self.is_node(self.left):
                self.left = self.left.insert(key, value, allow_overwrite)
            else:
                self.left = self.node(key, value)
        else:
            if self.is_node(self.right):
                self.right = self.right.insert(key, value, allow_overwrite)
            else:
                self.right = self.node(key, value)
        return self

    def delete(
        self, key: Comparable
    ) -> Union[BinarySearchTreeInternalNode[Comparable, Value], Sentinel[Comparable]]:
        if key < self.key:
            self.left = self.nonnull_left.delete(key)
        elif key > self.key:
            self.right = self.nonnull_right.delete(key)
        else:
            if self.is_leaf(self.left):
                return self.right
            elif self.is_leaf(self.right):
                return self.left
            else:
                successor = self.nonnull_right.minimum()
                self.key, self.value = successor.key, successor.value
                self.right = self.nonnull_right.delete(successor.key)
        return self

    def extract_min(
        self,
    ) -> tuple[
        tuple[Comparable, Value],
        Union[BinarySearchTreeInternalNode[Comparable, Value], Sentinel[Comparable]],
    ]:
        try:
            keyval, self.left = self.nonnull_left.extract_min()
            return keyval, self
        except SentinelReached:
            keyval = (self.key, self.value)
            if self.is_node(self.right):
                return keyval, self.right
            else:
                return keyval, self.leaf()

    def extract_max(
        self,
    ) -> tuple[
        tuple[Comparable, Value],
        Union[BinarySearchTreeInternalNode[Comparable, Value], Sentinel[Comparable]],
    ]:
        try:
            keyval, self.right = self.nonnull_right.extract_max()
            return keyval, self
        except SentinelReached:
            keyval = (self.key, self.value)
            if self.is_node(self.left):
                return keyval, self.left
            else:
                return keyval, self.leaf()

    def node(
        self, key: Comparable, value: Value, *args, **kwargs
    ) -> BinarySearchTreeInternalNode[Comparable, Value]:
        return BinarySearchTreeInternalNode(key, value)

    def is_node(
        self, node: Any
    ) -> TypeGuard[BinarySearchTreeInternalNode[Comparable, Value]]:
        if isinstance(node, BinarySearchTreeInternalNode):
            return True
        return False

    def inorder(self) -> Iterator[BinarySearchTreeInternalNode[Comparable, Value]]:
        if self.is_node(self.left):
            yield from self.left.inorder()
        yield self
        if self.is_node(self.right):
            yield from self.right.inorder()

    def preorder(self) -> Iterator[BinarySearchTreeInternalNode[Comparable, Value]]:
        yield self
        if self.is_node(self.left):
            yield from self.left.inorder()
        if self.is_node(self.right):
            yield from self.right.inorder()

    def postorder(self) -> Iterator[BinarySearchTreeInternalNode[Comparable, Value]]:
        if self.is_node(self.left):
            yield from self.left.inorder()
        if self.is_node(self.right):
            yield from self.right.inorder()
        yield self

    def level_order(self) -> Iterator[BinarySearchTreeInternalNode[Comparable, Value]]:
        raise NotImplementedError

    def swap(self, other: BinarySearchTreeInternalNode[Comparable, Value]) -> None:
        self.key = other.key
        self.value = other.value
        self.left = other.left
        self.right = other.right

    def __setitem__(self, key: Comparable, value: Value) -> None:
        self.insert(key, value, allow_overwrite=True)

    def __delitem__(self, key: Comparable) -> None:
        self.delete(key)

    def __getitem__(self, key: Comparable) -> Value:
        node = self.access(key)
        if self.is_node(node):
            return node.value
        raise KeyError(f"Key {key} not found")

    def __iter__(self) -> Iterator[Comparable]:
        for node in self.inorder():
            yield node.key


if __name__ == "__main__":
    BinarySearchTreeInternalNode(0, None)
