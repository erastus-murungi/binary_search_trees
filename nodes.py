from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, TypeGuard, TypeVar, Union, cast

from core import (
    AbstractNode,
    AbstractSentinel,
    Comparable,
    SentinelReferenceError,
    SentinelType,
    SupportsParent,
    Value,
)

BinaryNodeType = TypeVar("BinaryNodeType", bound="AbstractBinarySearchTreeNode")
BinaryNodeWithParentType = TypeVar(
    "BinaryNodeWithParentType", bound="AbstractBinarySearchTreeInternalNodeWithParent"
)


class Sentinel(AbstractSentinel[Comparable]):
    @classmethod
    def default(cls) -> Sentinel[Comparable]:
        return Sentinel()


class AbstractSentinelWithParent(
    AbstractSentinel[Comparable],
    SupportsParent[BinaryNodeWithParentType, SentinelType],
    ABC,
):
    pass


@dataclass
class AbstractBinarySearchTreeNode(
    AbstractNode[Comparable, Value, BinaryNodeType, SentinelType], ABC
):
    key: Comparable
    value: Value
    left: Union[BinaryNodeType, SentinelType] = field(repr=False)
    right: Union[BinaryNodeType, SentinelType] = field(repr=False)

    def _is_node(self, node: Any) -> TypeGuard[BinaryNodeType]:
        return isinstance(node, type(self))

    def choose(self, key: Comparable) -> Union[BinaryNodeType, SentinelType]:
        if key > self.key:
            return self.right
        elif key < self.key:
            return self.left
        else:
            raise ValueError(f"Key {key} already exists in tree")

    def choose_set(self, key: Comparable, node: BinaryNodeType) -> None:
        if key > self.key:
            self.right = node
        elif key < self.key:
            self.left = node
        else:
            raise ValueError(f"Key {key} already exists in tree")

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
        node = self.access_no_throw(cast(Comparable, key))
        return isinstance(node, type(self)) and node.key == key

    def access(self, key: Comparable) -> BinaryNodeType:
        if self.key == key:
            return cast(BinaryNodeType, self)
        elif key < self.key:
            if self._is_node(self.left):
                return self.left.access(key)
            raise SentinelReferenceError(
                f"Key {key} not found, Sentinel {self.left} reached"
            )
        else:
            if self._is_node(self.right):
                return self.right.access(key)
            raise SentinelReferenceError(
                f"Key {key} not found, Sentinel {self.right} reached"
            )

    def minimum(self) -> BinaryNodeType:
        if self._is_node(self.left):
            return self.left.minimum()
        return cast(BinaryNodeType, self)

    def maximum(self) -> BinaryNodeType:
        if self._is_node(self.right):
            return self.right.maximum()
        return cast(BinaryNodeType, self)

    @property
    def nonnull_right(self) -> BinaryNodeType:
        if self._is_node(self.right):
            return self.right
        raise SentinelReferenceError(f"Node {self.right} is not a BinarySearchTreeNode")

    @property
    def nonnull_left(self) -> BinaryNodeType:
        if self._is_node(self.left):
            return self.left
        raise SentinelReferenceError(f"Node {self.left} is not a BinarySearchTreeNode")

    def successor(self, key: Comparable) -> BinaryNodeType:
        if key > self.key:
            return self.nonnull_right.successor(key)
        elif key < self.key:
            try:
                return self.nonnull_left.successor(key)
            except ValueError:
                return cast(BinaryNodeType, self)
        else:
            return self.nonnull_right.minimum()

    def predecessor(self, key: Comparable) -> BinaryNodeType:
        if key < self.key:
            return self.nonnull_left.predecessor(key)
        elif key > self.key:
            try:
                return self.nonnull_right.predecessor(key)
            except ValueError:
                return cast(BinaryNodeType, self)
        else:
            return self.nonnull_left.maximum()

    def insert_node(
        self, node: BinaryNodeType, allow_overwrite: bool = False
    ) -> BinaryNodeType:
        if self.key == node.key:
            if allow_overwrite:
                self.value = node.value
            else:
                raise ValueError(f"Overwrites of {node.key} not allowed")
        elif node.key < self.key:
            if self._is_node(self.left):
                self.left = self.left.insert_node(node, allow_overwrite)
            else:
                self.left = node
        else:
            if self._is_node(self.right):
                self.right = self.right.insert_node(node, allow_overwrite)
            else:
                self.right = node
        return cast(BinaryNodeType, self)

    def _delete(self, key: Comparable) -> Union[BinaryNodeType, SentinelType]:
        if key < self.key:
            self.left = self.nonnull_left._delete(key)
        elif key > self.key:
            self.right = self.nonnull_right._delete(key)
        else:
            if not self._is_node(self.left):
                return self.right
            elif not self._is_node(self.right):
                return self.left
            else:
                successor = self.nonnull_right.minimum()
                self.key, self.value = successor.key, successor.value
                self.right = self.nonnull_right._delete(successor.key)
        return cast(BinaryNodeType, self)

    def _extract_min(
        self,
    ) -> tuple[tuple[Comparable, Value], Union[BinaryNodeType, SentinelType]]:
        try:
            keyval, self.left = self.nonnull_left._extract_min()
            return keyval, cast(BinaryNodeType, self)
        except SentinelReferenceError:
            keyval = (self.key, self.value)
            if self._is_node(self.right):
                return keyval, self.right
            else:
                return keyval, self.left

    def _extract_max(
        self,
    ) -> tuple[tuple[Comparable, Value], Union[BinaryNodeType, SentinelType]]:
        try:
            keyval, self.right = self.nonnull_right._extract_max()
            return keyval, cast(BinaryNodeType, self)
        except SentinelReferenceError:
            keyval = (self.key, self.value)
            if self._is_node(self.left):
                return keyval, self.left
            else:
                return keyval, self.right

    def inorder(self) -> Iterator[BinaryNodeType]:
        if self._is_node(self.left):
            yield from self.left.inorder()
        yield cast(BinaryNodeType, self)
        if self._is_node(self.right):
            yield from self.right.inorder()

    def preorder(self) -> Iterator[BinaryNodeType]:
        yield cast(BinaryNodeType, self)
        if self._is_node(self.left):
            yield from self.left.inorder()
        if self._is_node(self.right):
            yield from self.right.inorder()

    def postorder(self) -> Iterator[BinaryNodeType]:
        if self._is_node(self.left):
            yield from self.left.inorder()
        if self._is_node(self.right):
            yield from self.right.inorder()
        yield cast(BinaryNodeType, self)

    def level_order(self) -> Iterator[BinaryNodeType]:
        raise NotImplementedError

    def __setitem__(self, key: Comparable, value: Value) -> None:
        raise NotImplementedError("Use insert_node instead")

    def __delitem__(self, key: Comparable) -> None:
        self._delete(key)

    def __getitem__(self, key: Comparable) -> Value:
        raise NotImplementedError("Use access instead")

    def __iter__(self) -> Iterator[Comparable]:
        for node in self.inorder():
            yield node.key

    def children(self) -> Iterator[Union[BinaryNodeType, SentinelType]]:
        yield self.left
        yield self.right


class BinarySearchTreeInternalNode(
    AbstractBinarySearchTreeNode[
        Comparable,
        Value,
        "BinarySearchTreeInternalNode[Comparable, Value]",
        Sentinel[Comparable],
    ]
):
    pass


@dataclass
class AbstractBinarySearchTreeInternalNodeWithParent(
    Generic[Comparable, Value, BinaryNodeWithParentType, SentinelType],
    AbstractBinarySearchTreeNode[
        Comparable,
        Value,
        BinaryNodeWithParentType,
        SentinelType,
    ],
    SupportsParent[BinaryNodeWithParentType, SentinelType],
    ABC,
):
    pass


class BinarySearchTreeInternalNodeWithParent(
    AbstractBinarySearchTreeInternalNodeWithParent[
        Comparable,
        Value,
        "BinarySearchTreeInternalNodeWithParent[Comparable, Value]",
        Sentinel[Comparable],
    ]
):
    pass


if __name__ == "__main__":
    _ = BinarySearchTreeInternalNode(
        0, None, Sentinel[int].default(), Sentinel[int].default()
    )
