from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (Any, Generic, Hashable, Iterator, Optional, Self,
                    TypeGuard, TypeVar, Union)

from typeguard import typechecked  # type: ignore

from core import (AbstractTree, Comparable, NodeType, SentinelReached,
                  SentinelType, Value)
from nodes import BinarySearchTreeInternalNode, Sentinel


class BinarySearchTreeAbstract(
    AbstractTree[Comparable, Value, NodeType, SentinelType], ABC
):
    @property
    def nonnull_root(self) -> NodeType:
        if self.is_node(self.root):
            return self.root
        raise SentinelReached("Tree is empty")

    def inorder(self) -> Iterator[NodeType]:
        try:
            return self.nonnull_root.inorder()
        except SentinelReached:
            return iter([])

    def preorder(self) -> Iterator[NodeType]:
        try:
            return self.nonnull_root.preorder()
        except SentinelReached:
            return iter([])

    def postorder(self) -> Iterator[NodeType]:
        try:
            return self.nonnull_root.postorder()
        except SentinelReached:
            return iter([])

    def level_order(self) -> Iterator[NodeType]:
        try:
            return self.nonnull_root.level_order()
        except SentinelReached:
            return iter([])

    def __len__(self) -> int:
        return self.size

    def __contains__(self, key: Any) -> bool:
        return key in self.root

    def minimum(self) -> NodeType:
        return self.nonnull_root.minimum()

    def maximum(self) -> NodeType:
        return self.nonnull_root.maximum()

    def insert(
        self, key: Comparable, value: Value, allow_overwrite: bool = False
    ) -> NodeType:
        if self.is_leaf(self.root):
            node = self.node(key, value)
            self.root = node
        elif key in self.root:
            if allow_overwrite:
                self.nonnull_root[key] = value
                return self.nonnull_root
            else:
                raise KeyError(f"Key {key} already exists")
        else:
            node = self.nonnull_root.insert(key, value, allow_overwrite)
            self.root = node
        self.size += 1
        return node

    def access(self, key: Any) -> Union[NodeType, SentinelType]:
        if self.is_node(self.root):
            return self.nonnull_root.access(key)
        return self.root

    def delete(self, key: Comparable) -> Union[NodeType, SentinelType]:
        if key not in self.root:
            raise ValueError(f"Key {key} not found")
        else:
            self.root = self.nonnull_root.delete(key)
            self.size -= 1
            return self.root

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Comparable]:
        for node in self.inorder():
            yield node.key

    def validate(self, lower_limit: Comparable, upper_limit: Comparable):
        return self.root.validate(lower_limit, upper_limit)

    def predecessor(self, key: Comparable) -> NodeType:
        try:
            return self.nonnull_root.predecessor(key)
        except SentinelReached as e:
            raise KeyError(f"No predecessor found, {key} is minimum key in tree") from e

    def successor(self, key: Comparable) -> NodeType:
        try:
            return self.nonnull_root.successor(key)
        except SentinelReached as e:
            raise KeyError(f"No successor found, {key} is maximum key in tree") from e

    def pretty_str(self):
        return self.root.pretty_str()

    def extract_min(
        self,
    ) -> tuple[tuple[Comparable, Value], Union[NodeType, SentinelType]]:
        keyval, self.root = self.nonnull_root.extract_min()
        self.size -= 1
        return keyval, self.root

    def extract_max(
        self,
    ) -> tuple[tuple[Comparable, Value], Union[NodeType, SentinelType]]:
        keyval, self.root = self.nonnull_root.extract_max()
        self.size -= 1
        return keyval, self.root

    def __getitem__(self, key: Comparable) -> Value:
        node = self.access(key)
        if self.is_node(node):
            return node.value
        raise KeyError(f"Key {key} not found")

    def __delitem__(self, key: Comparable) -> None:
        self.delete(key)

    def __setitem__(self, key: Comparable, value: Value) -> None:
        self.insert(key, value)


class BinarySearchTree(
    BinarySearchTreeAbstract[Comparable, Value, BinarySearchTreeInternalNode, Sentinel]
):
    def __init__(self):
        self.root = self.leaf()
        self.size = 0

    def is_leaf(self, node: Any) -> TypeGuard[Sentinel[Comparable]]:
        return isinstance(node, Sentinel)

    def leaf(self, *args, **kwargs) -> Sentinel[Comparable]:
        return Sentinel()

    def is_node(
        self, node: Any
    ) -> TypeGuard[BinarySearchTreeInternalNode[Comparable, Value]]:
        return isinstance(node, BinarySearchTreeInternalNode)

    def node(
        self, key: Comparable, value: Value, *args, **kwargs
    ) -> BinarySearchTreeInternalNode[Comparable, Value]:
        return BinarySearchTreeInternalNode(key, value)


# @dataclass
# class BinarySearchTreeIterative(
#     BinarySearchTree[Comparable, Value, AuxiliaryData],
# ):
#     root: Node = field(default_factory=Leaf)
#     size: int = 0
#
#     def __post_init__(self):
#         assert len(self.root) == self.size
#
#     def __setitem__(self, key, value):
#         self.insert(key, value)
#
#     def level_order(self) -> Iterator[InternalNode]:
#         return self.root.level_order()
#
#     def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
#         raise NotImplementedError
#
#     def check_invariants(self, lower_limit: Comparable, upper_limit: Comparable):
#         return self.root.check_invariants(lower_limit, upper_limit)
#
#     def access(self, key: Comparable) -> Node:
#         x = self.root
#         while isinstance(x, Internal):
#             if x.key == key:
#                 return x
#             elif x.key < key:
#                 x = x.right
#             else:
#                 x = x.left
#         return x
#
#     def access_ancestry(self, key: Comparable) -> Optional[list[Internal]]:
#         ancestry = []
#         x = self.root
#         while isinstance(x, Internal):
#             ancestry.append(x)
#             if x.key == key:
#                 return ancestry
#             elif x.key < key:
#                 x = x.right
#             else:
#                 x = x.left
#         return None
#
#     def set_child(
#         self,
#         ancestry: list[InternalNode],
#         node: InternalNode,
#     ):
#         if ancestry:
#             ancestry[-1].choose_set(node.key, node)
#         else:
#             self.root = node
#
#     def insert_ancestry(
#         self, key: Comparable, value: Value, aux: Optional[AuxiliaryData] = None
#     ) -> list[InternalNode]:
#         ancestry: list[InternalNode] = []
#         node = self.root
#
#         while True:
#             if isinstance(node, Leaf):
#                 node = Internal(key, value, aux=aux)
#                 self.set_child(ancestry, node)
#                 ancestry.append(node)
#                 break
#             else:
#                 assert isinstance(node, Internal)
#                 if node.key == key:
#                     raise ValueError(f"Key {key} already in tree")
#                 else:
#                     ancestry.append(node)
#                     node = node.choose(key)
#
#         self.size += 1
#         return ancestry
#
#     @typechecked
#     def insert_parent(
#         self, key: Comparable, value: Value, aux: Optional[AuxiliaryData] = None
#     ) -> tuple[Node, InternalNode]:
#         node = self.root
#         parent: Node = Leaf()
#
#         while True:
#             if isinstance(node, Leaf):
#                 node = Internal(key, value, aux=aux)
#                 if isinstance(parent, Internal):
#                     parent.choose_set(node.key, node)
#                 else:
#                     self.root = node
#                 break
#             else:
#                 assert isinstance(node, Internal)
#                 if node.key == key:
#                     raise ValueError(f"Key {key} already in tree")
#                 else:
#                     parent = node
#                     node = node.choose(key)
#         self.size += 1
#         return parent, node
#
#     def clear(self):
#         self.root = Leaf()
#         self.size = 0
#
#     def __len__(self):
#         return self.size
#
#     def pretty_str(self):
#         return "".join(self.root.yield_line("", "R"))
#
#     @typechecked
#     def parent(self, node: InternalNode) -> Node:
#         current: Node = self.root
#         parent: Node = Leaf()
#         while isinstance(current, Internal) and current is not node:
#             parent = current
#             current = current.choose(node.key)
#         return parent
#
#     def __iter__(self):
#         yield from iter(self.root)
#
#     def insert_impl(
#         self, key: Comparable, value: Value, aux: Optional[AuxiliaryData] = None
#     ) -> InternalNode:
#         node: Node = self.root
#         parent: Node = Leaf()
#
#         while True:
#             if isinstance(node, Leaf):
#                 node = Internal(key, value, aux=aux)
#                 if isinstance(parent, Internal):
#                     parent.choose_set(key, node)
#                 else:
#                     self.root = node
#                 break
#             else:
#                 assert isinstance(node, Internal)
#                 if node.key == key:
#                     raise ValueError(f"Key {key} already in tree")
#                 else:
#                     parent = node
#                     node = node.choose(key)
#         self.size += 1
#         assert isinstance(node, Internal)
#         return node
#
#     def inorder(self) -> Iterator[Internal]:
#         yield from self.root.inorder()
#
#     def preorder(self) -> Iterator[Internal]:
#         yield from self.root.preorder()
#
#     def postorder(self) -> Iterator[Internal]:
#         yield from self.root.postorder()
#
#     def minimum(self):
#         if isinstance(self.root, Leaf):
#             raise ValueError("Empty tree has no minimum")
#         current = self.root
#         while isinstance(current.left, Internal):
#             current = current.left
#         return current
#
#     def maximum(self):
#         if isinstance(self.root, Leaf):
#             raise ValueError("Empty tree has no maximum")
#         current = self.root
#         while isinstance(current.right, Internal):
#             current = current.right
#         return current
#
#     def access_ancestry_min(self, node: Internal):
#         current: Internal = node
#         ancestry = self.access_ancestry(node.key)
#         assert ancestry is not None
#         while isinstance(current.left, Internal):
#             current = current.left
#             ancestry.append(current)
#         return ancestry
#
#     def access_ancestry_max(self, node: InternalNode):
#         current: Internal = node
#         ancestry = self.access_ancestry(node.key)
#         assert ancestry is not None
#         while isinstance(current.right, Internal):
#             current = current.right
#             ancestry.append(current)
#         return ancestry
#
#     @typechecked
#     def extract_min_iterative(self, node: Internal) -> KeyValue:
#         current: Internal = node
#         parent: Node = self.parent(node)
#         while isinstance(current.left, Internal):
#             parent, current = current, current.left
#         min_key_value = (current.key, current.value)
#
#         if isinstance(current.right, Internal):
#             current.swap(current.right)
#         elif current is self.root and current.has_no_children:
#             assert self.size == 1
#             self.reset_root()
#         else:
#             assert isinstance(parent, Internal)
#             parent.choose_set(current.key, Leaf())
#
#         return min_key_value
#
#     @typechecked
#     def successor(self, key: Comparable) -> Internal:
#         node = self.access(key)
#         if isinstance(node, Internal):
#             if isinstance(node.right, Internal):
#                 return node.right.minimum()
#             else:
#                 candidate: Optional[Internal] = None
#                 current = self.root
#                 while isinstance(current, Internal) and current != node:
#                     if current.key > node.key:
#                         candidate = current
#                         current = current.left
#                     else:
#                         current = current.right
#             if candidate is None:
#                 raise KeyError(f"Key {key} has no successor")
#             return candidate
#         else:
#             raise ValueError(f"Key {key} not found")
#
#     @typechecked
#     def predecessor(self, key: Comparable) -> Internal:
#         node = self.access(key)
#         if isinstance(node, Internal):
#             if isinstance(node.left, Internal):
#                 return node.left.maximum()
#             else:
#                 candidate: Optional[Internal] = None
#                 current = self.root
#                 while isinstance(current, Internal) and current != node:
#                     if current.key < node.key:
#                         candidate = current
#                         current = current.right
#                     else:
#                         current = current.left
#             if candidate is None:
#                 raise KeyError(f"Key {key} has no predecessor")
#             return candidate
#         else:
#             raise ValueError(f"Key {key} not found")
#
#     def delete(self, target_key: Comparable):
#         target_node = self.access(target_key)
#         if isinstance(target_node, Internal):
#             if isinstance(target_node.right, Internal) and isinstance(
#                 target_node.left, Internal
#             ):
#                 # swap key with successor and delete using extract_min
#                 key, value = self.extract_min_iterative(target_node.right)
#                 target_node.key, target_node.value = key, value
#
#             elif isinstance(target_node.right, Internal):
#                 target_node.swap(target_node.right)
#
#             elif isinstance(target_node.left, Internal):
#                 target_node.swap(target_node.left)
#             else:
#                 # the target node is the root, and it is a leaf node, then setting the root to None will suffice
#                 if target_node == self.root:
#                     self.reset_root()
#                 else:
#                     assert target_node.has_no_children
#                     parent = self.parent(target_node)
#                     assert isinstance(parent, Internal)
#                     # else the target is a leaf node which has a parent
#                     parent.choose_set(target_node.key, Leaf())
#
#             self.size -= 1
#             return self.root
#         else:
#             raise ValueError(f"Key {target_key} not found")
#
#     def reset_root(self):
#         self.root = Leaf()
#
#     def __repr__(self):
#         return repr(self.root)


if __name__ == "__main__":
    # bst: BinarySearchTree = BinarySearchTree()
    # values = [
    #     3,
    #     52,
    #     31,
    #     55,
    #     93,
    #     60,
    #     81,
    #     93,
    #     46,
    #     37,
    #     47,
    #     67,
    #     34,
    #     95,
    #     10,
    #     23,
    #     90,
    #     14,
    #     13,
    #     88,
    # ]
    #
    # for i, val in enumerate(values):
    #     bst.insert(val, None, allow_overwrite=True)
    #     print(values[i] in bst)
    #     assert 101 not in bst
    # print(bst.pretty_str())

    from random import randint
    from sys import maxsize

    for _ in range(50):
        values = frozenset([randint(-10000, 10000) for _ in range(100)])
        bst: BinarySearchTree = BinarySearchTree()
        for val in values:
            bst.insert(val, None)
            assert val in bst
            bst.validate(-maxsize, maxsize)

        for val in values:
            del bst[val]
            assert val not in bst
            bst.validate(-maxsize, maxsize)

        assert not bst
        assert len(bst) == 0
        assert isinstance(bst.root, Sentinel)
