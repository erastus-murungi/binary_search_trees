from __future__ import annotations

from abc import ABC
from typing import (
    Any,
    Iterator,
    Optional,
    TypeGuard,
    Union,
)

from typeguard import typechecked  # type: ignore

from core import AbstractTree, Comparable, SentinelReached, SentinelType, Value
from nodes import BinaryNodeType, BinarySearchTreeInternalNode, Sentinel


class BinarySearchTreeAbstract(
    AbstractTree[Comparable, Value, BinaryNodeType, SentinelType], ABC
):
    def __len__(self) -> int:
        return self.size

    def access_no_throw(self, key: Comparable) -> Union[BinaryNodeType, SentinelType]:
        try:
            return self.access(key)
        except SentinelReached:
            return self.sentinel()

    def __contains__(self, key: Any) -> bool:
        return key in self.root

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Comparable]:
        for node in self.inorder():
            yield node.key

    def validate(self, lower_limit: Comparable, upper_limit: Comparable):
        return self.root.validate(lower_limit, upper_limit)

    def pretty_str(self):
        return self.root.pretty_str()

    def __getitem__(self, key: Comparable) -> Value:
        node = self.access(key)
        if self.is_node(node):
            return node.value
        raise KeyError(f"Key {key} not found")

    def __delitem__(self, key: Comparable) -> None:
        self.delete(key)

    def __setitem__(self, key: Comparable, value: Value) -> None:
        self.insert(key, value, allow_overwrite=True)

    @property
    def nonnull_root(self) -> BinaryNodeType:
        if self.is_node(self.root):
            return self.root
        raise SentinelReached("Tree is empty")


class BinarySearchTreeRecursive(
    BinarySearchTreeAbstract[Comparable, Value, BinaryNodeType, SentinelType], ABC
):
    def level_order(self) -> Iterator[BinaryNodeType]:
        try:
            return self.nonnull_root.level_order()
        except SentinelReached:
            return iter([])

    def inorder(self) -> Iterator[BinaryNodeType]:
        try:
            return self.nonnull_root.inorder()
        except SentinelReached:
            return iter([])

    def preorder(self) -> Iterator[BinaryNodeType]:
        try:
            return self.nonnull_root.preorder()
        except SentinelReached:
            return iter([])

    def postorder(self) -> Iterator[BinaryNodeType]:
        try:
            return self.nonnull_root.postorder()
        except SentinelReached:
            return iter([])

    def __init__(self):
        self.root = self.sentinel()
        self.size = 0

    def predecessor(self, key: Comparable) -> BinaryNodeType:
        try:
            return self.nonnull_root.predecessor(key)
        except SentinelReached as e:
            raise KeyError(f"No predecessor found, {key} is minimum key in tree") from e

    def successor(self, key: Comparable) -> BinaryNodeType:
        try:
            return self.nonnull_root.successor(key)
        except SentinelReached as e:
            raise KeyError(f"No successor found, {key} is maximum key in tree") from e

    def access(self, key: Comparable) -> BinaryNodeType:
        return self.nonnull_root.access(key)

    def minimum(self) -> BinaryNodeType:
        return self.nonnull_root.minimum()

    def maximum(self) -> BinaryNodeType:
        return self.nonnull_root.maximum()

    def insert(
        self, key: Comparable, value: Value, allow_overwrite: bool = False
    ) -> BinaryNodeType:
        if self.is_sentinel(self.root):
            node = self.node(key, value)
            self.root = node
        elif key in self.nonnull_root:
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

    def delete(self, key: Comparable) -> Union[BinaryNodeType, SentinelType]:
        if key not in self.root:
            raise ValueError(f"Key {key} not found")
        else:
            self.root = self.nonnull_root.delete(key)
            self.size -= 1
            return self.root

    def extract_min(
        self,
    ) -> tuple[tuple[Comparable, Value], Union[BinaryNodeType, SentinelType]]:
        keyval, self.root = self.nonnull_root.extract_min()
        self.size -= 1
        return keyval, self.root

    def extract_max(
        self,
    ) -> tuple[tuple[Comparable, Value], Union[BinaryNodeType, SentinelType]]:
        keyval, self.root = self.nonnull_root.extract_max()
        self.size -= 1
        return keyval, self.root


class BinarySearchTree(
    BinarySearchTreeRecursive[
        Comparable,
        Value,
        BinarySearchTreeInternalNode[Comparable, Value],
        Sentinel[Comparable],
    ]
):
    def is_sentinel(self, node: Any) -> TypeGuard[Sentinel[Comparable]]:
        return isinstance(node, Sentinel)

    def sentinel(self, *args, **kwargs) -> Sentinel[Comparable]:
        return Sentinel()

    def is_node(
        self, node: Any
    ) -> TypeGuard[BinarySearchTreeInternalNode[Comparable, Value]]:
        return isinstance(node, BinarySearchTreeInternalNode)

    def node(
        self, key: Comparable, value: Value, *args, **kwargs
    ) -> BinarySearchTreeInternalNode[Comparable, Value]:
        return BinarySearchTreeInternalNode(key, value)


class BinarySearchTreeIterativeAbstract(
    BinarySearchTreeAbstract[
        Comparable,
        Value,
        BinaryNodeType,
        SentinelType,
    ],
    ABC,
):
    def access(self, key: Comparable) -> BinaryNodeType:
        current = self.nonnull_root
        while self.is_node(current):
            if current.key == key:
                return current
            elif key < current.key:
                current = current.nonnull_left
            else:
                current = current.nonnull_right
        return current

    def access_ancestry(self, key: Comparable) -> Optional[list[BinaryNodeType]]:
        ancestry = []
        current = self.root
        while self.is_node(current):
            ancestry.append(current)
            if current.key == key:
                return ancestry
            elif current.key < key:
                current = current.left
            else:
                current = current.right
        return None

    def set_child(
        self,
        ancestry: list[BinaryNodeType],
        node: BinaryNodeType,
    ):
        if ancestry:
            ancestry[-1].choose_set(node.key, node)
        else:
            self.root = node

    def minimum(self) -> BinaryNodeType:
        current = self.nonnull_root
        while self.is_node(current.left):
            current = current.left
        return current

    def maximum(self) -> BinaryNodeType:
        current = self.nonnull_root
        while self.is_node(current.right):
            current = current.right
        return current

    def insert(
        self, key: Comparable, value: Value, allow_overwrite=True
    ) -> BinaryNodeType:
        if self.is_sentinel(self.root):
            self.root = self.node(key, value)
        else:
            node = self.nonnull_root
            while True:
                if node.key == key:
                    if allow_overwrite:
                        node.value = value
                        return self.nonnull_root
                    else:
                        raise ValueError(f"Key {key} already in tree")
                elif node.key < key:
                    if self.is_node(node.right):
                        node = node.right
                    else:
                        node.right = self.node(key, value)
                        break
                else:
                    if self.is_node(node.left):
                        node = node.left
                    else:
                        node.left = self.node(key, value)
                        break
        self.size += 1
        return self.nonnull_root

    def insert_ancestry(self, key: Comparable, value: Value) -> list[BinaryNodeType]:
        if self.is_node(self.root):
            ancestry = []
            node = self.root
            while True:
                if node.key == key:
                    raise ValueError(f"Key {key} already in tree")
                elif node.key < key:
                    if self.is_node(node.right):
                        node = node.right
                    else:
                        node.right = self.node(key, value)
                        ancestry.append(node)
                        ancestry.append(node.right)
                        break
                else:
                    if self.is_node(node.left):
                        node = node.left
                    else:
                        node.left = self.node(key, value)
                        ancestry.append(node)
                        ancestry.append(node.left)
                        break
        else:
            node = self.node(key, value)
            self.root = node
            ancestry = [node]

        self.size += 1
        return ancestry

    def parent(self, node: BinaryNodeType) -> Union[BinaryNodeType, SentinelType]:
        current = self.root
        parent: Union[BinaryNodeType, SentinelType] = self.sentinel()
        while self.is_node(current) and current is not node:
            parent = current
            current = current.choose(node.key)
        return parent

    def transplant(
        self,
        target: BinaryNodeType,
        replacement: Union[BinaryNodeType, SentinelType],
        prev: Optional[Union[BinaryNodeType, SentinelType]] = None,
    ):
        if prev is None:
            parent = self.parent(target)
        else:
            parent = prev
        if self.is_node(parent):
            parent.choose_set(target.key, replacement)
        else:
            self.root = replacement

    def delete(self, key: Comparable) -> Union[BinaryNodeType, SentinelType]:
        try:
            target_node = self.access(key)
            if self.is_sentinel(target_node.left):
                self.transplant(target_node, target_node.right)
            elif self.is_sentinel(target_node.right):
                self.transplant(target_node, target_node.left)
            else:
                successor = target_node.nonnull_right.minimum()
                if successor != target_node.right:
                    self.transplant(successor, successor.right)
                    successor.right = target_node.right
                self.transplant(target_node, successor)
                successor.left = target_node.left
            self.size -= 1
            return self.root
        except SentinelReached as e:
            raise ValueError(f"Key {key} not in tree") from e

    def extract_min(
        self,
    ) -> tuple[tuple[Comparable, Value], Union[BinaryNodeType, SentinelType]]:
        root = self.nonnull_root
        prev: Union[BinaryNodeType, SentinelType] = self.sentinel()

        while self.is_node(root.left):
            prev = root
            root = root.left

        keyval = (root.key, root.value)
        self.transplant(root, root.right, prev)
        self.size -= 1
        return keyval, self.root

    def extract_max(
        self,
    ) -> tuple[tuple[Comparable, Value], Union[BinaryNodeType, SentinelType]]:
        root = self.nonnull_root
        prev: Union[BinaryNodeType, SentinelType] = self.sentinel()

        while self.is_node(root.right):
            prev = root
            root = root.right

        keyval = (root.key, root.value)
        self.transplant(root, root.left, prev)
        self.size -= 1
        return keyval, self.root

    def inorder(self) -> Iterator[BinaryNodeType]:
        if self.is_node(self.root):
            stack: list[tuple[bool, BinaryNodeType]] = [(False, self.root)]
            while stack:
                visited, node = stack.pop()
                if visited:
                    yield node
                else:
                    if self.is_node(node.right):
                        stack.append((False, node.right))
                    stack.append((True, node))
                    if self.is_node(node.left):
                        stack.append((False, node.left))

    def preorder(self) -> Iterator[BinaryNodeType]:
        if self.is_node(self.root):
            stack: list[BinaryNodeType] = [self.root]
            while stack:
                node = stack.pop()
                yield node
                if self.is_node(node.right):
                    stack.append(node.right)
                if self.is_node(node.left):
                    stack.append(node.left)

    def postorder(self) -> Iterator[BinaryNodeType]:
        if self.is_node(self.root):
            stack: list[tuple[bool, BinaryNodeType]] = [(False, self.root)]
            while stack:
                visited, node = stack.pop()
                if visited:
                    yield node
                else:
                    stack.append((True, node))
                    if self.is_node(node.right):
                        stack.append((False, node.right))
                    if self.is_node(node.left):
                        stack.append((False, node.left))

    def level_order(self) -> Iterator[BinaryNodeType]:
        raise NotImplementedError()

    def successor(self, key: Comparable) -> BinaryNodeType:
        try:
            node = self.access(key)
            if self.is_node(node.right):
                return node.right.minimum()
            else:
                candidate: Optional[BinaryNodeType] = None
                current = self.root
                while self.is_node(current) and current is not node:
                    if current.key > node.key:
                        candidate = current
                        current = current.left
                    else:
                        current = current.right
            if candidate is None:
                raise KeyError(f"Key {key} has no successor")
            return candidate
        except SentinelReached as e:
            raise ValueError(f"Key {key} not found") from e

    def predecessor(self, key: Comparable) -> BinaryNodeType:
        try:
            node = self.access(key)
            if self.is_node(node.left):
                return node.left.maximum()
            else:
                candidate: Optional[BinaryNodeType] = None
                current = self.root
                while self.is_node(current) and current is not node:
                    if current.key < node.key:
                        candidate = current
                        current = current.right
                    else:
                        current = current.left
            if candidate is None:
                raise KeyError(f"Key {key} has no predecessor")
            return candidate
        except SentinelReached as e:
            raise ValueError(f"Key {key} not found") from e

    #
    # @typechecked
    # def insert_parent(
    #     self, key: Comparable, value: Value, aux: Optional[AuxiliaryData] = None
    # ) -> tuple[Node, InternalNode]:
    #     node = self.root
    #     parent: Node = Leaf()
    #
    #     while True:
    #         if isinstance(node, Leaf):
    #             node = Internal(key, value, aux=aux)
    #             if isinstance(parent, Internal):
    #                 parent.choose_set(node.key, node)
    #             else:
    #                 self.root = node
    #             break
    #         else:
    #             assert isinstance(node, Internal)
    #             if node.key == key:
    #                 raise ValueError(f"Key {key} already in tree")
    #             else:
    #                 parent = node
    #                 node = node.choose(key)
    #     self.size += 1
    #     return parent, node
    #
    #
    # def access_ancestry_min(self, node: Internal):
    #     current: Internal = node
    #     ancestry = self.access_ancestry(node.key)
    #     assert ancestry is not None
    #     while isinstance(current.left, Internal):
    #         current = current.left
    #         ancestry.append(current)
    #     return ancestry
    #
    # def access_ancestry_max(self, node: InternalNode):
    #     current: Internal = node
    #     ancestry = self.access_ancestry(node.key)
    #     assert ancestry is not None
    #     while isinstance(current.right, Internal):
    #         current = current.right
    #         ancestry.append(current)
    #     return ancestry


class BinarySearchTreeIterative(
    BinarySearchTreeIterativeAbstract[
        Comparable,
        Value,
        BinarySearchTreeInternalNode[Comparable, Value],
        Sentinel[Comparable],
    ]
):
    def __init__(
        self,
        root: Optional[
            Union[BinarySearchTreeInternalNode[Comparable, Value], Sentinel]
        ] = None,
        size=0,
    ):
        root = Sentinel.default() if root is None else root
        super().__init__(root, size)

    def is_sentinel(self, node: Any) -> TypeGuard[Sentinel[Comparable]]:
        return isinstance(node, Sentinel)

    def sentinel(self, *args, **kwargs) -> Sentinel[Comparable]:
        return Sentinel()

    def is_node(
        self, node: Any
    ) -> TypeGuard[BinarySearchTreeInternalNode[Comparable, Value]]:
        return isinstance(node, BinarySearchTreeInternalNode)

    def node(
        self, key: Comparable, value: Value, *args, **kwargs
    ) -> BinarySearchTreeInternalNode[Comparable, Value]:
        return BinarySearchTreeInternalNode(key, value)

    def access_no_throw(
        self, key: Comparable
    ) -> Union[BinarySearchTreeInternalNode[Comparable, Value], Sentinel[Comparable]]:
        try:
            return self.access(key)
        except SentinelReached:
            return Sentinel.default()


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
    pass
