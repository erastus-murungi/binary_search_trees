from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, TypeGuard, Union

from typeguard import typechecked  # type: ignore

from core import AbstractTree, Comparable, SentinelReached, SentinelType, Value
from nodes import (
    BinaryNodeType,
    BinaryNodeWithParentType,
    BinarySearchTreeInternalNode,
    BinarySearchTreeInternalNodeWithParent,
    Sentinel,
)


class AbstractBinarySearchTree(
    AbstractTree[Comparable, Value, BinaryNodeType, SentinelType], ABC
):
    def __contains__(self, key: Any) -> bool:
        return key in self.root

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
        raise SentinelReached(f"Tree is empty")

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


class AbstractBinarySearchTreeWithParent(
    AbstractBinarySearchTree[Comparable, Value, BinaryNodeWithParentType, SentinelType],
    ABC,
):
    def left_rotate(
        self,
        node: BinaryNodeWithParentType,
        update: Optional[
            Callable[[Union[BinaryNodeWithParentType, SentinelType]], None]
        ] = None,
    ) -> BinaryNodeWithParentType:

        assert self.is_node(node)
        assert self.is_node(node.right)

        right_child = node.right
        node.right = right_child.left

        if self.is_node(right_child.left):
            right_child.left.parent = node

        right_child.parent = node.parent

        if self.is_sentinel(node.parent):
            self.root = right_child

        elif node is node.parent.left:
            node.parent.left = right_child

        else:
            node.parent.right = right_child

        right_child.left = node
        node.parent = right_child

        if update:
            update(node)
            update(right_child)

        return right_child

    def right_rotate(
        self,
        node: BinaryNodeWithParentType,
        update: Optional[
            Callable[[Union[BinaryNodeWithParentType, SentinelType]], None]
        ] = None,
    ) -> BinaryNodeWithParentType:

        assert self.is_node(node)
        assert self.is_node(node.left)

        left_child = node.left
        node.left = left_child.right

        if self.is_node(left_child.right):
            left_child.right.parent = node

        left_child.parent = node.parent

        if self.is_sentinel(node.parent):
            self.root = left_child

        elif node is node.parent.right:
            node.parent.right = left_child

        else:
            node.parent.left = left_child

        left_child.right = node
        node.parent = left_child

        if update:
            update(node)
            update(left_child)

        return left_child


class AbstractBinarySearchTreeRecursive(
    AbstractBinarySearchTree[Comparable, Value, BinaryNodeType, SentinelType], ABC
):
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
            node = self.nonnull_root._insert(key, value, allow_overwrite)
            self.root = node
        self.size += 1
        return node

    def delete(self, key: Comparable) -> BinaryNodeType:
        if node := self.access(key):
            self.root = self.nonnull_root._delete(key)
            self.size -= 1
            return node
        else:
            raise KeyError(f"Key {key} not found")

    def extract_min(
        self,
    ) -> tuple[Comparable, Value]:
        keyval, self.root = self.nonnull_root._extract_min()
        self.size -= 1
        return keyval

    def extract_max(
        self,
    ) -> tuple[Comparable, Value]:
        keyval, self.root = self.nonnull_root._extract_max()
        self.size -= 1
        return keyval

    def minimum(self) -> BinaryNodeType:
        return self.nonnull_root.minimum()

    def maximum(self) -> BinaryNodeType:
        return self.nonnull_root.maximum()

    def right_rotate(
        self,
        node: BinaryNodeType,
        update: Optional[Callable[[Union[BinaryNodeType, SentinelType]], None]] = None,
    ) -> BinaryNodeType:
        """
                 y                              x
               // \\                          // \\
              x   Ω  = {right_rotate(y)} =>  𝛼   y
            // \\                              // \\
            𝛼   ß                              ß   Ω
        """
        assert self.is_node(node)
        assert self.is_node(node.left)

        left_child = node.left

        node.left = left_child.right
        left_child.right = node

        if update:
            update(node)
            update(left_child)

        return left_child

    def left_rotate(
        self,
        node: BinaryNodeType,
        update: Optional[Callable[[Union[BinaryNodeType, SentinelType]], None]] = None,
    ) -> BinaryNodeType:
        assert self.is_node(node)
        assert self.is_node(node.right)

        right_child = node.right

        node.right = right_child.left
        right_child.left = node

        if update:
            update(node)
            update(right_child)

        return right_child


class AbstractBinarySearchTreeWithParentRecursive(
    AbstractBinarySearchTreeRecursive[
        Comparable, Value, BinaryNodeWithParentType, SentinelType
    ],
    ABC,
):
    pass


class AbstractBinarySearchTreeIterative(
    AbstractBinarySearchTree[
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

    def delete(self, key: Comparable) -> BinaryNodeType:
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
            return target_node
        except SentinelReached as e:
            raise ValueError(f"Key {key} not in tree") from e

    def extract_min(self) -> tuple[Comparable, Value]:
        root = self.nonnull_root
        prev: Union[BinaryNodeType, SentinelType] = self.sentinel()

        while self.is_node(root.left):
            prev = root
            root = root.left

        keyval = (root.key, root.value)
        self.transplant(root, root.right, prev)
        self.size -= 1
        return keyval

    def extract_max(self) -> tuple[Comparable, Value]:
        root = self.nonnull_root
        prev: Union[BinaryNodeType, SentinelType] = self.sentinel()

        while self.is_node(root.right):
            prev = root
            root = root.right

        keyval = (root.key, root.value)
        self.transplant(root, root.left, prev)
        self.size -= 1
        return keyval

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

    def right_rotate(
        self,
        node: BinaryNodeType,
        update: Optional[Callable[[Union[BinaryNodeType, SentinelType]], None]] = None,
    ) -> BinaryNodeType:
        """
                 y                              x
               // \\                          // \\
              x   Ω  = {right_rotate(y)} =>  𝛼   y
            // \\                              // \\
            𝛼   ß                              ß   Ω
        """
        assert self.is_node(node)
        assert self.is_node(node.left)

        left_child = node.left

        node.left = left_child.right
        left_child.right = node

        if update:
            update(node)
            update(left_child)

        return left_child

    def left_rotate(
        self,
        node: BinaryNodeType,
        update: Optional[Callable[[Union[BinaryNodeType, SentinelType]], None]] = None,
    ) -> BinaryNodeType:

        assert self.is_node(node)
        assert self.is_node(node.right)

        right_child = node.right

        node.right = right_child.left
        right_child.left = node

        if update:
            update(node)
            update(right_child)

        return right_child


class AbstractBinarySearchTreeWithParentIterative(
    AbstractBinarySearchTreeIterative[
        Comparable, Value, BinaryNodeWithParentType, SentinelType
    ],
    ABC,
):
    def insert(
        self, key: Comparable, value: Value, allow_overwrite: bool = True
    ) -> BinaryNodeWithParentType:
        x = self.root
        y: Union[BinaryNodeWithParentType, SentinelType] = self.sentinel()

        while self.is_node(x):
            y = x
            if key < x.key:
                x = x.left
            elif key > x.key:
                x = x.right
            elif allow_overwrite:
                x.value = value
                return x
            else:
                raise ValueError(f"Key {key} already in tree")
        z = self.node(key, value, parent=y)
        if self.is_node(y):
            if key < y.key:
                y.left = z
            else:
                y.right = z
        else:
            self.root = z
        self.size += 1
        return z

    def transplant(
        self,
        u: BinaryNodeWithParentType,
        v: Union[BinaryNodeWithParentType, SentinelType],
        _: Optional[Union[BinaryNodeWithParentType, SentinelType]] = None,
    ) -> None:
        if self.is_sentinel(u.parent):
            self.root = v
        else:
            assert self.is_node(u.parent)
            if u is u.parent.left:
                u.parent.left = v
            else:
                u.parent.right = v
        if self.is_node(v):
            v.parent = u.parent

    def delete(self, key: Comparable) -> BinaryNodeWithParentType:
        try:
            node = self.access(key)
            if self.is_sentinel(node.left):
                self.transplant(node, node.right)
            elif self.is_sentinel(node.right):
                self.transplant(node, node.left)
            else:
                successor = node.right.minimum()
                if successor is not node.right:
                    self.transplant(successor, successor.right)
                    successor.right = node.right
                    successor.right.parent = successor
                self.transplant(node, successor)
                successor.left = node.left
                successor.left.parent = successor
            self.size -= 1
            return node
        except SentinelReached as e:
            raise ValueError(f"Key {key} not in tree") from e

    def left_rotate(
        self,
        node: BinaryNodeWithParentType,
        update: Optional[
            Callable[[Union[BinaryNodeWithParentType, SentinelType]], None]
        ] = None,
    ) -> BinaryNodeWithParentType:

        assert self.is_node(node)
        assert self.is_node(node.right)

        right_child = node.right
        node.right = right_child.left

        if self.is_node(right_child.left):
            right_child.left.parent = node

        right_child.parent = node.parent

        if self.is_sentinel(node.parent):
            self.root = right_child

        elif node is node.parent.left:
            node.parent.left = right_child

        else:
            node.parent.right = right_child

        right_child.left = node
        node.parent = right_child

        if update:
            update(node)
            update(right_child)

        return right_child

    def right_rotate(
        self,
        node: BinaryNodeWithParentType,
        update: Optional[
            Callable[[Union[BinaryNodeWithParentType, SentinelType]], None]
        ] = None,
    ) -> BinaryNodeWithParentType:

        assert self.is_node(node)
        assert self.is_node(node.left)

        left_child = node.left
        node.left = left_child.right

        if self.is_node(left_child.right):
            left_child.right.parent = node

        left_child.parent = node.parent

        if self.is_sentinel(node.parent):
            self.root = left_child

        elif node is node.parent.right:
            node.parent.right = left_child

        else:
            node.parent.left = left_child

        left_child.right = node
        node.parent = left_child

        if update:
            update(node)
            update(left_child)

        return left_child


@dataclass
class BinarySearchTreeRecursive(
    AbstractBinarySearchTreeRecursive[
        Comparable,
        Value,
        BinarySearchTreeInternalNode[Comparable, Value],
        Sentinel[Comparable],
    ]
):
    root: Union[
        BinarySearchTreeInternalNode[Comparable, Value], Sentinel[Comparable]
    ] = field(default_factory=Sentinel[Comparable].default)
    size: int = 0

    def is_sentinel(self, node: Any) -> TypeGuard[Sentinel[Comparable]]:
        return isinstance(node, Sentinel)

    def sentinel(self, *args, **kwargs) -> Sentinel[Comparable]:
        return Sentinel()

    def is_node(
        self, node: Any
    ) -> TypeGuard[BinarySearchTreeInternalNode[Comparable, Value]]:
        return isinstance(node, BinarySearchTreeInternalNode)

    def node(
        self,
        key: Comparable,
        value: Value,
        left: Union[
            BinarySearchTreeInternalNode[Comparable, Value], Sentinel[Comparable]
        ] = Sentinel.default(),
        right: Union[
            BinarySearchTreeInternalNode[Comparable, Value], Sentinel[Comparable]
        ] = Sentinel.default(),
        *args,
        **kwargs,
    ) -> BinarySearchTreeInternalNode[Comparable, Value]:
        return BinarySearchTreeInternalNode(key, value, left, right)


class BinarySearchTreeIterative(
    AbstractBinarySearchTreeIterative[
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
        self,
        key: Comparable,
        value: Value,
        left: Union[
            BinarySearchTreeInternalNode[Comparable, Value], Sentinel[Comparable]
        ] = Sentinel.default(),
        right: Union[
            BinarySearchTreeInternalNode[Comparable, Value], Sentinel[Comparable]
        ] = Sentinel.default(),
        *args,
        **kwargs,
    ) -> BinarySearchTreeInternalNode[Comparable, Value]:
        return BinarySearchTreeInternalNode(key, value, left, right)


class BinarySearchTreeIterativeWithParent(
    AbstractBinarySearchTreeWithParent[
        Comparable,
        Value,
        BinarySearchTreeInternalNodeWithParent[Comparable, Value],
        Sentinel[Comparable],
    ],
    AbstractBinarySearchTreeIterative[
        Comparable,
        Value,
        BinarySearchTreeInternalNodeWithParent[Comparable, Value],
        Sentinel[Comparable],
    ],
):
    def __init__(
        self,
        root: Optional[
            BinarySearchTreeInternalNodeWithParent[Comparable, Value]
        ] = None,
        size=0,
    ):
        super().__init__(Sentinel[Comparable].default() if root is None else root, size)

    def is_sentinel(self, node: Any) -> TypeGuard[Sentinel[Comparable]]:
        return isinstance(node, Sentinel)

    def sentinel(self, *args, **kwargs) -> Sentinel[Comparable]:
        return Sentinel()

    def is_node(
        self, node: Any
    ) -> TypeGuard[BinarySearchTreeInternalNodeWithParent[Comparable, Value]]:
        return isinstance(node, BinarySearchTreeInternalNodeWithParent)

    def node(
        self,
        key: Comparable,
        value: Value,
        left: Union[
            BinarySearchTreeInternalNodeWithParent[Comparable, Value],
            Sentinel[Comparable],
        ] = Sentinel.default(),
        right: Union[
            BinarySearchTreeInternalNodeWithParent[Comparable, Value],
            Sentinel[Comparable],
        ] = Sentinel.default(),
        parent: Union[
            BinarySearchTreeInternalNodeWithParent[Comparable, Value],
            Sentinel[Comparable],
        ] = Sentinel.default(),
        *args,
        **kwargs,
    ) -> BinarySearchTreeInternalNodeWithParent[Comparable, Value]:
        return BinarySearchTreeInternalNodeWithParent(
            key=key, value=value, left=left, right=right, parent=parent
        )


if __name__ == "__main__":
    from random import randint

    # for _ in range(10000):
    #     bst: BinarySearchTreeRecursive[int, None] = BinarySearchTreeRecursive[
    #         int, None
    #     ]()
    #     num_values = 3
    #     # values = list({randint(0, 100) for _ in range(num_values)})
    #     # values = [35, 70, 51, 52, 20, 55, 91]
    #     values = [50, 44, 94]
    #
    #     for i, val in enumerate(values):
    #         bst.insert(val, None, allow_overwrite=True)
    #         bst.validate(0, 1000)
    #         assert val in bst
    #
    #     for i, val in enumerate(values):
    #         bst.delete(val)
    #         print(bst.pretty_str())
    #         assert val not in bst, values
