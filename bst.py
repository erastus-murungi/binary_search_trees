from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, TypeGuard, Union

from core import Key, SentinelReferenceError, SentinelType, Tree, Value
from nodes import (
    BinaryNodeType,
    BinaryNodeWithParentType,
    BSTNode,
    BSTNodeWithParent,
    Sentinel,
    draw_tree,
)


class AbstractBST(Tree[Key, Value, BinaryNodeType, SentinelType], ABC):
    root: Union[BinaryNodeType, SentinelType]

    def __contains__(self, key: Any) -> bool:
        return key in self.root

    def __iter__(self) -> Iterator[Key]:
        for node in self.inorder():
            yield node.key

    def validate(self, lower_limit: Key, upper_limit: Key):
        return self.root.validate(lower_limit, upper_limit)

    def pretty_str(self):
        return self.root.pretty_str()

    def dot(self, output_file_path: str = "tree.pdf"):
        return draw_tree(self.nonnull_root, output_file_path)

    def __getitem__(self, key: Key) -> Value:
        node = self.access(key)
        if self.is_node(node):
            return node.value
        raise KeyError(f"Key {key} not found")

    def __delitem__(self, key: Key) -> None:
        self.delete(key)

    def __setitem__(self, key: Key, value: Value) -> None:
        self.insert(key, value, allow_overwrite=True)

    def level_order(self) -> Iterator[BinaryNodeType]:
        try:
            return self.nonnull_root.level_order()
        except SentinelReferenceError:
            return iter([])

    def inorder(self) -> Iterator[BinaryNodeType]:
        try:
            return self.nonnull_root.inorder()
        except SentinelReferenceError:
            return iter([])

    def preorder(self) -> Iterator[BinaryNodeType]:
        try:
            return self.nonnull_root.preorder()
        except SentinelReferenceError:
            return iter([])

    def postorder(self) -> Iterator[BinaryNodeType]:
        try:
            return self.nonnull_root.postorder()
        except SentinelReferenceError:
            return iter([])

    def yield_edges(self) -> Iterator[tuple[BinaryNodeType, BinaryNodeType]]:
        try:
            return self.nonnull_root.yield_edges()
        except SentinelReferenceError:
            return iter([])

    def predecessor_node(self, key: Key) -> BinaryNodeType:
        try:
            return self.nonnull_root.predecessor_node(key)
        except SentinelReferenceError as e:
            raise KeyError(f"No predecessor found, {key} is minimum key in tree") from e

    def successor_node(self, key: Key) -> BinaryNodeType:
        try:
            return self.nonnull_root.successor_node(key)
        except SentinelReferenceError as e:
            raise KeyError(f"No successor found, {key} is maximum key in tree") from e

    def successor(self, key: Key) -> tuple[Key, Value]:
        node = self.successor_node(key)
        return node.key, node.value

    def predecessor(self, key: Key) -> tuple[Key, Value]:
        node = self.predecessor_node(key)
        return node.key, node.value

    def access(self, key: Key) -> BinaryNodeType:
        return self.nonnull_root.access(key)

    def insert(
        self, key: Key, value: Value, allow_overwrite: bool = True
    ) -> BinaryNodeType:
        node = self.node(key, value)
        self.insert_node(node, allow_overwrite)
        return node

    def insert_node(self, node: BinaryNodeType, allow_overwrite: bool = False) -> bool:
        if node.key in self:
            if allow_overwrite:
                existing_node = self.access(node.key)
                existing_node.value = node.value
                return False
            raise KeyError(f"Key {node.key} already exists")
        return bool(
            self._replace_root(
                node
                if self.is_sentinel(self.root)
                else self.nonnull_root.insert_node(node, allow_overwrite)
            )
        )

    def delete_node(self, key: Key) -> BinaryNodeType:
        if node := self.access(key):
            self.root = self.nonnull_root.delete_key(key)
            self.size -= 1
            return node
        else:
            raise KeyError(f"Key {key} not found")

    def delete(self, key: Key) -> Value:
        node = self.delete_node(key)
        return node.value

    def extract_min(
        self,
    ) -> tuple[Key, Value]:
        keyval, self.root = self.nonnull_root.extract_min()
        self.size -= 1
        return keyval

    def extract_max(
        self,
    ) -> tuple[Key, Value]:
        keyval, self.root = self.nonnull_root.extract_max()
        self.size -= 1
        return keyval

    def delete_min(self):
        self.root = self.nonnull_root.delete_min()
        self.size -= 1

    def delete_max(self):
        self.root = self.nonnull_root.delete_max()
        self.size -= 1

    def minimum_node(self, _: Optional[BinaryNodeType] = None) -> BinaryNodeType:
        return self.nonnull_root.minimum_node()

    def maximum_node(self, _: Optional[BinaryNodeType] = None) -> BinaryNodeType:
        return self.nonnull_root.maximum_node()

    def minimum(self) -> tuple[Key, Value]:
        node = self.minimum_node()
        return node.key, node.value

    def maximum(self) -> tuple[Key, Value]:
        node = self.maximum_node()
        return node.key, node.value

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


class BSTWithParent(
    AbstractBST[Key, Value, BinaryNodeWithParentType, SentinelType],
    ABC,
):
    root: Union[BinaryNodeWithParentType, SentinelType]

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
            self.root.parent = self.sentinel()

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


class AbstractBSTIterative(
    AbstractBST[
        Key,
        Value,
        BinaryNodeType,
        SentinelType,
    ],
    ABC,
):
    def access(self, key: Key) -> BinaryNodeType:
        current = self.nonnull_root
        while self.is_node(current):
            if current.key == key:
                return current
            elif key < current.key:
                current = current.nonnull_left
            else:
                current = current.nonnull_right
        return current

    def access_with_parent(
        self, key: Key
    ) -> tuple[BinaryNodeType, Union[BinaryNodeType, SentinelType]]:
        parent: Union[BinaryNodeType, SentinelType] = self.sentinel()
        current = self.nonnull_root
        while self.is_node(current):
            if current.key == key:
                return current, parent
            elif key < current.key:
                parent = current
                current = current.nonnull_left
            else:
                parent = current
                current = current.nonnull_right
        return current, parent

    def minimum_node(self, root: Optional[BinaryNodeType] = None) -> BinaryNodeType:
        current = self.nonnull_root if root is None else root
        while self.is_node(current.left):
            current = current.left
        return current

    def maximum_node(self, root: Optional[BinaryNodeType] = None) -> BinaryNodeType:
        current = self.nonnull_root if root is None else root
        while self.is_node(current.right):
            current = current.right
        return current

    def maximum(self) -> tuple[Key, Value]:
        node = self.maximum_node()
        return node.key, node.value

    def minimum(self) -> tuple[Key, Value]:
        node = self.minimum_node()
        return node.key, node.value

    def insert_node(self, node: BinaryNodeType, allow_overwrite: bool = True) -> bool:
        if self.is_sentinel(self.root):
            self.root = node
        else:
            current = self.nonnull_root
            while True:
                if current.key == node.key:
                    if allow_overwrite:
                        current.value = node.value
                        return False
                    else:
                        raise ValueError(f"Key {node.key} already in tree")
                elif current.key < node.key:
                    if self.is_node(current.right):
                        current = current.right
                    else:
                        current.right = node
                        break
                else:
                    if self.is_node(current.left):
                        current = current.left
                    else:
                        current.left = node
                        break
        self.size += 1
        return True

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

    def delete_node(self, key: Key) -> BinaryNodeType:
        try:
            target_node = self.access(key)
            if self.is_sentinel(target_node.left):
                self.transplant(target_node, target_node.right)
            elif self.is_sentinel(target_node.right):
                self.transplant(target_node, target_node.left)
            else:
                successor = target_node.nonnull_right.minimum_node()
                if successor != target_node.right:
                    self.transplant(successor, successor.right)
                    successor.right = target_node.right
                self.transplant(target_node, successor)
                successor.left = target_node.left
            self.size -= 1
            return target_node
        except SentinelReferenceError as e:
            raise ValueError(f"Key {key} not in tree") from e

    def extract_min(self) -> tuple[Key, Value]:
        root = self.nonnull_root
        prev: Union[BinaryNodeType, SentinelType] = self.sentinel()

        while self.is_node(root.left):
            prev = root
            root = root.left

        keyval = (root.key, root.value)
        self.transplant(root, root.right, prev)
        self.size -= 1
        return keyval

    def extract_max(self) -> tuple[Key, Value]:
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

    def successor_node(self, key: Key) -> BinaryNodeType:
        try:
            node = self.access(key)
            if self.is_node(node.right):
                return node.right.minimum_node()
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
        except SentinelReferenceError as e:
            raise ValueError(f"Key {key} not found") from e

    def predecessor_node(self, key: Key) -> BinaryNodeType:
        try:
            node = self.access(key)
            if self.is_node(node.left):
                return node.left.maximum_node()
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
        except SentinelReferenceError as e:
            raise ValueError(f"Key {key} not found") from e

    def successor(self, key: Key) -> tuple[Key, Value]:
        node = self.successor_node(key)
        return node.key, node.value

    def predecessor(self, key: Key) -> tuple[Key, Value]:
        node = self.predecessor_node(key)
        return node.key, node.value

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


class AbstractBSTWithParentIterative(
    AbstractBSTIterative[Key, Value, BinaryNodeWithParentType, SentinelType],
    ABC,
):
    def insert_node(
        self, node: BinaryNodeWithParentType, allow_overwrite: bool = True
    ) -> bool:
        x = self.root
        y: Union[BinaryNodeWithParentType, SentinelType] = self.sentinel()

        while self.is_node(x):
            y = x
            if node.key < x.key:
                x = x.left
            elif node.key > x.key:
                x = x.right
            elif allow_overwrite:
                x.value = node.value
                return False
            else:
                raise ValueError(f"Key {node.key} already in tree")
        node.parent = y
        if self.is_node(y):
            if node.key < y.key:
                y.left = node
            else:
                y.right = node
        else:
            self.root = node
        self.size += 1
        return True

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

    def delete_node(self, key: Key) -> BinaryNodeWithParentType:
        try:
            node = self.access(key)
            if self.is_sentinel(node.left):
                self.transplant(node, node.right)
            elif self.is_sentinel(node.right):
                self.transplant(node, node.left)
            else:
                successor = node.right.minimum_node()
                if successor is not node.right:
                    self.transplant(successor, successor.right)
                    successor.right = node.right
                    successor.right.parent = successor
                self.transplant(node, successor)
                successor.left = node.left
                successor.left.parent = successor
            self.size -= 1
            return node
        except SentinelReferenceError as e:
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

    def successor_node(self, key: Key) -> BinaryNodeWithParentType:
        try:
            node = self.access(key)
            if self.is_node(node.right):
                return self.minimum_node(node.right)
            else:
                y = node.parent
                while self.is_node(y) and node is y.right:
                    node = y
                    y = y.parent
                if self.is_node(y):
                    return y
                raise KeyError(f"{key=} has no successor")
        except SentinelReferenceError as e:
            raise KeyError(f"{key=} not found") from e


@dataclass
class BST(
    AbstractBST[
        Key,
        Value,
        BSTNode[Key, Value],
        Sentinel,
    ]
):
    def __init__(self, root: Optional[BSTNode[Key, Value]] = None):
        super().__init__(root)

    @classmethod
    def sentinel_class(cls):
        return Sentinel

    @staticmethod
    def is_node(node: Any) -> TypeGuard[BSTNode[Key, Value]]:
        return isinstance(node, BSTNode)

    @staticmethod
    def node(
        key: Key,
        value: Value,
        left: Union[BSTNode[Key, Value], Sentinel] = Sentinel(),
        right: Union[BSTNode[Key, Value], Sentinel] = Sentinel(),
        *args,
        **kwargs,
    ) -> BSTNode[Key, Value]:
        return BSTNode[Key, Value](key, value, left, right)


class BSTIterative(AbstractBSTIterative[Key, Value, BSTNode[Key, Value], Sentinel]):
    @classmethod
    def sentinel_class(cls):
        return Sentinel

    @staticmethod
    def is_node(node: Any) -> TypeGuard[BSTNode[Key, Value]]:
        return isinstance(node, BSTNode)

    @staticmethod
    def node(
        key: Key,
        value: Value,
        left: Union[BSTNode[Key, Value], Sentinel] = Sentinel(),
        right: Union[BSTNode[Key, Value], Sentinel] = Sentinel(),
        *args,
        **kwargs,
    ) -> BSTNode[Key, Value]:
        return BSTNode(key, value, left, right)


class BSTWithParentIterative(
    AbstractBSTWithParentIterative[
        Key,
        Value,
        BSTNodeWithParent[Key, Value],
        Sentinel,
    ],
):
    @classmethod
    def sentinel_class(cls):
        return Sentinel

    @staticmethod
    def is_node(item: Any) -> TypeGuard[BSTNodeWithParent[Key, Value]]:
        return isinstance(item, BSTNodeWithParent)

    @staticmethod
    def node(
        key: Key,
        value: Value,
        left: Union[
            BSTNodeWithParent[Key, Value],
            Sentinel,
        ] = Sentinel(),
        right: Union[
            BSTNodeWithParent[Key, Value],
            Sentinel,
        ] = Sentinel(),
        parent: Union[
            BSTNodeWithParent[Key, Value],
            Sentinel,
        ] = Sentinel(),
        *args,
        **kwargs,
    ) -> BSTNodeWithParent[Key, Value]:
        return BSTNodeWithParent[Key, Value](
            key=key, value=value, left=left, right=right, parent=parent
        )
