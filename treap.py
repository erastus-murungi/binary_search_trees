from __future__ import annotations

from dataclasses import dataclass
from random import randint
from sys import maxsize
from typing import Any, TypeGuard, Union

from bst import AbstractBSTIterative, AbstractBSTWithParentIterative
from core import Key, Value
from nodes import AbstractBSTNode, AbstractBSTNodeWithParent, Sentinel


@dataclass(slots=True)
class TreapNodeWithParent(
    AbstractBSTNodeWithParent[Key, Value, "TreapNodeWithParent[Key, Value]", Sentinel]
):
    priority: float


@dataclass(slots=True)
class TreapNode(AbstractBSTNode[Key, Value, "TreapNode[Key, Value]", Sentinel]):
    priority: float


class Treap(
    AbstractBSTWithParentIterative[
        Key, Value, TreapNodeWithParent[Key, Value], Sentinel
    ]
):
    @classmethod
    def sentinel_class(cls) -> type[Sentinel]:
        return Sentinel

    @staticmethod
    def node(
        key: Key,
        value: Value,
        left: Union[TreapNodeWithParent[Key, Value], Sentinel] = Sentinel(),
        right: Union[TreapNodeWithParent[Key, Value], Sentinel] = Sentinel(),
        parent: Union[TreapNodeWithParent[Key, Value], Sentinel] = Sentinel(),
        priority: float = 0,
        *args,
        **kwargs,
    ) -> TreapNodeWithParent[Key, Value]:
        return TreapNodeWithParent(
            key=key,
            value=value,
            left=left,
            right=right,
            parent=parent,
            priority=priority,
        )

    @staticmethod
    def is_node(node: Any) -> TypeGuard[TreapNodeWithParent[Key, Value]]:
        return isinstance(node, TreapNodeWithParent)

    def bubble_up(self, node: TreapNodeWithParent[Key, Value]):
        while self.is_node(node.parent) and node.priority < node.parent.priority:
            if node is node.parent.left:
                self.right_rotate(node.parent)
            else:
                self.left_rotate(node.parent)
        if not self.is_node(node.parent):
            self.root = node

    def insert_with_priority(
        self, key: Key, value: Value, priority: float, allow_overwrite: bool = True
    ) -> TreapNodeWithParent[Key, Value]:
        node = self.node(key=key, value=value, priority=priority)
        self.insert_node(node, allow_overwrite)
        return node

    def insert(
        self, key: Key, value: Value, allow_overwrite: bool = True
    ) -> TreapNodeWithParent[Key, Value]:
        return self.insert_with_priority(
            key, value, randint(0, maxsize), allow_overwrite
        )

    def insert_node(
        self, node: TreapNodeWithParent[Key, Value], allow_overwrite: bool = True
    ):
        if super().insert_node(node, allow_overwrite):
            self.bubble_up(node)
            return True
        return False

    def trickle_down_to_leaf(self, node: TreapNodeWithParent[Key, Value]):
        """
        Trickle down the node until it is a leaf
        """
        while self.is_node(node.left) or self.is_node(node.right):
            if self.is_node(node.left) and self.is_node(node.right):
                if node.left.priority < node.right.priority:
                    self.right_rotate(node)
                else:
                    self.left_rotate(node)
            elif self.is_sentinel(node.left):
                self.left_rotate(node)
            else:
                self.right_rotate(node)
            if self.root is node:
                self.root = node.parent

        assert self.is_sentinel(node.left) and self.is_sentinel(node.right)

    def delete(self, key: Key) -> TreapNodeWithParent[Key, Value]:
        node = self.access(key)
        self.trickle_down_to_leaf(node)
        return super().delete(key)


class TreapSplitMerge(
    AbstractBSTIterative[Key, Value, TreapNode[Key, Value], Sentinel]
):
    @classmethod
    def sentinel_class(cls) -> type[Sentinel]:
        return Sentinel

    def split(
        self, key: Key, node: Union[TreapNode[Key, Value], Sentinel]
    ) -> tuple[
        Union[TreapNode[Key, Value], Sentinel], Union[TreapNode[Key, Value], Sentinel]
    ]:
        """
        Split the tree rooted at node into two trees, one with keys <= key and one with keys > key
        """
        if self.is_sentinel(node):
            return node, node
        if self.is_node(node):
            if node.key <= key:
                left, right = self.split(key, node.right)
                return (
                    self.node(node.key, node.value, node.left, left, node.priority),
                    right,
                )
            else:
                left, right = self.split(key, node.left)
                return left, self.node(
                    node.key, node.value, right, node.right, node.priority
                )
        return node, node

    def merge(
        self,
        left: Union[TreapNode[Key, Value], Sentinel],
        right: Union[TreapNode[Key, Value], Sentinel],
    ) -> Union[TreapNode[Key, Value], Sentinel]:
        """
        Merge two trees into one
        """
        if self.is_sentinel(left):
            return right
        if self.is_sentinel(right):
            return left
        assert self.is_node(left) and self.is_node(right)
        if left.priority < right.priority:
            return self.node(
                left.key,
                left.value,
                left.left,
                self.merge(left.right, right),
                left.priority,
            )
        else:
            return self.node(
                right.key,
                right.value,
                self.merge(left, right.left),
                right.right,
                right.priority,
            )

    def insert_with_priority(
        self, key: Key, value: Value, priority: float, allow_overwrite: bool = True
    ) -> TreapNode[Key, Value]:
        node = self.node(key=key, value=value, priority=priority)
        self.insert_node(node, allow_overwrite)
        return node

    def insert(
        self, key: Key, value: Value, allow_overwrite: bool = True
    ) -> TreapNode[Key, Value]:
        return self.insert_with_priority(
            key, value, randint(0, maxsize), allow_overwrite
        )

    def _insert_node(
        self,
        t: Union[TreapNode[Key, Value], Sentinel],
        it: TreapNode[Key, Value],
        allow_overwrite: bool = True,
    ) -> TreapNode[Key, Value]:
        if self.is_sentinel(t):
            return it
        assert self.is_node(t)
        if it.priority > t.priority:
            left, right = self.split(it.key, t)
            return self.node(it.key, it.value, left, right, it.priority)
        elif it.key < t.key:
            return self.node(
                t.key,
                t.value,
                self._insert_node(t.left, it, allow_overwrite),
                t.right,
                t.priority,
            )
        elif it.key > t.key:
            return self.node(
                t.key,
                t.value,
                t.left,
                self._insert_node(t.right, it, allow_overwrite),
                t.priority,
            )
        else:
            if allow_overwrite:
                self.size -= 1
                return self.node(t.key, it.value, t.left, t.right, t.priority)
            else:
                raise KeyError(f"Key {it.key} already exists in tree")

    def insert_node(
        self, node: TreapNode[Key, Value], allow_overwrite: bool = True
    ) -> bool:
        self.root = self._insert_node(self.root, node, allow_overwrite)
        self.size += 1
        return True

    @staticmethod
    def is_node(item: Any) -> TypeGuard[TreapNode[Key, Value]]:
        return isinstance(item, TreapNode)

    @staticmethod
    def node(
        key: Key,
        value: Value,
        left: Union[TreapNode[Key, Value], Sentinel] = Sentinel(),
        right: Union[TreapNode[Key, Value], Sentinel] = Sentinel(),
        priority: float = 0,
        *args,
        **kwargs,
    ) -> TreapNode[Key, Value]:
        return TreapNode[Key, Value](
            key=key, value=value, left=left, right=right, priority=priority
        )

    def delete(self, key: Key) -> TreapNode[Key, Value]:
        node, parent = self.access_with_parent(key)
        if self.is_sentinel(parent):
            self.root = self.merge(node.left, node.right)
        elif self.is_node(parent):
            if parent.left is node:
                parent.left = self.merge(node.left, node.right)
            else:
                parent.right = self.merge(node.left, node.right)
        self.size -= 1
        return node


if __name__ == "__main__":
    for _ in range(10):
        bst: TreapSplitMerge[int, None] = TreapSplitMerge[int, None]()
        num_values = 100
        values = list({randint(0, 100) for _ in range(num_values)})
        # values = [98, 27, 12]

        for k in values:
            bst.insert(k, None, allow_overwrite=True)
            # print(bst.pretty_str())
            bst.validate(0, 1000)
            assert k in bst

        # bst.dot()

        for k in values:
            deleted = bst.delete(k)
            assert deleted.key == k, (k, deleted.key)
            # print(bst.pretty_str())
            assert k not in bst, values
