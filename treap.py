from __future__ import annotations

from dataclasses import dataclass
from random import randint
from sys import maxsize
from typing import Any, TypeGuard, Union

from bst import AbstractBSTWithParentIterative
from core import Key, Value
from nodes import AbstractBSTNodeWithParent, Sentinel


@dataclass(slots=True)
class TreapNode(
    AbstractBSTNodeWithParent[Key, Value, "TreapNode[Key, Value]", Sentinel]
):
    priority: float


class Treap(
    AbstractBSTWithParentIterative[Key, Value, TreapNode[Key, Value], Sentinel]
):
    @classmethod
    def sentinel_class(cls) -> type[Sentinel]:
        return Sentinel

    @staticmethod
    def node(
        key: Key,
        value: Value,
        left: Union[TreapNode[Key, Value], Sentinel] = Sentinel(),
        right: Union[TreapNode[Key, Value], Sentinel] = Sentinel(),
        parent: Union[TreapNode[Key, Value], Sentinel] = Sentinel(),
        priority: float = 0,
        *args,
        **kwargs,
    ) -> TreapNode[Key, Value]:
        return TreapNode(
            key=key,
            value=value,
            left=left,
            right=right,
            parent=parent,
            priority=priority,
        )

    @staticmethod
    def is_node(node: Any) -> TypeGuard[TreapNode[Key, Value]]:
        return isinstance(node, TreapNode)

    def bubble_up(self, node: TreapNode[Key, Value]):
        while self.is_node(node.parent) and node.priority < node.parent.priority:
            if node is node.parent.left:
                self.right_rotate(node.parent)
            else:
                self.left_rotate(node.parent)
        if not self.is_node(node.parent):
            self.root = node

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

    def insert_node(self, node: TreapNode[Key, Value], allow_overwrite: bool = True):
        if super().insert_node(node, allow_overwrite):
            self.bubble_up(node)
            return True
        return False

    def trickle_down_to_leaf(self, node: TreapNode[Key, Value]):
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

    def delete(self, key: Key) -> TreapNode[Key, Value]:
        node = self.access(key)
        self.trickle_down_to_leaf(node)
        return super().delete(key)


if __name__ == "__main__":
    for _ in range(10000):
        bst: Treap[int, None] = Treap[int, None]()
        num_values = 10
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
