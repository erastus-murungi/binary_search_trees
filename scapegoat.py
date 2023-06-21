import math
from dataclasses import dataclass, field
from functools import cache
from math import log
from typing import Any, Final, TypeGuard, Union

from bst import BSTWithParentIterative
from core import Comparable, Value
from nodes import BSTNodeWithParent, Sentinel

# log (3/2)
LOG_1_5: Final[float] = 0.4054651081081644


def log3_2(m: float) -> int:
    return math.ceil(log(m) / LOG_1_5)


def compute_size(
    node: Union[BSTNodeWithParent[Comparable, Value], Sentinel[Comparable]]
) -> int:
    if isinstance(node, Sentinel):
        return 0
    return 1 + compute_size(node.left) + compute_size(node.right)


@dataclass
class ScapeGoatTree(BSTWithParentIterative[Comparable, Value]):
    max_size: int = 0
    root: Union[BSTNodeWithParent[Comparable, Value], Sentinel[Comparable]] = field(
        default_factory=Sentinel[Comparable]
    )
    size: int = 0

    def insert_with_depth(
        self, node: BSTNodeWithParent[Comparable, Value], allow_overwrite: bool
    ) -> int:
        parent: Union[
            BSTNodeWithParent[Comparable, Value], Sentinel[Comparable]
        ] = self.sentinel()
        root = self.root
        depth = 0
        while self.is_node(root):
            if node.key == root.key:
                if allow_overwrite:
                    root.value = node.value
                    return -1
                else:
                    raise ValueError(f"Key {node.key} already exists in tree")
            elif node.key < root.key:
                parent = root
                root = root.left
            else:
                parent = root
                root = root.right
            depth += 1
        if self.is_sentinel(parent):
            self.root = node
        else:
            assert self.is_node(parent)
            if node.key < parent.key:
                parent.left = node
            else:
                parent.right = node
            node.parent = parent

        self.size += 1
        self.max_size += 1

        return depth

    def insert(
        self, key: Comparable, value: Value, allow_overwrite: bool = False
    ) -> BSTNodeWithParent[Comparable, Value]:
        node = self.node(key, value)
        depth = self.insert_with_depth(node, allow_overwrite)
        if depth < 0:
            return node
        if depth > log3_2(self.max_size):
            w = node.parent
            cached_size = cache(compute_size)
            while self.is_node(w) and 3 * cached_size(w) <= 2 * cached_size(w.parent):
                w = w.parent
            assert self.is_node(w) and self.is_node(w.parent)
            self.rebuild(w.parent)
        return node

    def rebuild(self, node: BSTNodeWithParent[Comparable, Value]) -> None:
        assert self.is_node(node)
        nodes = list(node.inorder())

        parent = node.parent
        root = self.build_from_sorted_list(nodes, parent)
        assert self.is_node(root)

        if self.is_sentinel(parent):
            self.root = root
            self.nonnull_root.parent = self.sentinel()
        else:
            assert self.is_node(parent)
            if parent.left is node:
                parent.left = root
                root.parent = parent
            else:
                parent.right = root
                root.parent = parent

    def build_from_sorted_list(
        self,
        nodes: list[BSTNodeWithParent[Comparable, Value]],
        parent: Union[Sentinel[Comparable], BSTNodeWithParent[Comparable, Value]],
    ) -> Union[BSTNodeWithParent[Comparable, Value], Sentinel[Comparable]]:
        if not nodes:
            return self.sentinel()
        mid = len(nodes) // 2
        node = nodes[mid]
        node.left = self.build_from_sorted_list(nodes[:mid], node)
        node.right = self.build_from_sorted_list(nodes[mid + 1 :], node)
        node.parent = parent
        return node

    def delete(self, key: Comparable) -> BSTNodeWithParent[Comparable, Value]:
        node = super().delete(key)
        if 2 * self.size < self.max_size:
            if self.is_node(self.root):
                self.rebuild(self.nonnull_root)
            self.max_size = self.size
        return node


if __name__ == "__main__":
    from random import randint

    for _ in range(1000):
        bst: ScapeGoatTree[int, None] = ScapeGoatTree[int, None]()
        num_values = 5
        # values = list({randint(0, 100000) for _ in range(num_values)})
        # values = [35, 70, 51, 52, 20, 55, 91]
        # values = [50, 44, 94]
        values = [59630, 8240, 5365, 19574, 59542]
        for k in values:
            bst.insert(k, None, allow_overwrite=True)
            assert compute_size(bst.root) == len(bst)
            bst.validate(0, 1000)
            assert k in bst
            assert bst.max_size == len(bst)

        # bst.dot()
        for k in values:
            deleted = bst.delete(k)
            assert deleted.key == k, (k, deleted.key)
            assert k not in bst, values
