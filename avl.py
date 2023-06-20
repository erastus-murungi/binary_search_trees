from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Iterator, TypeGuard, TypeVar, Union

from bst import AbstractBinarySearchTreeWithParentIterative, Comparable, Value
from core import AbstractSentinel, SentinelReferenceError
from nodes import AbstractBinarySearchTreeInternalNodeWithParent

AVLInternalNodeType = TypeVar("AVLInternalNodeType", bound="AVLTreeInternalNode")
AVLSentinelType = TypeVar("AVLSentinelType", bound="AVLSentinel")


@dataclass(slots=True)
class AVLTreeNodeTraits(ABC):
    height: int
    balance_factor: int


@dataclass(slots=True)
class AVLSentinel(AbstractSentinel[Comparable], AVLTreeNodeTraits):
    height: int = -1
    balance_factor: int = 0

    @classmethod
    def default(cls) -> AVLSentinel[Comparable]:
        return AVLSentinel()

    def pretty_str(self) -> str:
        return "∅"

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----'∅'\n"

    def validate(self, *arg, **kwargs) -> bool:
        return True

    def __len__(self) -> int:
        return 0


@dataclass(slots=True)
class AVLTreeInternalNode(
    AbstractBinarySearchTreeInternalNodeWithParent[
        Comparable,
        Value,
        "AVLTreeInternalNode[Comparable, Value]",
        AVLSentinel[Comparable],
    ],
    AVLTreeNodeTraits,
):
    height: int
    balance_factor: int

    def update_height_and_balance_factor(self) -> bool:
        self.height = max(self.left.height, self.right.height) + 1
        new_bf = self.right.height - self.left.height
        old_bf = self.balance_factor
        self.balance_factor = new_bf
        return old_bf != new_bf


def update_balance_factor_and_height(
    node: Union[AVLTreeInternalNode[Comparable, Value], AVLSentinel[Comparable]]
) -> None:
    if isinstance(node, AVLSentinel):
        return
    node.update_height_and_balance_factor()


class AVLTreeIterative(
    AbstractBinarySearchTreeWithParentIterative[
        Comparable,
        Value,
        AVLTreeInternalNode[Comparable, Value],
        AVLSentinel[Comparable],
    ]
):
    def re_balance(self, node: AVLTreeInternalNode[Comparable, Value]) -> None:
        if node.balance_factor < -1:
            if node.left.balance_factor > 0:
                node.left = self.left_rotate(
                    node.nonnull_left, update=update_balance_factor_and_height
                )
            self.right_rotate(node, update=update_balance_factor_and_height)
        if node.balance_factor > 1:
            if node.right.balance_factor < 0:
                node.right = self.right_rotate(
                    node.nonnull_right, update=update_balance_factor_and_height
                )
            self.left_rotate(node, update=update_balance_factor_and_height)

    def validate(self, lower_limit: Comparable, upper_limit: Comparable):
        super().validate(lower_limit, upper_limit)
        for node in self.inorder():
            if abs(node.balance_factor) not in (-1, 0, 1):
                raise RuntimeError(
                    f"Invalid AVL Tree, balance factor = {node.balance_factor} not in [-1, 0, 1] {values}"
                )

        def visit(
            node: Union[
                AVLTreeInternalNode[Comparable, Value], AVLSentinel[Comparable]
            ],
        ) -> int:
            if isinstance(node, AVLSentinel):
                if node.height != -1:
                    raise RuntimeError(
                        f"Invalid AVL Tree, height = {node.height} != -1"
                    )
                if node.balance_factor != 0:
                    raise RuntimeError(
                        f"Invalid AVL Tree, balance factor = {node.balance_factor} != 0"
                    )
                return -1
            h = max(visit(node.left), visit(node.right)) + 1
            if node.height != h:
                raise RuntimeError(f"Invalid AVL Tree, height = {node.height} != {h}")
            bf = node.right.height - node.left.height
            if node.balance_factor != bf:
                raise RuntimeError(
                    f"Invalid AVL Tree, balance factor = {node.balance_factor} != {bf}"
                )
            return h

        visit(self.root)

    def insert(
        self, key: Comparable, value: Value, allow_overwrite: bool = False
    ) -> AVLTreeInternalNode[Comparable, Value]:
        node = super().insert(key, value, allow_overwrite)
        assert self.is_node(node)
        self.avl_fixup(node.parent)
        return node

    def delete(self, key: Comparable) -> AVLTreeInternalNode[Comparable, Value]:
        try:
            node = self.access(key)
            if self.is_sentinel(node.left):
                self.transplant(node, node.right)
                self.avl_fixup(node.parent)
            elif self.is_sentinel(node.right):
                self.transplant(node, node.left)
                self.avl_fixup(node.parent)
            else:
                successor = node.nonnull_right.minimum()
                if successor is not node.right:
                    assert self.is_sentinel(successor.left)
                    assert self.is_node(node.right)
                    self.transplant(successor, successor.right)
                    self.avl_fixup(successor.parent)
                    successor.right = node.right
                    successor.right.parent = successor

                self.transplant(node, successor)
                successor.left = node.left
                if self.is_node(successor.left):
                    successor.left.parent = successor
                self.avl_fixup(successor)
            self.size -= 1
            return node
        except SentinelReferenceError as e:
            raise ValueError(f"Key {key} not in tree") from e

    def extract_min(self) -> tuple[Comparable, Value]:
        root = self.minimum()
        keyval = (root.key, root.value)
        self.transplant(root, root.right)
        self.avl_fixup(root.parent)
        self.size -= 1
        return keyval

    def extract_max(self) -> tuple[Comparable, Value]:
        root = self.maximum()
        keyval = (root.key, root.value)
        self.transplant(root, root.left)
        self.avl_fixup(root.parent)
        self.size -= 1
        return keyval

    def avl_fixup(
        self,
        node: Union[AVLTreeInternalNode[Comparable, Value], AVLSentinel[Comparable]],
    ):
        while self.is_node(node) and node.update_height_and_balance_factor():
            parent = node.parent
            self.re_balance(node)
            node = parent

    def is_node(
        self,
        node: Any,
    ) -> TypeGuard[AVLTreeInternalNode[Comparable, Value]]:
        return isinstance(node, AVLTreeInternalNode)

    def node(
        self,
        key: Comparable,
        value: Value,
        left: Union[
            AVLTreeInternalNode[Comparable, Value], AVLSentinel[Comparable]
        ] = AVLSentinel[Comparable].default(),
        right: Union[
            AVLTreeInternalNode[Comparable, Value], AVLSentinel[Comparable]
        ] = AVLSentinel[Comparable].default(),
        parent: Union[
            AVLTreeInternalNode[Comparable, Value], AVLSentinel[Comparable]
        ] = AVLSentinel[Comparable].default(),
        height: int = 0,
        balance_factor: int = 0,
        *args,
        **kwargs,
    ) -> AVLTreeInternalNode[Comparable, Value]:
        return AVLTreeInternalNode(
            key=key,
            value=value,
            left=left,
            right=right,
            parent=parent,
            height=height,
            balance_factor=balance_factor,
        )

    def is_sentinel(
        self,
        node: Any,
    ) -> TypeGuard[AVLSentinel[Comparable]]:
        return isinstance(node, AVLSentinel)

    def sentinel(self, *args, **kwargs):
        return AVLSentinel[Comparable].default()


if __name__ == "__main__":
    from random import randint
    from sys import maxsize

    for _ in range(10000):
        bst: AVLTreeIterative[int, None] = AVLTreeIterative[int, None]()
        num_values = 7
        # values = list({randint(0, 100) for _ in range(num_values)})
        values = [64, 98, 8, 44, 81, 20, 27]

        for val in values:
            bst.insert(val, None, allow_overwrite=True)
            # print(bst.pretty_str())
            bst.validate(-maxsize, maxsize)
            assert val in bst

        print(bst.pretty_str())

        for val in values:
            bst.delete(val)
            print(bst.pretty_str())
            bst.validate(0, 1000)
            # print(bst.pretty_str())
            assert val not in bst, values
