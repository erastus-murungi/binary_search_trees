from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Iterator, TypeGuard, TypeVar, Union

from bst import AbstractBinarySearchTreeWithParentIterative, Comparable, Value
from core import AbstractSentinel, SentinelReached
from nodes import BinarySearchTreeInternalNodeWithParentAbstract

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
    BinarySearchTreeInternalNodeWithParentAbstract[
        Comparable,
        Value,
        "AVLTreeInternalNode[Comparable, Value]",
        AVLSentinel[Comparable],
    ],
    AVLTreeNodeTraits,
):
    height: int
    balance_factor: int

    def update_balance_factor(self) -> int:
        new_bf = self.right.height - self.left.height
        old_bf = self.balance_factor
        self.balance_factor = new_bf
        return old_bf != new_bf

    def update_height(self) -> bool:
        """
        Updates the height of the node.

        Returns True if the height of the node has changed.
        """
        old_height = self.height
        self.height = max(self.left.height, self.right.height) + 1
        return old_height != self.height

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
            balance_factor=0,
        )

    def is_sentinel(
        self,
        node: Any,
    ) -> TypeGuard[AVLSentinel[Comparable]]:
        return isinstance(node, AVLSentinel)

    def sentinel(self, *args, **kwargs):
        return AVLSentinel[Comparable].default()


class AVLTreeIterative(
    AbstractBinarySearchTreeWithParentIterative[
        Comparable,
        Value,
        AVLTreeInternalNode[Comparable, Value],
        AVLSentinel[Comparable],
    ]
):
    def __init__(
        self,
        root: Union[
            AVLTreeInternalNode[Comparable, Value], AVLSentinel[Comparable]
        ] = AVLSentinel[Comparable].default(),
        size: int = 0,
    ):
        super().__init__(root, size)

    def re_balance(
        self, node: AVLTreeInternalNode[Comparable, Value]
    ) -> Union[AVLTreeInternalNode[Comparable, Value], AVLSentinel[Comparable]]:
        def update_height(
            node_: Union[
                AVLTreeInternalNode[Comparable, Value], AVLSentinel[Comparable]
            ]
        ):
            if self.is_node(node_):
                node_.update_height()
                node_.update_balance_factor()

        bf = node.balance_factor
        if bf < -1:
            if node.left.balance_factor > 0:
                node.left = self.left_rotate(node.nonnull_left, update=update_height)
            return self.right_rotate(node, update=update_height)
        if bf > 1:
            if node.right.balance_factor < 0:
                node.right = self.right_rotate(node.nonnull_right, update=update_height)
            return self.left_rotate(node, update=update_height)
        return self.sentinel()

    def validate(self, lower_limit: Comparable, upper_limit: Comparable):
        super().validate(lower_limit, upper_limit)
        for node in self.inorder():
            if abs(node.balance_factor) not in (-1, 0, 1):
                raise RuntimeError(
                    f"Invalid AVL Tree, balance factor = {node.balance_factor} not in [-1, 0, 1] {values}"
                )

    def insert(
        self, key: Comparable, value: Value, allow_overwrite: bool = False
    ) -> AVLTreeInternalNode[Comparable, Value]:
        node = super().insert(key, value, allow_overwrite)
        assert self.is_node(node)
        self.avl_fixup(node.parent)
        return node

    def transplant_(
        self,
        u: AVLTreeInternalNode[Comparable, Value],
        v: Union[AVLTreeInternalNode[Comparable, Value], AVLSentinel[Comparable]],
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

    def delete(self, key: Comparable) -> AVLTreeInternalNode[Comparable, Value]:
        try:
            node = self.access(key)
            p = node.parent
            if self.is_sentinel(node.left):
                self.transplant_(node, node.right)
                self.avl_fixup(p)
            elif self.is_sentinel(node.right):
                self.transplant_(node, node.left)
                self.avl_fixup(p)
            else:
                successor = node.nonnull_right.minimum()
                if successor is not node.right:
                    # delete the successor
                    assert self.is_sentinel(successor.left)
                    self.transplant_(successor, successor.right)
                    successor.right = node.right
                    successor.right.parent = successor
                    self.avl_fixup(successor.right)
                self.transplant_(node, successor)
                successor.left = node.left
                successor.left.parent = successor
                self.avl_fixup(successor)
            self.size -= 1
            return node
        except SentinelReached as e:
            raise ValueError(f"Key {key} not in tree") from e

    def avl_fixup(
        self,
        node: Union[AVLTreeInternalNode[Comparable, Value], AVLSentinel[Comparable]],
    ):
        while self.is_node(node) and node.update_balance_factor():
            self.re_balance(node)
            node = node.parent

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

    for _ in range(10000):
        bst: AVLTreeIterative[int, None] = AVLTreeIterative[int, None]()
        num_values = 4
        # values = list({randint(0, 100) for _ in range(num_values)})
        values = [40, 9, 12, 7]

        for val in values:
            bst.insert(val, None, allow_overwrite=True)
            print(bst.pretty_str())
            bst.validate(0, 1000)
            assert val in bst

        # print(bst.pretty_str())

        for val in values:
            bst.delete(val)
            print(bst.pretty_str())
            bst.validate(0, 1000)
            # print(bst.pretty_str())
            assert val not in bst, values
