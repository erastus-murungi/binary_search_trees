from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Self, TypeGuard, TypeVar, Union

from bst import AbstractBinarySearchTreeIterativeWithParent, Comparable, Value
from core import AbstractSentinel, SentinelReached
from nodes import BinarySearchTreeInternalNodeWithParentAbstract

AVLInternalNodeType = TypeVar("AVLInternalNodeType", bound="AVLTreeInternalNode")
AVLSentinelType = TypeVar("AVLSentinelType", bound="AVLSentinel")


@dataclass(slots=True)
class AVLTreeNodeTraits(ABC):
    height: int

    @abstractmethod
    def balance_factor(self) -> int:
        ...

    def update_height(self):
        ...


@dataclass(slots=True)
class AVLSentinel(AbstractSentinel[Comparable], AVLTreeNodeTraits):
    height: int = -1

    def balance_factor(self) -> int:
        return 0

    def update_height(self):
        pass

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
        Comparable, Value, "AVLTreeInternalNode[Comparable, Value]", AVLSentinel[Comparable]
    ],
    AVLTreeNodeTraits,
):
    height: int

    def balance_factor(self) -> int:
        return self.right.height - self.left.height

    def update_height(self):
        self.height = max(self.left.height, self.right.height) + 1

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
            key=key, value=value, left=left, right=right, parent=parent, height=height
        )

    def is_sentinel(
        self,
        node: Any,
    ) -> TypeGuard[AVLSentinel[Comparable]]:
        return isinstance(node, AVLSentinel)

    def sentinel(self, *args, **kwargs):
        return AVLSentinel[Comparable].default()


class AVLTreeIterative(
    AbstractBinarySearchTreeIterativeWithParent[
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
    ) -> Optional[AVLTreeInternalNode[Comparable, Value]]:
        bf = node.balance_factor()
        if bf < -1:
            if node.left.balance_factor() > 0:
                node.left = self.left_rotate(node.left)
            return self.right_rotate(node)
        if bf > 1:
            if node.right.balance_factor() < 0:
                node.right = self.right_rotate(node.right)
            return self.left_rotate(node)
        return None

    def validate(self, lower_limit: Comparable, upper_limit: Comparable):
        super().validate(lower_limit, upper_limit)
        for node in self.inorder():
            if abs(node.balance_factor()) not in (-1, 0, 1):
                raise RuntimeError(
                    f"Invalid AVL Tree, balance factor = {node.balance_factor()} not in [-1, 0, 1]"
                )

    def insert(
        self, key: Comparable, value: Value, allow_overwrite: bool = False
    ) -> AVLTreeInternalNode[Comparable, Value]:
        if node := super().insert(key, value, allow_overwrite):
            assert self.is_node(node)
            self.avl_fixup(node)
        return node

    def delete(
        self, key: Comparable
    ) -> Union[AVLTreeInternalNode[Comparable, Value], AVLSentinel[Comparable]]:
        if node := super().delete(key):
            assert self.is_node(node)
            self.avl_fixup(node.parent)
        return node

    def avl_fixup(self, node: AVLTreeInternalNode[Comparable, Value]):
        while node:
            node.update_height()
            node = self.re_balance(node)

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
            key=key, value=value, left=left, right=right, parent=parent, height=height
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

    for _ in range(1000000):
        bst: AVLTreeIterative[int, None] = AVLTreeIterative[int, None]()
        num_values = 7
        values = list({randint(0, 100) for _ in range(num_values)})
        # values = [35, 70, 51, 52, 20, 55, 91]

        for val in values:
            # print(bst.pretty_str())
            bst.insert(val, None, allow_overwrite=True)
            bst.validate(0, 1000)
            assert val in bst

        # print(bst.pretty_str())
        for val in values:
            bst.validate(0, 1000)
            bst.delete(val)
            # print(bst.pretty_str())
            assert val not in bst, values

        # print(bst.pretty_str())
