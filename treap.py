from __future__ import annotations

from dataclasses import dataclass
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


if __name__ == "__main__":
    print(Treap().pretty_str())
