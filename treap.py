from __future__ import annotations

from dataclasses import dataclass

from core import Key, Value
from nodes import AbstractBSTNodeWithParent, Sentinel
from bst import AbstractBSTWithParentIterative


@dataclass(slots=True)
class TreapNode(
    AbstractBSTNodeWithParent[Key, Value, "TreapNode[Key, Value]", Sentinel]
):
    priority: int


class Treap(
    AbstractBSTWithParentIterative[Key, Value, TreapNode[Key, Value], Sentinel]
):
    pass
