from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    runtime_checkable,
    Protocol,
    TypeVar,
    Generic,
    Iterator,
    TypeGuard,
    Union,
    Self,
    Sized,
)


@runtime_checkable
class SupportsLessThan(Protocol):
    def __lt__(self: Comparable, other: Comparable) -> bool:
        pass


Comparable = TypeVar("Comparable", bound=SupportsLessThan)
Value = TypeVar("Value")
KeyValue = tuple[Comparable, Value]


class PrettyStrMixin(ABC):
    @abstractmethod
    def pretty_str(self) -> str:
        pass


class PrettyLineYieldMixin(ABC):
    @abstractmethod
    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        pass


class ValidatorMixin(ABC):
    @abstractmethod
    def validate(self, *arg, **kwargs) -> bool:
        pass


class MemberMixin(Generic[Comparable], ABC):
    @abstractmethod
    def __contains__(self, key: Comparable) -> bool:
        pass

    @abstractmethod
    def access(self, key: Comparable) -> Self:
        pass


class AbstractSentinel(
    Generic[Comparable],
    PrettyStrMixin,
    PrettyLineYieldMixin,
    ValidatorMixin,
    Sized,
    MemberMixin[Comparable],
    ABC,
):
    """Base class for all leaves in a tree."""

    def access(self, key: Comparable) -> Self:
        return self

    def delete(self, key: Comparable) -> Self:
        return self

    @classmethod
    @abstractmethod
    def default(cls) -> Self:
        pass


SentinelType = TypeVar("SentinelType", bound=AbstractSentinel)


@dataclass
class AbstractNode(
    Generic[Comparable, Value, SentinelType],
    PrettyStrMixin,
    ValidatorMixin,
    Sized,
    PrettyLineYieldMixin,
    MemberMixin[Comparable],
    ABC,
):
    """Base class for all nodes in a tree."""

    key: Comparable
    value: Value

    @abstractmethod
    def leaf(self) -> SentinelType:
        pass


NodeType = TypeVar("NodeType", bound=AbstractNode)


class TreeQueryMixin(Generic[NodeType, Comparable, Value], ABC):
    @abstractmethod
    def minimum(self) -> NodeType:
        pass

    @abstractmethod
    def maximum(self) -> NodeType:
        pass

    @abstractmethod
    def successor(self, key: Comparable) -> NodeType:
        pass

    @abstractmethod
    def predecessor(self, key: Comparable) -> NodeType:
        pass


class TreeIterativeMixin(Generic[NodeType], ABC):
    @abstractmethod
    def inorder(self) -> Iterator[NodeType]:
        pass

    @abstractmethod
    def preorder(self) -> Iterator[NodeType]:
        pass

    @abstractmethod
    def postorder(self) -> Iterator[NodeType]:
        pass

    @abstractmethod
    def level_order(self) -> Iterator[NodeType]:
        pass


class LeafConstructorMixin(Generic[SentinelType], ABC):
    @abstractmethod
    def leaf(self, *args, **kwargs) -> SentinelType:
        pass

    @abstractmethod
    def is_leaf(self, node: AbstractNode) -> TypeGuard[SentinelType]:
        pass


class NodeConstructorMixin(Generic[NodeType, Comparable, Value], ABC):
    @abstractmethod
    def node(self, key: Comparable, value: Value, *args, **kwargs) -> NodeType:
        pass

    @abstractmethod
    def is_node(self, node: AbstractNode) -> TypeGuard[NodeType]:
        pass


class TreeMutationMixin(Generic[NodeType, SentinelType, Comparable, Value], ABC):
    @abstractmethod
    def insert(self, key: Comparable, value: Value) -> NodeType:
        pass

    @abstractmethod
    def delete(self, key: Comparable) -> Union[NodeType, SentinelType]:
        pass

    @abstractmethod
    def extract_min(self) -> KeyValue:
        pass

    @abstractmethod
    def extract_max(self) -> KeyValue:
        pass


@dataclass
class AbstractTree(
    Generic[Comparable, Value, NodeType, SentinelType],
    NodeConstructorMixin[NodeType, Comparable, Value],
    LeafConstructorMixin[SentinelType],
    TreeQueryMixin[NodeType, Comparable, Value],
    TreeIterativeMixin[NodeType],
    TreeMutationMixin[NodeType, SentinelType, Comparable, Value],
    PrettyStrMixin,
    ValidatorMixin,
    ABC,
):
    """Base class for all trees."""

    root: Union[NodeType, SentinelType]
    size: int = 0
