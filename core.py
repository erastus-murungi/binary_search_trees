from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (Any, Container, Generic, Iterator, MutableMapping,
                    Protocol, Self, Sized, TypeGuard, TypeVar, Union,
                    runtime_checkable)


class SentinelReached(ValueError):
    pass


@runtime_checkable
class SupportsLessThan(Protocol):
    def __lt__(self: Comparable, other: Comparable) -> bool:
        pass


Comparable = TypeVar("Comparable", bound=SupportsLessThan)
Value = TypeVar("Value")


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


SentinelType = TypeVar("SentinelType", bound="AbstractSentinel")
NodeType = TypeVar("NodeType", bound="AbstractNode")


class MemberMixin(
    Generic[Comparable, NodeType, SentinelType], Container[Comparable], ABC
):
    @abstractmethod
    def __contains__(self, item: Any) -> bool:
        pass

    @abstractmethod
    def access(self, key: Comparable) -> Union[NodeType, SentinelType]:
        pass


class AbstractSentinel(
    Generic[Comparable, NodeType, SentinelType],
    PrettyStrMixin,
    PrettyLineYieldMixin,
    ValidatorMixin,
    Sized,
    MemberMixin[Comparable, NodeType, SentinelType],
    ABC,
):
    """Base class for all leaves in a tree."""

    @classmethod
    @abstractmethod
    def default(cls) -> Self:
        pass


class TreeQueryMixin(Generic[Comparable, NodeType], ABC):
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


class TreeMutationMixin(Generic[Comparable, Value, NodeType, SentinelType], ABC):
    @abstractmethod
    def insert(self, key: Comparable, value: Value, allow_overwrite: bool) -> NodeType:
        pass

    @abstractmethod
    def delete(self, key: Comparable) -> Union[NodeType, SentinelType]:
        pass

    @abstractmethod
    def extract_min(
        self,
    ) -> tuple[tuple[Comparable, Value], Union[NodeType, SentinelType]]:
        pass

    @abstractmethod
    def extract_max(
        self,
    ) -> tuple[tuple[Comparable, Value], Union[NodeType, SentinelType]]:
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
    def is_leaf(self, node: Any) -> TypeGuard[SentinelType]:
        pass


class NodeConstructorMixin(Generic[NodeType, Comparable, Value], ABC):
    @abstractmethod
    def node(self, key: Comparable, value: Value, *args, **kwargs) -> NodeType:
        pass

    @abstractmethod
    def is_node(self, node: Any) -> TypeGuard[NodeType]:
        pass


@dataclass
class AbstractNode(
    Generic[Comparable, Value, NodeType, SentinelType],
    TreeMutationMixin[Comparable, Value, NodeType, SentinelType],
    NodeConstructorMixin[NodeType, Comparable, Value],
    MutableMapping[Comparable, Value],
    TreeQueryMixin[Comparable, NodeType],
    LeafConstructorMixin[SentinelType],
    TreeIterativeMixin[NodeType],
    MemberMixin[Comparable, NodeType, SentinelType],
    PrettyLineYieldMixin,
    PrettyStrMixin,
    ValidatorMixin,
    Sized,
    ABC,
):
    """Base class for all nodes in a tree."""

    key: Comparable
    value: Value

    @abstractmethod
    def swap(self, node: NodeType):
        pass


@dataclass
class AbstractTree(
    Generic[Comparable, Value, NodeType, SentinelType],
    TreeMutationMixin[Comparable, Value, NodeType, SentinelType],
    NodeConstructorMixin[NodeType, Comparable, Value],
    MutableMapping[Comparable, Value],
    TreeQueryMixin[Comparable, NodeType],
    LeafConstructorMixin[SentinelType],
    TreeIterativeMixin[NodeType],
    MemberMixin[Comparable, NodeType, SentinelType],
    PrettyStrMixin,
    ValidatorMixin,
    Sized,
    ABC,
):
    """Base class for all trees."""

    root: Union[NodeType, SentinelType]
    size: int = 0
