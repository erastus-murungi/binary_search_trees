from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Container,
    Generic,
    Iterator,
    MutableMapping,
    Protocol,
    Self,
    Sized,
    TypeGuard,
    TypeVar,
    Union,
    runtime_checkable,
)


class SentinelReferenceError(ValueError):
    pass


@runtime_checkable
class SupportsLessThan(Protocol):
    def __lt__(self: Comparable, other: Comparable) -> bool:
        pass


Comparable = TypeVar("Comparable", bound=SupportsLessThan)
Value = TypeVar("Value")
VisitorReturnType = TypeVar("VisitorReturnType")


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


class Visitor(Generic[NodeType, SentinelType, VisitorReturnType], ABC):
    @abstractmethod
    def visit(self, node: Union[NodeType, SentinelType]) -> VisitorReturnType:
        pass


class SentinelConstructorMixin(Generic[SentinelType], ABC):
    @abstractmethod
    def sentinel(self, *args, **kwargs) -> SentinelType:
        pass

    @abstractmethod
    def is_sentinel(self, node: Any) -> TypeGuard[SentinelType]:
        pass


class MemberMixin(
    Generic[Comparable, NodeType, SentinelType],
    Container[Comparable],
    SentinelConstructorMixin[SentinelType],
):
    @abstractmethod
    def __contains__(self, key: Any) -> bool:
        pass

    @abstractmethod
    def access(self, key: Comparable) -> NodeType:
        pass

    def access_no_throw(self, key: Comparable) -> Union[NodeType, SentinelType]:
        try:
            return self.access(key)
        except SentinelReferenceError:
            return self.sentinel()


@dataclass(slots=True)
class AbstractSentinel(
    Generic[Comparable],
    Container[Comparable],
    PrettyStrMixin,
    PrettyLineYieldMixin,
    ValidatorMixin,
    Sized,
    ABC,
):
    """Base class for all leaves in a tree."""

    @classmethod
    @abstractmethod
    def default(cls) -> Self:
        pass

    def __contains__(self, key: Any) -> bool:
        return False

    def __bool__(self):
        return False

    def pretty_str(self) -> str:
        return "∅"

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----'∅'\n"

    def validate(self, *arg, **kwargs) -> bool:
        return True

    def __len__(self) -> int:
        return 0


class TreeQueryMixin(Generic[Comparable, NodeType, SentinelType], ABC):
    @abstractmethod
    def minimum(self) -> NodeType:
        pass

    @abstractmethod
    def maximum(self) -> NodeType:
        pass

    @abstractmethod
    def successor(self, key: Comparable) -> Union[NodeType, SentinelType]:
        pass

    @abstractmethod
    def predecessor(self, key: Comparable) -> Union[NodeType, SentinelType]:
        pass


class TreeMutationMixin(Generic[Comparable, Value, NodeType], ABC):
    @abstractmethod
    def insert(
        self, key: Comparable, value: Value, allow_overwrite: bool = True
    ) -> NodeType:
        """Insert a new node with the specified key and value into the tree.

        If allow_overwrite is True, then the value associated with the key will be
        overwritten if the key already exists in the tree.

        Parameters
        ----------
        key : Comparable
            The key to insert into the tree.
        value : Value
            The value to associate with the key.
        allow_overwrite : bool
            Whether to overwrite the value associated with the key if the key already
            exists in the tree.

        Returns
        -------
        node : NodeType
            The newly inserted node.

        Raises
        ------
        ValueError
            If allow_overwrite is False and the key already exists in the tree.
        """

    @abstractmethod
    def delete(self, key: Comparable) -> NodeType:
        """Delete the node with the specified key from the tree.


        Parameters
        ----------
        key : Comparable
            The key to delete from the tree.

        Returns
        -------
        node : NodeType
            The deleted node.

        Raises
        ------
        KeyError
            If the key does not exist in the tree.
        """

    @abstractmethod
    def extract_min(
        self,
    ) -> tuple[Comparable, Value]:
        """
        Extract the minimum node from the tree.

        Returns
        -------
        tuple[Comparable, Value]
            The key value tuple representing the minimum

        """

    @abstractmethod
    def extract_max(
        self,
    ) -> tuple[Comparable, Value]:
        """
        Extract the maximum node from the tree.

        Returns
        -------
        tuple[Comparable, Value]
            The key value tuple representing the maximum

        """


class NodeMutationMixin(Generic[Comparable, Value, NodeType, SentinelType], ABC):
    @abstractmethod
    def _insert(
        self, key: Comparable, value: Value, allow_overwrite: bool = True
    ) -> NodeType:
        """Insert a new node with the specified key and value into the tree.

        If allow_overwrite is True, then the value associated with the key will be
        overwritten if the key already exists in the tree.

        Parameters
        ----------
        key : Comparable
            The key to insert into the tree.
        value : Value
            The value to associate with the key.
        allow_overwrite : bool
            Whether to overwrite the value associated with the key if the key already
            exists in the tree.

        Returns
        -------
        node : NodeType
            The new root of the tree

        Raises
        ------
        ValueError
            If allow_overwrite is False and the key already exists in the tree.
        """

    @abstractmethod
    def _delete(self, key: Comparable) -> Union[NodeType, SentinelType]:
        """Delete the node with the specified key from the tree.


        Parameters
        ----------
        key : Comparable
            The key to delete from the tree.

        Returns
        -------
        node : NodeType
            The root of the tree

        Raises
        ------
        KeyError
            If the key does not exist in the tree.
        """

    @abstractmethod
    def _extract_min(
        self,
    ) -> tuple[tuple[Comparable, Value], Union[NodeType, SentinelType]]:
        """
        Extract the minimum node from the tree.

        Returns
        -------
        tuple[tuple[Comparable, Value], Union[NodeType, SentinelType]]
            The key value tuple representing the minimum
            and the new root of the tree

        """

    @abstractmethod
    def _extract_max(
        self,
    ) -> tuple[tuple[Comparable, Value], Union[NodeType, SentinelType]]:
        """
        Extract the maximum node from the tree.

        Returns
        -------
        tuple[tuple[Comparable, Value], Union[NodeType, SentinelType]]
            The key value tuple representing the maximum
            and the new root of the tree
        """


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
    NodeMutationMixin[Comparable, Value, NodeType, SentinelType],
    NodeConstructorMixin[NodeType, Comparable, Value],
    MutableMapping[Comparable, Value],
    TreeQueryMixin[Comparable, NodeType, SentinelType],
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


@dataclass
class AbstractTree(
    Generic[Comparable, Value, NodeType, SentinelType],
    TreeMutationMixin[Comparable, Value, NodeType],
    NodeConstructorMixin[NodeType, Comparable, Value],
    MutableMapping[Comparable, Value],
    TreeQueryMixin[Comparable, NodeType, SentinelType],
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

    def __post_init__(self):
        assert len(self.root) == self.size

    def clear(self):
        self.root = self.sentinel()
        self.size = 0

    def __len__(self):
        return self.size
