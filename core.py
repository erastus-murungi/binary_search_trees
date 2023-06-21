from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Container,
    Generic,
    Iterator,
    MutableMapping,
    Optional,
    Protocol,
    Sized,
    Type,
    TypeGuard,
    TypeVar,
    Union,
    runtime_checkable,
)


class SentinelReferenceError(ValueError):
    pass


@runtime_checkable
class SupportsLessThan(Protocol):
    def __lt__(self: Key, other: Key) -> bool:
        pass


Key = TypeVar("Key", bound=SupportsLessThan)
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
    @classmethod
    @abstractmethod
    def sentinel_class(cls) -> Type[SentinelType]:
        pass

    def sentinel(self) -> SentinelType:
        return self.sentinel_class()()

    def is_sentinel(self, node: Any) -> TypeGuard[SentinelType]:
        return isinstance(node, self.sentinel_class())


class MemberMixin(
    Generic[Key, NodeType],
    Container[Key],
):
    @abstractmethod
    def __contains__(self, key: Any) -> bool:
        pass

    @abstractmethod
    def access(self, key: Key) -> NodeType:
        pass

    def access_no_throw(self, key: Key) -> Optional[NodeType]:
        try:
            return self.access(key)
        except SentinelReferenceError:
            return None


@dataclass(slots=True)
class AbstractSentinel(
    Container,
    PrettyStrMixin,
    PrettyLineYieldMixin,
    ValidatorMixin,
    Sized,
    ABC,
):
    """Base class for all leaves in a tree."""

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


class TreeQueryMixin(Generic[Key, NodeType, SentinelType], ABC):
    @abstractmethod
    def minimum(self) -> NodeType:
        pass

    @abstractmethod
    def maximum(self) -> NodeType:
        pass

    @abstractmethod
    def successor(self, key: Key) -> Union[NodeType, SentinelType]:
        pass

    @abstractmethod
    def predecessor(self, key: Key) -> Union[NodeType, SentinelType]:
        pass


class TreeMutationMixin(Generic[Key, Value, NodeType], ABC):
    @abstractmethod
    def insert(self, key: Key, value: Value, allow_overwrite: bool = True) -> NodeType:
        """Insert a new node with the specified key and value into the tree.

        If allow_overwrite is True, then the value associated with the key will be
        overwritten if the key already exists in the tree.

        Parameters
        ----------
        key : Key
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
    def insert_node(self, node: NodeType, allow_overwrite: bool = True) -> bool:
        """
        Insert a new node with the specified key and value into the tree.

        Parameters
        ----------
        node: NodeType
            The node to insert into the tree.
        allow_overwrite : bool
            Whether to overwrite the value associated with the key if the key already
            exists in the tree.

        Returns
        -------
        bool
            Whether the node was inserted or not.
        """

    @abstractmethod
    def delete(self, key: Key) -> NodeType:
        """Delete the node with the specified key from the tree.


        Parameters
        ----------
        key : Key
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
    ) -> tuple[Key, Value]:
        """
        Extract the minimum node from the tree.

        Returns
        -------
        tuple[Key, Value]
            The key value tuple representing the minimum

        """

    @abstractmethod
    def extract_max(
        self,
    ) -> tuple[Key, Value]:
        """
        Extract the maximum node from the tree.

        Returns
        -------
        tuple[Key, Value]
            The key value tuple representing the maximum

        """


class NodeMutationMixin(Generic[Key, Value, NodeType, SentinelType], ABC):
    @abstractmethod
    def insert_node(self, node: NodeType, allow_overwrite: bool = True) -> NodeType:
        """Insert a new node with the specified key and value into the tree.

        If allow_overwrite is True, then the value associated with the key will be
        overwritten if the key already exists in the tree.

        Parameters
        ----------
        node: NodeType
            The node to insert into the tree.
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
    def delete_key(self, key: Key) -> Union[NodeType, SentinelType]:
        """Delete the node with the specified key from the tree.


        Parameters
        ----------
        key : Key
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
    ) -> tuple[tuple[Key, Value], Union[NodeType, SentinelType]]:
        """
        Extract the minimum node from the tree.

        Returns
        -------
        tuple[tuple[Key, Value], Union[NodeType, SentinelType]]
            The key value tuple representing the minimum
            and the new root of the tree

        """

    @abstractmethod
    def _extract_max(
        self,
    ) -> tuple[tuple[Key, Value], Union[NodeType, SentinelType]]:
        """
        Extract the maximum node from the tree.

        Returns
        -------
        tuple[tuple[Key, Value], Union[NodeType, SentinelType]]
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

    @abstractmethod
    def yield_edges(self) -> Iterator[tuple[NodeType, NodeType]]:
        pass


class NodeConstructorMixin(Generic[NodeType, Key, Value], ABC):
    @staticmethod
    @abstractmethod
    def node(key: Key, value: Value, *args, **kwargs) -> NodeType:
        pass

    @staticmethod
    @abstractmethod
    def is_node(node: Any) -> TypeGuard[NodeType]:
        pass


@dataclass(slots=True)
class AbstractNode(
    Generic[Key, Value, NodeType, SentinelType],
    NodeMutationMixin[Key, Value, NodeType, SentinelType],
    MutableMapping[Key, Value],
    TreeQueryMixin[Key, NodeType, SentinelType],
    TreeIterativeMixin[NodeType],
    MemberMixin[Key, NodeType],
    PrettyLineYieldMixin,
    PrettyStrMixin,
    ValidatorMixin,
    Sized,
    ABC,
):
    """Base class for all nodes in a tree."""


NodeWithParentType = TypeVar("NodeWithParentType", bound="SupportsParent")


@dataclass
class SupportsParent(Generic[NodeWithParentType, SentinelType], ABC):
    parent: Union[NodeWithParentType, SentinelType] = field(repr=False)


@dataclass(slots=True)
class AbstractTree(
    Generic[Key, Value, NodeType, SentinelType],
    TreeMutationMixin[Key, Value, NodeType],
    NodeConstructorMixin[NodeType, Key, Value],
    MutableMapping[Key, Value],
    TreeQueryMixin[Key, NodeType, SentinelType],
    TreeIterativeMixin[NodeType],
    MemberMixin[Key, NodeType],
    SentinelConstructorMixin[SentinelType],
    PrettyStrMixin,
    ValidatorMixin,
    Sized,
    ABC,
):
    """Base class for all trees."""

    root: Union[NodeType, SentinelType]
    size: int = 0

    def __init__(self, root: Optional[NodeType] = None):
        self.root = root if root is not None else self.sentinel()
        self.size = len(self.root)

    def clear(self):
        self.root = self.sentinel()
        self.size = 0

    def __len__(self):
        return self.size
