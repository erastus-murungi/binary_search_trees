from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Generic,
    Protocol,
    TypeVar,
    Optional,
    MutableMapping,
    Iterator,
    Type,
    Self,
    runtime_checkable,
)

from typeguard import typechecked  # type: ignore
import operator as op


@runtime_checkable
class SupportsLessThan(Protocol):
    def __lt__(self: "Comparable", other: "Comparable") -> bool:
        pass


Comparable = TypeVar("Comparable", bound=SupportsLessThan)
Value = TypeVar("Value")
KeyValue = tuple[Comparable, Value]


@dataclass(slots=True)
class InternalNode(Generic[Comparable, Value], ABC):
    key: Comparable
    value: Value
    left: Optional["InternalNode"] = None
    right: Optional["InternalNode"] = None

    @typechecked
    def __lt__(self, other: "InternalNode[Comparable, Value]") -> bool:
        return self.key < other.key

    def insert(self, key: Comparable, value: Value) -> Self:
        if key < self.key:
            if isinstance(self.left, InternalNode):
                self.left.insert(key, value)
            else:
                self.left = self.__class__(key, value)
        elif key == self.key:
            self.value = value
        else:
            if isinstance(self.right, InternalNode):
                self.right.insert(key, value)
            else:
                self.right = self.__class__(key, value)
        return self

    def access(self, key: Comparable) -> "Node":
        if key == self.key:
            return self
        elif key < self.key:
            return self.left.access(key) if self.left else None
        else:
            return self.right.access(key) if self.right else None

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    def choose(self, key: Comparable) -> "Node":
        if key < self.key:
            return self.left
        else:
            return self.right

    def choose_set(self, key: Comparable, node: "Node") -> "Node":
        if key < self.key:
            self.left = node
            return self.left
        else:
            self.right = node
            return self.right

    def minimum(self) -> "Node":
        return self.left.minimum() if isinstance(self.left, InternalNode) else self

    def maximum(self) -> "Node":
        return self.right.maximum() if isinstance(self.right, InternalNode) else self

    def inorder(self):
        if self.left:
            yield from self.left.inorder()
        yield self
        if self.right:
            yield from self.right.inorder()

    def preorder(self):
        yield self
        if self.left:
            yield from self.left.preorder()
        if self.right:
            yield from self.right.preorder()

    def postorder(self):
        if self.left:
            yield from self.left.postorder()
        if self.right:
            yield from self.right.postorder()
        yield self

    def aggregate(self, f, g, initial):
        return g(
            self,
            f(
                self.left.aggregate(f, g, initial) if self.left else initial,
                self.right.aggregate(f, g, initial) if self.right else initial,
            ),
        )

    def height(self):
        return self.aggregate(max, lambda _, y: 1 + y, 0)

    def s_value(self):
        return self.aggregate(min, op.add, 0)

    def __len__(self):
        return (
            1
            + (0 if self.left is None else len(self.left))
            + (0 if self.right is None else len(self.right))
        )

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----{self}\n"
        indent += "     " if prefix == "R" else "|    "
        if self.left:
            yield from self.left.yield_line(indent, "L")
        else:
            yield f"{indent}{prefix}----{self}\n"
        if self.right:
            yield from self.right.yield_line(indent, "R")
        else:
            yield f"{indent}{prefix}----{self}\n"

    def swap(self, other: "InternalNode") -> None:
        self.key = other.key
        self.value = other.value
        self.left = other.left
        self.right = other.right

    def check_bst_property(
        self, lower_limit: Comparable, upper_limit: Comparable
    ) -> bool:
        def valid(
            node: Node, lower_limit_: Comparable, upper_limit_: Comparable
        ) -> bool:
            if not node:
                return True
            if not (lower_limit_ < node.key < upper_limit_):
                return False
            return valid(node.left, lower_limit_, node.key) and valid(
                node.right, node.key, upper_limit_
            )

        return valid(self, lower_limit, upper_limit)

    def __iter__(self):
        yield from (node.key for node in self.inorder())


NodeClassType = TypeVar("NodeClassType", bound=InternalNode)
Node = Optional[NodeClassType]


class NodeClassMixin(ABC, Generic[NodeClassType]):
    @abstractmethod
    def node_class(self) -> Type[NodeClassType]:
        pass


class BST(
    Generic[Comparable, Value, NodeClassType],
    MutableMapping[Comparable, Value],
    NodeClassMixin[InternalNode],
):
    def __init__(self):
        self.root: Node = None
        self.size = 0

    @property
    def node_class(self) -> Type[InternalNode]:
        return InternalNode

    def __setitem__(self, key, value):
        self.insert(key, value)

    def access(self, key: Comparable) -> "Node":
        x = self.root
        while x is not None:
            if x.key == key:
                return x
            elif x.key < key:
                x = x.right
            else:
                x = x.left
        return None

    def __contains__(self, item):
        return self.access(item) is not None

    def __getitem__(self, item):
        x = self.access(item)
        if x is None:
            raise KeyError(f"{item} not found")
        else:
            return x.value

    def __delitem__(self, key):
        self.delete(key)

    def clear(self):
        self.root = None
        self.size = 0

    def __len__(self):
        return self.size

    def pretty_str(self):
        if self.root is None:
            return "Empty Tree"
        return "".join(self.root.yield_line("", "R"))

    @typechecked
    def parent(self, node: InternalNode) -> "Node":
        current, parent = self.root, None
        while current is not None and current is not node:
            parent = current
            current = current.choose(node.key)
        return parent

    def __iter__(self):
        yield from iter(self.root)

    def insert(self, key: Comparable, value: Value) -> Node:
        if self.root is None:
            self.root = self.node_class(key, value)
            self.size = 1
            return self.root
        else:
            current: Node = self.root
            parent: InternalNode = self.root

            while current is not None and not current.is_leaf:
                if current.key == key:
                    current.value = value
                    return current

                parent = current
                current = current.choose(key)

            self.size += 1
            if current is None:
                return parent.choose_set(key, self.node_class(key, value))
            else:
                return current.choose_set(key, self.node_class(key, value))

    def inorder(self) -> Iterator[InternalNode]:
        if self.root is not None:
            yield from self.root.inorder()

    def preorder(self) -> Iterator[InternalNode]:
        if self.root is not None:
            yield from self.root.preorder()

    def postorder(self) -> Iterator[InternalNode]:
        if self.root is not None:
            yield from self.root.postorder()

    def iteritems(self) -> Iterator[tuple[Comparable, Value]]:
        yield from ((node.key, node.value) for node in self.inorder())

    def minimum(self):
        if self.root is None:
            return None

        current = self.root
        while current.left is not None:
            current = current.left
        return current

    def maximum(self):
        if self.root is None:
            return None

        current = self.root
        while current.right is not None:
            current = current.right
        return current

    @typechecked
    def extract_max(self, node: InternalNode) -> KeyValue:
        current: InternalNode = node
        parent: Node = self.parent(node)
        while current.right is not None:
            parent, current = current, current.right

        max_key_value = (current.key, current.value)

        if current.left:
            current.swap(current.left)
        elif current is self.root and current.is_leaf:
            assert self.size == 1
            self.root = None
        else:
            assert parent is not None
            parent.choose_set(current.key, None)

        return max_key_value

    @typechecked
    def extract_min(self, node: InternalNode) -> KeyValue:
        current: InternalNode = node
        parent: Node = self.parent(node)
        while current.left is not None:
            parent, current = current, current.left
        min_key_value = (current.key, current.value)

        if current.right:
            current.swap(current.right)
        elif current is self.root and current.is_leaf:
            assert self.size == 1
            self.root = None
        else:
            assert parent is not None
            parent.choose_set(current.key, None)

        return min_key_value

    @typechecked
    def successor(self, key: Comparable) -> Node:
        if (node := self.access(key)) is None:
            raise ValueError(f"Key {key} not found")
        if node.right:
            return node.right.minimum()
        else:
            candidate = None
            current = self.root
            while current != node:
                if current.key > node.key:
                    candidate = current
                    current = current.left
                else:
                    current = current.right
        return candidate

    @typechecked
    def predecessor(self, key: Comparable) -> Node:
        if (node := self.access(key)) is None:
            raise ValueError(f"Key {key} not found")
        if node.left:
            return node.left.maximum()
        else:
            candidate = None
            current = self.root
            while current != node:
                if current.key < node.key:
                    candidate = current
                    current = current.right
                else:
                    current = current.left
        return candidate

    def delete(self, target_key: Comparable):
        target_node = self.access(target_key)
        if target_node is None:
            raise ValueError(f"Key {target_key} not found")

        if target_node.right and target_node.left:
            # swap key with successor and delete using extract_min
            key, value = self.extract_min(target_node.right)
            target_node.key, target_node.value = key, value

        elif target_node.right:
            target_node.swap(target_node.right)

        elif target_node.left:
            target_node.swap(target_node.left)
        else:
            # the target node is the root, and it is a leaf node, then setting the root to None will suffice
            if target_node == self.root:
                self.root = None
            else:
                assert target_node.is_leaf
                # else the target is a leaf node which has a parent
                self.parent(target_node).choose_set(target_node.key, None)

        self.size -= 1

    def __repr__(self):
        return repr(self.root)


if __name__ == "__main__":
    bst: BST = BST()
    values = [
        3,
        52,
        31,
        55,
        93,
        60,
        81,
        93,
        46,
        37,
        47,
        67,
        34,
        95,
        10,
        23,
        90,
        14,
        13,
        88,
    ]

    for i, val in enumerate(values):
        bst.insert(val, None)
        print(values[i] in bst)
        assert 101 not in bst

    print(bst.pretty_str())
