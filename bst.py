from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Protocol,
    TypeVar,
    Optional,
    MutableMapping,
    Iterator,
    Self,
    runtime_checkable,
)

from typeguard import typechecked  # type: ignore


@runtime_checkable
class SupportsLessThan(Protocol):
    def __lt__(self: "Comparable", other: "Comparable") -> bool:
        pass


Comparable = TypeVar("Comparable", bound=SupportsLessThan)
Value = TypeVar("Value")
KeyValue = tuple[Comparable, Value]


class BinarySearchTreeNode(MutableMapping[Comparable, Value], ABC):
    @abstractmethod
    def insert(self, key: Comparable, value: Value) -> "Node":
        pass

    @abstractmethod
    def access(self, key: Comparable) -> "Node":
        pass

    @abstractmethod
    def delete(self, key: Comparable) -> "Node":
        pass

    @abstractmethod
    def inorder(self) -> Iterator["Internal[Comparable, Value]"]:
        pass

    @abstractmethod
    def preorder(self) -> Iterator["Internal[Comparable, Value]"]:
        pass

    @abstractmethod
    def postorder(self) -> Iterator["Internal[Comparable, Value]"]:
        pass

    @abstractmethod
    def level_order(self) -> Iterator["Internal[Comparable, Value]"]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __contains__(self, key: Comparable) -> bool:
        pass

    @abstractmethod
    def __getitem__(self, key: Comparable) -> Value:
        pass

    @abstractmethod
    def __setitem__(self, key: Comparable, value: Value) -> None:
        pass

    @abstractmethod
    def __delitem__(self, key: Comparable) -> None:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Comparable]:
        pass

    @abstractmethod
    def minimum(self) -> "Internal[Comparable, Value]":
        pass

    @abstractmethod
    def maximum(self) -> "Internal[Comparable, Value]":
        pass

    @abstractmethod
    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        pass

    @abstractmethod
    def check_invariants(self, lower_limit: Comparable, upper_limit: Comparable):
        pass

    @abstractmethod
    def successor(self, key: Comparable) -> "Internal[Comparable, Value]":
        pass

    @abstractmethod
    def predecessor(self, key: Comparable) -> "Internal[Comparable, Value]":
        pass


Node = BinarySearchTreeNode[Comparable, Value]


class Leaf(BinarySearchTreeNode[Comparable, Value]):
    def __iter__(self) -> Iterator[Comparable]:
        yield from ()

    def __repr__(self):
        return "Leaf()"

    def __str__(self):
        return "Leaf()"

    def insert(self, key: Comparable, value: Value) -> Self:
        return Internal(key, value)

    def access(self, key: Comparable) -> Optional[Value]:
        return None

    def delete(self, key: Comparable) -> Node:
        return self

    def inorder(self) -> Iterator["Internal[Comparable, Value]"]:
        yield from ()

    def preorder(self) -> Iterator["Internal[Comparable, Value]"]:
        yield from ()

    def postorder(self) -> Iterator["Internal[Comparable, Value]"]:
        yield from ()

    def level_order(self) -> Iterator["Internal[Comparable, Value]"]:
        yield from ()

    def __len__(self) -> int:
        return 0

    def __contains__(self, key: Comparable) -> bool:
        return False

    def __getitem__(self, key: Comparable) -> Value:
        raise ValueError(f"Leaf nodes hold no keys")

    def __setitem__(self, key: Comparable, value: Value) -> None:
        raise ValueError(f"Leaf nodes hold no keys")

    def __delitem__(self, key: Comparable) -> None:
        raise ValueError(f"Leaf nodes hold no keys")

    def minimum(self) -> "Internal[Comparable, Value]":
        raise ValueError("Empty tree has no minimum")

    def maximum(self) -> "Internal[Comparable, Value]":
        raise ValueError("Empty tree has no maximum")

    def successor(self, key: Comparable) -> "Internal[Comparable, Value]":
        raise ValueError("Empty tree has no successor")

    def predecessor(self, key: Comparable) -> "Internal[Comparable, Value]":
        raise ValueError("Empty tree has no predecessor")

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}Leaf"

    def check_invariants(self, lower_limit: Comparable, upper_limit: Comparable):
        return True


@dataclass(slots=True)
class Internal(BinarySearchTreeNode[Comparable, Value]):
    key: Comparable
    value: Value
    left: BinarySearchTreeNode[Comparable, Value] = field(default_factory=Leaf)
    right: BinarySearchTreeNode[Comparable, Value] = field(default_factory=Leaf)

    def __iter__(self) -> Iterator[Comparable]:
        for node in self.inorder():
            yield node.key

    def check_invariants(self, lower_limit: Comparable, upper_limit: Comparable):
        if not (lower_limit < self.key < upper_limit):
            return False
        return self.left.check_invariants(
            lower_limit, self.key
        ) and self.right.check_invariants(self.key, upper_limit)

    def insert(self, key: Comparable, value: Value) -> Self:
        if self.key == key:
            self.value = value
            return None
        elif key < self.key:
            self.left = self.left.insert(key, value)
        else:
            self.right = self.right.insert(key, value)
        return self

    def access(self, key: Comparable) -> BinarySearchTreeNode:
        if self.key == key:
            return self
        elif key < self.key:
            return self.left.access(key)
        else:
            return self.right.access(key)

    def delete(self, key: Comparable) -> "Node":
        if key < self.key:
            self.left = self.left.delete(key)
        elif key > self.key:
            self.right = self.right.delete(key)
        else:
            if isinstance(self.left, Leaf):
                return self.right
            elif isinstance(self.right, Leaf):
                return self.left
            else:
                successor = self.right.minimum()
                self.key, self.value = successor.key, successor.value
                self.right = self.right.delete(successor.key)
        return self

    def inorder(self) -> Iterator["Internal[Comparable, Value]"]:
        yield from self.left.inorder()
        yield self
        yield from self.right.inorder()

    def preorder(self) -> Iterator["Internal[Comparable, Value]"]:
        yield self
        yield from self.left.preorder()
        yield from self.right.preorder()

    def postorder(self) -> Iterator["Internal[Comparable, Value]"]:
        yield from self.left.postorder()
        yield from self.right.postorder()
        yield self

    def level_order(self) -> Iterator["Internal[Comparable, Value]"]:
        pass

    def __len__(self) -> int:
        return 1 + len(self.left) + len(self.right)

    def __contains__(self, key: Comparable) -> bool:
        return self.access(key) is not None

    def maximum(self) -> "Internal[Comparable, Value]":
        if isinstance(self.right, Leaf):
            return self
        else:
            return self.right.maximum()

    def minimum(self) -> "Internal[Comparable, Value]":
        if isinstance(self.left, Leaf):
            return self
        else:
            return self.left.minimum()

    def __lt__(self, other: "Internal[Comparable, Value]") -> bool:
        return self.key < other.key

    @property
    def has_no_children(self):
        return isinstance(self.left, Leaf) and isinstance(self.right, Leaf)

    def choose(self, key: Comparable) -> "BinarySearchTreeNode[Comparable, Value]":
        if key < self.key:
            return self.left
        else:
            return self.right

    def choose_set(
        self, key: Comparable, node: "BinarySearchTreeNode[Comparable, Value]"
    ):
        if key < self.key:
            self.left = node
        else:
            self.right = node

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----{self}\n"
        indent += "     " if prefix == "R" else "|    "
        yield from self.left.yield_line(indent, "L")
        yield from self.right.yield_line(indent, "R")

    def swap(self, other: "Internal[Comparable, Value]") -> None:
        self.key = other.key
        self.value = other.value
        self.left = other.left
        self.right = other.right

    def __getitem__(self, key: Comparable) -> Value:
        node = self.access(key)
        if isinstance(node, Internal):
            return node.value
        raise ValueError(f"Key {key} not found")

    def __setitem__(self, key: Comparable, value: Value) -> None:
        self.insert(key, value)

    def __delitem__(self, key: Comparable) -> None:
        self.delete(key)

    def successor(self, key: Comparable) -> "Internal[Comparable, Value]":
        if key > self.key:
            return self.right.successor(key)
        elif key < self.key:
            try:
                return self.left.successor(key)
            except ValueError:
                return self
        else:
            return self.right.minimum()

    def predecessor(self, key: Comparable) -> "Internal[Comparable, Value]":
        if key < self.key:
            return self.left.predecessor(key)
        elif key > self.key:
            try:
                return self.right.predecessor(key)
            except ValueError:
                return self
        else:
            return self.left.maximum()


@dataclass
class BinarySearchTree(BinarySearchTreeNode[Comparable, Value]):
    root: BinarySearchTreeNode[Comparable, Value] = field(default_factory=Leaf)
    size: int = 0

    def inorder(self) -> Iterator["Internal[Comparable, Value]"]:
        return self.root.inorder()

    def preorder(self) -> Iterator["Internal[Comparable, Value]"]:
        return self.root.preorder()

    def postorder(self) -> Iterator["Internal[Comparable, Value]"]:
        return self.root.postorder()

    def level_order(self) -> Iterator["Internal[Comparable, Value]"]:
        return self.root.level_order()

    def __len__(self) -> int:
        return self.size

    def __contains__(self, key: Comparable) -> bool:
        return key in self.root

    def __getitem__(self, key: Comparable) -> Value:
        return self.root.__getitem__(key)

    def __delitem__(self, key: Comparable) -> None:
        self.delete(key)

    def __setitem__(self, key: Comparable, value: Value) -> None:
        self.root.__setitem__(key, value)

    def minimum(self) -> "Internal[Comparable, Value]":
        return self.root.minimum()

    def maximum(self) -> "Internal[Comparable, Value]":
        return self.root.maximum()

    def insert(self, key: Comparable, value: Value) -> Self:
        if node := self.root.insert(key, value):
            self.root = node
            self.size += 1
        return self

    def access(self, key: Comparable) -> Optional[Value]:
        return self.root.access(key)

    def delete(self, key: Comparable) -> None:
        if key not in self.root:
            raise ValueError(f"Key {key} not found")
        else:
            self.root = self.root.delete(key)
            self.size -= 1

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Comparable]:
        for node in self.inorder():
            yield node.key

    def check_invariants(self, lower_limit: Comparable, upper_limit: Comparable):
        return self.root.check_invariants(lower_limit, upper_limit)

    def predecessor(self, key: Comparable) -> "Node":
        try:
            return self.root.predecessor(key)
        except ValueError as e:
            raise KeyError(f"No predecessor found, {key} is minimum key in tree") from e

    def successor(self, key: Comparable) -> "Node":
        try:
            return self.root.successor(key)
        except ValueError as e:
            raise KeyError(f"No successor found, {key} is maximum key in tree") from e


@dataclass
class BinarySearchTreeIterative(
    BinarySearchTree[Comparable, Value],
):
    root: Node = field(default_factory=Leaf)
    size: int = 0

    def __setitem__(self, key, value):
        self.insert(key, value)

    def level_order(self) -> Iterator["Internal[Comparable, Value]"]:
        return self.root.level_order()

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        raise NotImplementedError

    def check_invariants(self, lower_limit: Comparable, upper_limit: Comparable):
        return self.root.check_invariants(lower_limit, upper_limit)

    def access(self, key: Comparable) -> BinarySearchTreeNode:
        x = self.root
        while isinstance(x, Internal):
            if x.key == key:
                return x
            elif x.key < key:
                x = x.right
            else:
                x = x.left
        return x

    def access_ancestry(self, key: Comparable) -> Optional[list[Node]]:
        ancestry = []
        x = self.root
        while isinstance(x, Internal):
            ancestry.append(x)
            if x.key == key:
                return ancestry
            elif x.key < key:
                x = x.right
            else:
                x = x.left
        return None

    def set_child(
        self,
        ancestry: list[Internal[Comparable, Value]],
        node: Internal[Comparable, Value],
    ):
        if ancestry:
            ancestry[-1].choose_set(node.key, node)
        else:
            self.root = node

    def insert_ancestry(self, key: Comparable, value: Value) -> Optional[list[Node]]:
        ancestry, node = [], self.root

        while True:
            if isinstance(node, BinarySearchTreeNode):
                node = Internal(key, value)
                self.set_child(ancestry, node)
                ancestry.append(node)
                break
            elif node.key == key:
                node.value = value
                return None
            else:
                ancestry.append(node)
                node = node.choose(key)

        self.size += 1
        return ancestry

    def __contains__(self, item):
        return isinstance(self.access(item), Internal)

    def __delitem__(self, key):
        self.delete(key)

    def clear(self):
        self.root = Leaf()
        self.size = 0

    def __len__(self):
        return self.size

    def pretty_str(self):
        if isinstance(self.root, Leaf):
            return "Empty Tree"
        return "".join(self.root.yield_line("", "R"))

    @typechecked
    def parent(self, node: Internal[Comparable, Value]) -> "Node":
        current, parent = self.root, Leaf()
        while isinstance(current, Internal) and current is not node:
            parent = current
            current = current.choose(node.key)
        return parent

    def __iter__(self):
        yield from iter(self.root)

    def insert(self, key: Comparable, value: Value) -> Optional[Node]:
        node: Node = self.root
        parent: Node = Leaf()

        while True:
            if isinstance(node, Leaf):
                node = Internal(key, value)
                if isinstance(parent, Internal):
                    parent.choose_set(key, node)
                else:
                    self.root = node
                break
            else:
                assert isinstance(node, Internal)
                if node.key == key:
                    node.value = value
                    return None
                else:
                    parent = node
                    node = node.choose(key)
        self.size += 1

    def inorder(self) -> Iterator[Internal]:
        yield from self.root.inorder()

    def preorder(self) -> Iterator[Internal]:
        yield from self.root.preorder()

    def postorder(self) -> Iterator[Internal]:
        yield from self.root.postorder()

    def minimum(self):
        if isinstance(self.root, Leaf):
            raise ValueError("Empty tree has no minimum")
        current = self.root
        while isinstance(current.left, Internal):
            current = current.left
        return current

    def maximum(self):
        if isinstance(self.root, Leaf):
            raise ValueError("Empty tree has no maximum")
        current = self.root
        while isinstance(current.right, Internal):
            current = current.right
        return current

    @typechecked
    def extract_max(self, node: Internal) -> KeyValue:
        current = node
        parent: Node = self.parent(node)
        while isinstance(current.right, Internal):
            parent, current = current, current.right

        max_key_value = (current.key, current.value)

        if isinstance(current.left, Internal):
            current.swap(current.left)
        elif current is self.root and current.has_no_children:
            assert self.size == 1
            self.reset_root()
        else:
            assert isinstance(parent, Internal)
            parent.choose_set(current.key, Leaf())

        return max_key_value

    @typechecked
    def extract_min(self, node: Internal) -> KeyValue:
        current: Internal = node
        parent: Node = self.parent(node)
        while isinstance(current.left, Internal):
            parent, current = current, current.left
        min_key_value = (current.key, current.value)

        if isinstance(current.right, Internal):
            current.swap(current.right)
        elif current is self.root and current.has_no_children:
            assert self.size == 1
            self.reset_root()
        else:
            assert isinstance(parent, Internal)
            parent.choose_set(current.key, Leaf())

        return min_key_value

    @typechecked
    def successor(self, key: Comparable) -> Node:
        node = self.access(key)
        if isinstance(node, Internal):
            if isinstance(node.right, Internal):
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
            if candidate is None:
                raise KeyError(f"Key {key} has no successor")
            return candidate
        else:
            raise ValueError(f"Key {key} not found")

    @typechecked
    def predecessor(self, key: Comparable) -> Node:
        node = self.access(key)
        if isinstance(node, Internal):
            if isinstance(node.left, Internal):
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
            if candidate is None:
                raise KeyError(f"Key {key} has no predecessor")
            return candidate
        else:
            raise ValueError(f"Key {key} not found")

    def delete(self, target_key: Comparable):
        target_node = self.access(target_key)
        if isinstance(target_node, Internal):
            if isinstance(target_node.right, Internal) and isinstance(
                target_node.left, Internal
            ):
                # swap key with successor and delete using extract_min
                key, value = self.extract_min(target_node.right)
                target_node.key, target_node.value = key, value

            elif isinstance(target_node.right, Internal):
                target_node.swap(target_node.right)

            elif isinstance(target_node.left, Internal):
                target_node.swap(target_node.left)
            else:
                # the target node is the root, and it is a leaf node, then setting the root to None will suffice
                if target_node == self.root:
                    self.reset_root()
                else:
                    assert target_node.has_no_children
                    parent = self.parent(target_node)
                    assert isinstance(parent, Internal)
                    # else the target is a leaf node which has a parent
                    parent.choose_set(target_node.key, Leaf())

            self.size -= 1
        else:
            raise ValueError(f"Key {target_key} not found")

    def reset_root(self):
        self.root = Leaf()

    def __repr__(self):
        return repr(self.root)


if __name__ == "__main__":
    # bst: BinarySearchTreeIterative = BinarySearchTreeIterative()
    # values = [
    #     3,
    #     52,
    #     31,
    #     55,
    #     93,
    #     60,
    #     81,
    #     93,
    #     46,
    #     37,
    #     47,
    #     67,
    #     34,
    #     95,
    #     10,
    #     23,
    #     90,
    #     14,
    #     13,
    #     88,
    # ]
    #
    # for i, val in enumerate(values):
    #     bst.insert(val, None)
    #     print(values[i] in bst)
    #     assert 101 not in bst
    # print(bst.pretty_str())
    from random import randint
    from sys import maxsize

    for _ in range(50):
        values = [-1172, -8059, 3159]
        bst = BinarySearchTree[int, None]()
        for val in values:
            bst.insert(val, None)
            assert val in bst
            bst.check_invariants(-maxsize, maxsize)

        for val in values:
            del bst[val]
            assert val not in bst, values
            bst.check_invariants(-maxsize, maxsize)

        assert not bst
        assert len(bst) == 0
        assert isinstance(bst.root, Leaf)
