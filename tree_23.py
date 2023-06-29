from __future__ import annotations

from bisect import insort
from dataclasses import field, dataclass
from sys import maxsize
from typing import Any, Iterator, Optional, Type, TypeGuard, Union

from core import Key, Node, NodeValidationError, SentinelReferenceError, Tree, Value
from nodes import Sentinel


def all_equal(iterator: Iterator[Any]) -> bool:
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


@dataclass(slots=True, eq=True)
class Node23(Node[Key, Value, "Node23[Key, Value]", Sentinel]):
    content: list[tuple[Key, Value]]
    children: list[Node23[Key, Value]] | Sentinel = field(default_factory=Sentinel)
    parent: Node23[Key, Value] | Sentinel = field(default_factory=Sentinel)

    def is_2node(self):
        return len(self.content) == 1

    def is_3_node(self):
        return len(self.content) == 2

    def validate(
        self,
        lower_limit: Key,
        upper_limit: Key,
    ):
        match self.content:
            case [(key, _)]:
                if not (lower_limit < key < upper_limit):
                    raise NodeValidationError(f"2-node {self} violates BST property")
                match self.children:
                    case Sentinel():
                        pass
                    case (Node23() as left, Node23() as right):
                        assert left.parent == self
                        assert right.parent == self
                        left.validate(lower_limit, key)
                        right.validate(key, upper_limit)
                    case _:
                        raise NodeValidationError(
                            f"2-node has invalid children {self.children}"
                        )
            case [(smaller_key, _), (larger_key, _)]:
                if smaller_key > larger_key:
                    raise NodeValidationError(
                        f"3-node {self} violates 2-3 tree property"
                    )
                if not (lower_limit < smaller_key < upper_limit):
                    raise ValueError(f"3-node {self} violates 2-3 property")
                if not (lower_limit < larger_key < upper_limit):
                    raise NodeValidationError(f"3-node {self} violates 2-3 property")
                match self.children:
                    case Sentinel():
                        pass
                    case (Node23() as left, Node23() as middle, Node23() as right):
                        assert left.parent == self
                        assert middle.parent == self
                        assert right.parent == self
                        left.validate(lower_limit, smaller_key)
                        middle.validate(smaller_key, larger_key)
                        right.validate(larger_key, upper_limit)
                    case _:
                        raise NodeValidationError(
                            f"3-node has invalid children {self.children}"
                        )

    def __repr__(self):
        if self.is_2node():
            return f"({self.content[0][0]})"
        else:
            assert self.is_3_node()
            return f"({self.content[0][0]} : {self.content[1][0]})"

    def contains_key(self, key: Key) -> bool:
        for k, _ in self.content:
            if k == key:
                return True
        return False

    @property
    def is_leaf(self):
        return isinstance(self.children, Sentinel)

    def access(self, key: Key) -> Node23[Key, Value]:
        if self.contains_key(key):
            return self
        else:
            return self.drop_level(key).access(key)

    def __len__(self):
        return 1 + len(self.left) + len(self.right)

    def min_child(self) -> Node23[Key, Value] | Sentinel:
        if isinstance(self.children, list):
            return self.children[0]
        raise NodeValidationError("leaf")

    def max_child(self) -> Node23[Key, Value] | Sentinel:
        if isinstance(self.children, list):
            return self.children[-1]
        raise NodeValidationError("leaf")

    def nonnull_min_child(self) -> Node23[Key, Value]:
        if isinstance(min_child := self.min_child(), Node23):
            return min_child
        raise SentinelReferenceError("min child in null")

    def nonnull_max_child(self) -> Node23[Key, Value]:
        if isinstance(max_child := self.max_child(), Node23):
            return max_child
        raise SentinelReferenceError("max child in null")

    def nonnull_children(self) -> list[Node23[Key, Value]]:
        if isinstance(self.children, list):
            return self.children
        raise SentinelReferenceError("this is a leaf node with no children")

    def max_key(self) -> Key:
        return self.content[-1][0]

    def min_key(self) -> Key:
        return self.content[0][0]

    def minimum(self) -> Node23[Key, Value]:
        if isinstance(left := self.min_child(), Node23):
            return left.minimum()
        return self

    def maximum(self) -> Node23[Key, Value]:
        if isinstance(right := self.max_child(), Node23):
            return right.maximum()
        return self

    def trickle_down(self, key: Key) -> Node23[Key, Value]:
        assert not self.contains_key(key)

        if self.is_leaf:
            return self
        return self.drop_level(key).trickle_down(key)

    def drop_level(self, key: Key) -> Node23[Key, Value]:
        assert not self.contains_key(key)

        for (my_key, _), child in zip(self.content, self.nonnull_children()):
            if key < my_key:
                return child
        assert key > self.max_key()
        return self.nonnull_max_child()

    def insert_node(
        self, node: Node23[Key, Value], allow_overwrite: bool = True
    ) -> Node23[Key, Value]:
        raise NotImplementedError()

    def delete_key(self, key: Key) -> Union[Node23[Key, Value], Sentinel]:
        raise NotImplementedError()

    def extract_min(
        self,
    ) -> tuple[tuple[Key, Value], Union[Node23[Key, Value], Sentinel]]:
        raise NotImplementedError()

    def extract_max(
        self,
    ) -> tuple[tuple[Key, Value], Union[Node23[Key, Value], Sentinel]]:
        raise NotImplementedError()

    def delete_min(self) -> Node23[Key, Value]:
        raise NotImplementedError()

    def delete_max(self) -> Node23[Key, Value]:
        raise NotImplementedError()

    def __setitem__(self, key: Key, value: Value) -> None:
        raise NotImplementedError()

    def __delitem__(self, key: Key) -> None:
        raise NotImplementedError()

    def __getitem__(self, key: Key):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Key]:
        raise NotImplementedError()

    def successor(self, key: Key) -> Union[Node23[Key, Value], Sentinel]:
        raise NotImplementedError()

    def predecessor(self, key: Key) -> Union[Node23[Key, Value], Sentinel]:
        raise NotImplementedError()

    def inorder(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def preorder(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def postorder(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def level_order(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def yield_edges(self) -> Iterator[tuple[Node23[Key, Value], Node23[Key, Value]]]:
        raise NotImplementedError()

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----{self}\n"
        indent += "     " if prefix == "R" else "|    "
        if not self.is_leaf:
            children = self.nonnull_children()
            prefixes = ("L", "R") if self.is_2node() else ("L", "M", "R")
            for prefix, child in zip(prefixes, children):
                yield from child.yield_line(indent, prefix)

    def pretty_str(self) -> str:
        raise NotImplementedError()

    def is_unsupported_4node(self):
        return len(self.content) == 3


class Tree23(Tree[Key, Value, Node23[Key, Value], Sentinel]):
    def _try_overwrite(
        self, key: Key, new_value: Value, allow_overwrite: bool
    ) -> Optional[Node23[Key, Value]]:
        try:
            node = self.access(key)
            assert node.contains_key(key)
            if allow_overwrite:
                new_content = (
                    (k, new_value if k == key else val) for k, val in node.content
                )
                node.content = new_content  # type: ignore
                return node
            else:
                raise RuntimeError("Key already exists")
        except SentinelReferenceError:
            return None

    def insert(
        self, key: Key, value: Value, allow_overwrite: bool = True
    ) -> Node23[Key, Value]:
        if self.is_sentinel(self.root):
            return self._replace_root(self.node(key, value))

        if found := self._try_overwrite(key, value, allow_overwrite):
            return found

        leaf = self.nonnull_root.trickle_down(key)
        insort(leaf.content, (key, value))
        if leaf.is_unsupported_4node():
            self._balance_4node(leaf)
        self.size += 1
        return self.access(key)

    def _balance_4node(self, node: Node23[Key, Value]):
        assert node.is_unsupported_4node()
        (min_key, min_value), (mid_key, mid_value), (max_key, max_value) = node.content

        assert min_key < mid_key < max_key

        if self.is_sentinel(node.parent):
            assert node is self.root

            # since the node is the root, the new acceptor will be a new node
            # to replace the root
            acceptor = self.node(mid_key, mid_value)
        else:
            assert self.is_node(node.parent)

            # potential 4 node created
            insort(node.parent.content, (mid_key, mid_value))

            # the acceptor is just the parent
            acceptor = node.parent

        if isinstance(node.children, list):
            left_node_left, left_node_right = node.children[:2]
            right_node_left, right_node_right = node.children[2:]
            left_node = self.node(
                min_key,
                min_value,
                children=[left_node_left, left_node_right],
                parent=acceptor,
            )
            right_node = self.node(
                max_key,
                max_value,
                children=[right_node_left, right_node_right],
                parent=acceptor,
            )
            left_node_left.parent = left_node_right.parent = left_node
            right_node_left.parent = right_node_right.parent = right_node
        else:
            left_node = self.node(min_key, min_value, parent=acceptor)
            right_node = self.node(max_key, max_value, parent=acceptor)

        if self.is_sentinel(node.parent):
            acceptor.children = [left_node, right_node]
            self._replace_root(acceptor)
        else:
            assert isinstance(acceptor.children, list)
            acceptor.children.remove(node)
            acceptor.children.extend([left_node, right_node])
            acceptor.children.sort(key=Node23.min_key)

            if acceptor.is_unsupported_4node():
                self._balance_4node(acceptor)

    def delete(self, key: Key) -> Node23[Key, Value]:
        target_node = self.access(key)
        pass

    def extract_min(self) -> tuple[Key, Value]:
        raise NotImplementedError()

    def extract_max(self) -> tuple[Key, Value]:
        raise NotImplementedError()

    def delete_min(self) -> None:
        raise NotImplementedError()

    def delete_max(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def node(
        key: Key,
        value: Value,
        children: list[Node23[Key, Value]] | Sentinel = Sentinel(),
        parent: Union[Node23[Key, Value], Sentinel] = Sentinel(),
        *args,
        **kwargs,
    ) -> Node23[Key, Value]:
        return Node23[Key, Value](
            content=[(key, value)], children=children, parent=parent
        )

    @staticmethod
    def is_node(node: Any) -> TypeGuard[Node23[Key, Value]]:
        return isinstance(node, Node23)

    def __setitem__(self, key: Key, value: Value) -> None:
        raise NotImplementedError()

    def __delitem__(self, key: Key) -> None:
        raise NotImplementedError()

    def __getitem__(self, key: Key) -> Value:
        try:
            node = self.access(key)
            for k, v in node.content:
                if k == key:
                    return v
            raise RuntimeError("Implementation error")
        except SentinelReferenceError as e:
            raise KeyError from e

    def __iter__(self) -> Iterator[Key]:
        def traverse(node: Node23[Key, Value]) -> Iterator[Key]:
            match node.content:
                case ((key, _),):
                    match node.children:
                        case (left, right):
                            yield from traverse(left)
                            yield key
                            yield from traverse(right)
                        case Sentinel():
                            yield key
                case (smaller_key, _), (larger_key, _):
                    match node.children:
                        case (left, middle, right):
                            yield from traverse(left)
                            yield smaller_key
                            yield from traverse(middle)
                            yield larger_key
                            yield from traverse(right)
                        case Sentinel():
                            yield smaller_key
                            yield larger_key
                case _:
                    raise NodeValidationError(f"invalid node {node}")

        if self.is_node(self.root):
            yield from traverse(self.root)

    def minimum(self) -> Node23[Key, Value]:
        raise NotImplementedError()

    def maximum(self) -> Node23[Key, Value]:
        raise NotImplementedError()

    def successor(self, key: Key) -> Union[Node23[Key, Value], Sentinel]:
        raise NotImplementedError()

    def predecessor(self, key: Key) -> Union[Node23[Key, Value], Sentinel]:
        raise NotImplementedError()

    def inorder(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def preorder(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def postorder(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def level_order(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def yield_edges(self) -> Iterator[tuple[Node23[Key, Value], Node23[Key, Value]]]:
        raise NotImplementedError()

    def access(self, key: Key) -> Node23[Key, Value]:
        current = self.nonnull_root
        while self.is_node(current) and not current.contains_key(key):
            current = current.drop_level(key)
        if self.is_node(current):
            return current
        raise SentinelReferenceError(f"could not find {key}")

    def insert_node(
        self, node: Node23[Key, Value], allow_overwrite: bool = True
    ) -> bool:
        raise NotImplementedError()

    @classmethod
    def sentinel_class(cls) -> Type[Sentinel]:
        return Sentinel

    def pretty_str(self) -> str:
        return "".join(self.root.yield_line("", "R"))

    def validate(self, lower_limit: Key, upper_limit: Key) -> None:
        self._check_leaves_at_same_depth()
        self.root.validate(lower_limit, upper_limit)

    def _check_leaves_at_same_depth(self) -> None:
        def recurse(node: Union[Node23[Key, Value], Sentinel], depth: int) -> int:
            if self.is_sentinel(node):
                return depth
            else:
                assert self.is_node(node)
                if node.is_leaf:
                    leaf_depth = depth
                else:
                    children = iter(node.nonnull_children())
                    leaf_depth = recurse(next(children), depth)
                    assert all_equal(
                        recurse(child, depth + 1) == leaf_depth for child in children
                    )
                return leaf_depth + 1

        recurse(self.root, 0)

    def __contains__(self, item):
        try:
            self.access(item)
            return True
        except SentinelReferenceError:
            return False


if __name__ == "__main__":
    from random import randint

    for _ in range(1):
        tree: Tree23[int, None] = Tree23[int, None]()
        num_values = 1000
        values = list({randint(0, 1000) for _ in range(num_values)})
        # values = [96, 33, 99, 73, 45, 82, 51, 25, 30]

        for V in values:
            tree.insert(V, None, allow_overwrite=True)
            # print(tree.pretty_str())
            tree.validate(-maxsize, maxsize)
            assert V in tree

        # bst.dot()
        # print(bst.pretty_str())
        # for k in values:
        #     deleted = bst.delete(k)
        #     assert deleted.key == k, (k, deleted.key)
        #     print(bst.pretty_str())
        #     assert k not in bst, values
