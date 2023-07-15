from __future__ import annotations

from bisect import insort
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any, Iterator, Optional, Type, TypeGuard, Union

from more_itertools import one

from core import Key, Node, NodeValidationError, SentinelReferenceError, Tree, Value
from nodes import Sentinel


def all_equal(iterator: Iterator[Any]) -> bool:
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def insort_lists(A, B, key):
    A.extend(B)
    A.sort(key=key)


@dataclass(slots=True, eq=True)
class Node23(Node[Key, Value, "Node23[Key, Value]", Sentinel]):
    data: list[tuple[Key, Value]]
    children: list[Node23[Key, Value]] | Sentinel = field(default_factory=Sentinel)
    parent: Node23[Key, Value] | Sentinel = field(default_factory=Sentinel)

    @property
    def is_2node(self):
        return len(self.data) == 1

    @property
    def is_3node(self):
        return len(self.data) == 2

    @property
    def is_hole(self):
        return self.data == [] and len(self.children) <= 1

    def to_hole(self, node: Optional[Node23[Key, Value]] = None) -> Node23:
        self.data = []
        if node is not None:
            self.children = [node]
        else:
            self.children = []
        return self

    def remove_item(self, key: Key) -> Optional[Value]:
        updated_data = []
        removed_value = None
        for _key, value in self.data:
            if _key != key:
                updated_data.append((_key, value))
            else:
                removed_value = value
        self.data = updated_data
        return removed_value

    def remove_child(self, child: Node23[Key, Value]):
        self.nonnull_children.remove(child)
        if not self.children:
            self.children = Sentinel()

    def validate(
        self,
        lower_limit: Key,
        upper_limit: Key,
    ):
        match self.data:
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
        if self.is_hole:
            return "hole"
        if self.is_2node:
            return f"({self.data[0][0]})"
        else:
            assert self.is_3node
            return f"({self.data[0][0]} : {self.data[1][0]})"

    def contains_key(self, key: Key) -> bool:
        for k, _ in self.data:
            if k == key:
                return True
        return False

    @property
    def is_leaf(self):
        return isinstance(self.children, Sentinel)

    def sibling(self) -> Node23[Key, Value]:
        if isinstance(self.parent, Sentinel):
            raise ValueError("root has no sibling")
        assert isinstance(self.parent, Node23)
        if self.parent.is_3node:
            raise ValueError("siblings in a 3-node are not well defined sibling")
        if self.parent.left == self:
            return self.parent.right
        else:
            return self.parent.left

    @property
    def left(self) -> Node23[Key, Value]:
        return self.nonnull_children[0]

    @property
    def right(self) -> Node23[Key, Value]:
        if self.is_2node:
            return self.nonnull_children[1]
        elif self.is_3node:
            return self.nonnull_children[2]
        raise ValueError("left only defined for 2- and 3-nodes")

    @property
    def middle(self) -> Node23[Key, Value]:
        if self.is_2node:
            raise ValueError("middle only defined for 3-nodes")
        return self.nonnull_children[1]

    def access(self, key: Key) -> Node23[Key, Value]:
        if self.contains_key(key):
            return self
        else:
            return self.drop_level(key).access(key)

    def __len__(self):
        return 1 + len(self.left) + len(self.right)

    @property
    def nonnull_children(self) -> list[Node23[Key, Value]]:
        if isinstance(self.children, list):
            return self.children
        raise SentinelReferenceError("this is a leaf node with no children")

    def max_key(self) -> Key:
        return self.data[-1][0]

    def min_key(self) -> Key:
        return self.data[0][0]

    def minimum_node(
        self, _: Optional[Node23[Key, Value]] = None
    ) -> Node23[Key, Value]:
        if self.is_leaf:
            return self
        return self.left.minimum_node()

    def maximum_node(
        self, _: Optional[Node23[Key, Value]] = None
    ) -> Node23[Key, Value]:
        if self.is_leaf:
            return self
        return self.right.maximum_node()

    def trickle_down(self, key: Key) -> Node23[Key, Value]:
        assert not self.contains_key(key)

        if self.is_leaf:
            return self
        return self.drop_level(key).trickle_down(key)

    def drop_level(self, key: Key) -> Node23[Key, Value]:
        assert not self.contains_key(key)

        for (my_key, _), child in zip(self.data, self.nonnull_children):
            if key < my_key:
                return child
        assert key > self.max_key()
        return self.right

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

    def successor_node(self, key: Key) -> Union[Node23[Key, Value], Sentinel]:
        raise NotImplementedError()

    def predecessor_node(self, key: Key) -> Union[Node23[Key, Value], Sentinel]:
        raise NotImplementedError()

    def successor(self, key: Key) -> tuple[Key, Value]:
        raise NotImplementedError()

    def predecessor(self, key: Key) -> tuple[Key, Value]:
        raise NotImplementedError()

    def inorder(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def preorder(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def postorder(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def level_order(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def minimum(self) -> tuple[Key, Value]:
        return self.minimum_node().data[0]

    def maximum(self) -> tuple[Key, Value]:
        return self.maximum_node().data[-1]

    def yield_edges(self) -> Iterator[tuple[Node23[Key, Value], Node23[Key, Value]]]:
        if isinstance(self.children, list):
            for child in self.children:
                yield self, child
                if isinstance(child, Node23):
                    yield from child.yield_edges()

    def yield_nodes(self):
        if isinstance(self.children, list):
            for child in self.children:
                yield child
                if isinstance(child, Node23):
                    yield from child.yield_nodes()
        yield self

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----{self}\n"
        indent += "     " if prefix == "R" else "|    "
        if not self.is_leaf:
            prefixes = ("L", "R") if self.is_2node else ("L", "M", "R")
            for prefix, child in zip(prefixes, self.nonnull_children):
                yield from child.yield_line(indent, prefix)

    def pretty_str(self) -> str:
        raise NotImplementedError()

    def is_unsupported_4node(self):
        return len(self.data) == 3

    def access_value(self, key: Key) -> Value:
        try:
            for k, v in self.data:
                if k == key:
                    return v
            raise KeyError(f"Key {key} not found locally")
        except SentinelReferenceError as e:
            raise KeyError from e

    def insort_data(self, data: list[tuple[Key, Value]]):
        self.data.extend(data)
        self.data.sort(key=itemgetter(0))

    def insort_children(self, children: list[Node23[Key, Value]]):
        self.nonnull_children.extend(children)
        self.nonnull_children.sort(key=lambda x: x.min_key())
        for child in children:
            child.parent = self


class Tree23(Tree[Key, Value, Node23[Key, Value], Sentinel]):
    def _try_overwrite(
        self, key: Key, new_value: Value, allow_overwrite: bool
    ) -> Optional[Node23[Key, Value]]:
        try:
            node = self.access(key)
            assert node.contains_key(key)
            if allow_overwrite:
                new_content = [
                    (k, new_value if k == key else val) for k, val in node.data
                ]
                node.data = new_content
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
        insort(leaf.data, (key, value))

        if leaf.is_unsupported_4node():
            self._balance_4node(leaf)

        self.size += 1
        return self.access(key)

    def _balance_4node(self, node: Node23[Key, Value]):
        assert node.is_unsupported_4node()
        (min_key, min_value), (mid_key, mid_value), (max_key, max_value) = node.data

        assert min_key < mid_key < max_key

        if self.is_sentinel(node.parent):
            assert node is self.root

            # since the node is the root, the new acceptor will be a new node
            # to replace the root
            acceptor = self.node(mid_key, mid_value)
        else:
            assert self.is_node(node.parent)

            # potential 4 node created
            insort(node.parent.data, (mid_key, mid_value))

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
            self._replace_root(acceptor, 0)
        else:
            assert isinstance(acceptor.children, list)
            acceptor.children.remove(node)
            acceptor.children.extend([left_node, right_node])
            acceptor.children.sort(key=Node23.min_key)  # type: ignore

            if acceptor.is_unsupported_4node():
                self._balance_4node(acceptor)

    def delete_node(self, key: Key) -> Node23[Key, Value]:
        raise NotImplementedError(
            "getting exact node where key was deleted "
            "is not supported because nodes share keys"
        )

    def _maybe_move_child(
        self,
        src: Node23[Key, Value],
        dst: Node23[Key, Value],
        *,
        at_start: bool,
        first_child: bool = False,
    ) -> None:
        if not self.is_sentinel(src.children) and not self.is_sentinel(dst.children):
            if first_child:
                child = src.nonnull_children.pop(0)
            else:
                child = src.nonnull_children.pop()
            if at_start:
                dst.nonnull_children.insert(0, child)
            else:
                dst.nonnull_children.append(child)
            child.parent = dst

    def _fixup_case1(self, hole: Node23[Key, Value]) -> Optional[Node23[Key, Value]]:
        parent = hole.parent
        assert self.is_node(parent) and parent.is_2node

        sibling = hole.sibling()
        sibling.insort_data(parent.data)
        if sibling.children:
            sibling.insort_children(hole.nonnull_children)
        return parent.to_hole(sibling)

    def _fixup_case2(self, hole: Node23[Key, Value]) -> Optional[Node23[Key, Value]]:
        assert hole.is_hole
        parent = hole.parent
        assert self.is_node(parent) and parent.is_2node
        if hole is parent.left:
            sibling = parent.right
            hole.data = parent.data
            parent.data = sibling.data[:1]
            sibling.data = sibling.data[1:]
            self._maybe_move_child(sibling, hole, at_start=False, first_child=True)
        else:
            sibling = parent.left
            hole.data = parent.data
            parent.data = sibling.data[-1:]
            sibling.data = sibling.data[:-1]
            self._maybe_move_child(sibling, hole, at_start=True, first_child=False)
        if not hole.children:
            hole.children = self.sentinel()

        return None

    def _fixup_case3(self, hole: Node23[Key, Value]) -> Optional[Node23[Key, Value]]:
        parent = hole.parent
        assert self.is_node(parent) and parent.is_3node
        left, middle, right = parent.left, parent.middle, parent.right

        if hole is left:
            if middle.is_2node:
                middle.data.insert(0, parent.data.pop(0))
                self._maybe_move_child(hole, middle, at_start=True)
            else:
                assert right.is_2node
                right.data.insert(0, parent.data.pop())
                middle.data.insert(0, parent.data.pop())
                parent.data.append(middle.data.pop())
                self._maybe_move_child(hole, middle, at_start=True)
                self._maybe_move_child(middle, right, at_start=True)
        elif hole is middle:
            if left.is_2node:
                left.data.append(parent.data.pop(0))
                self._maybe_move_child(hole, left, at_start=False)
            else:
                assert right.is_2node
                right.data.insert(0, parent.data.pop())
                self._maybe_move_child(hole, right, at_start=True)
        else:
            assert hole is right
            if middle.is_2node:
                middle.data.append(parent.data.pop())
                self._maybe_move_child(hole, middle, at_start=False)
            else:
                assert left.is_2node
                middle.data.append(parent.data.pop())
                left.data.append(parent.data.pop())
                parent.data.append(middle.data.pop(0))
                self._maybe_move_child(middle, left, at_start=False, first_child=True)
                self._maybe_move_child(hole, middle, at_start=False)
        parent.remove_child(hole)
        return None

    def _fixup_case4(self, hole: Node23[Key, Value]) -> Optional[Node23[Key, Value]]:
        parent = hole.parent
        assert self.is_node(parent)
        left, middle, right = parent.left, parent.middle, parent.right

        if hole is parent.right:
            right.data.append(parent.data.pop())
            parent.data.append(middle.data.pop())
            self._maybe_move_child(middle, right, at_start=True, first_child=False)

        elif hole is parent.middle:
            middle.data.append(parent.data.pop())
            parent.data.append(right.data.pop(0))
            self._maybe_move_child(right, middle, at_start=False, first_child=True)
        else:
            assert hole is parent.left
            left.data.append(parent.data.pop(0))
            parent.data.insert(0, middle.data.pop(0))
            self._maybe_move_child(middle, left, at_start=False, first_child=True)

        # seal the hole if it does not have any children
        if not hole.children:
            hole.children = self.sentinel()

        return None

    def _replace_internal_node_with_successor(
        self, internal_node: Node23[Key, Value], key: Key
    ) -> Node23[Key, Value]:
        # replace node with successor
        assert not internal_node.is_leaf
        if internal_node.is_3node and key == internal_node.min_key():
            root_successor = self.minimum_node(internal_node.middle)
        else:
            root_successor = self.minimum_node(internal_node.right)
        # assert root.successor.is_leaf and is_sorted(root.successor.content, key=itemgetter(0))
        successor_key, successor_value = root_successor.data.pop(0)
        internal_node.remove_item(key)
        internal_node.insort_data([(successor_key, successor_value)])
        return root_successor

    def delete(self, key: Key) -> Value:
        target_node = self.access(key)
        value = target_node.access_value(key)

        if not target_node.is_leaf:
            target_node = self._replace_internal_node_with_successor(target_node, key)

            if target_node.is_2node:
                self.size -= 1
                return value

        assert target_node.is_leaf

        if target_node.is_3node:
            # Easy case. Replace 3-node by 2-node
            target_node.remove_item(key)
        else:
            assert target_node.is_2node or target_node.is_hole
            self._remove_hole(target_node.to_hole())

        self.size -= 1
        return value

    def _remove_hole(self, hole: Optional[Node23[Key, Value]]) -> None:
        # hole is a leaf and is the root
        if hole is self.root:
            # case 0: The hole is the root
            self._clear_root()
        else:
            while hole is not None:
                if hole is self.root:
                    # case 0: The hole is the root
                    new_root = one(hole.nonnull_children)
                    new_root.parent = self.sentinel()
                    self._replace_root(new_root, 0)
                    break
                else:
                    parent = hole.parent
                    assert self.is_node(parent)

                    # case 1: The hole has a 2-node as a parent and a 2-node as a sibling
                    if parent.is_2node and hole.sibling().is_2node:
                        hole = self._fixup_case1(hole)

                    # case 2: The hole has a 2-node as a parent and a 3-node as a sibling
                    elif parent.is_2node and hole.sibling().is_3node:
                        hole = self._fixup_case2(hole)

                    # case 3: The hole has a 3-node as a parent with a 2-node as a sibling
                    elif parent.is_3node and any(
                        node.is_2node for node in parent.nonnull_children
                    ):
                        hole = self._fixup_case3(hole)
                    # case 4: The hole has a 3-node as a parent with only 3-node siblings
                    else:
                        hole = self._fixup_case4(hole)

    def extract_min(self) -> tuple[Key, Value]:
        min_node = self.minimum_node()
        assert min_node.is_leaf

        keyval = min_node.data.pop(0)
        if not min_node.data:
            self._remove_hole(min_node.to_hole())

        self.size -= 1

        return keyval

    def extract_max(self) -> tuple[Key, Value]:
        max_node = self.maximum_node()
        assert max_node.is_leaf

        keyval = max_node.data.pop()
        if not max_node.data:
            self._remove_hole(max_node.to_hole())

        self.size -= 1

        return keyval

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
        return Node23[Key, Value](data=[(key, value)], children=children, parent=parent)

    @staticmethod
    def is_node(node: Any) -> TypeGuard[Node23[Key, Value]]:
        return isinstance(node, Node23)

    def __setitem__(self, key: Key, value: Value) -> None:
        raise NotImplementedError()

    def __delitem__(self, key: Key) -> None:
        raise NotImplementedError()

    def __getitem__(self, key: Key) -> Value:
        try:
            for k, v in self.access(key).data:
                if k == key:
                    return v
            raise RuntimeError("Implementation error")
        except SentinelReferenceError as e:
            raise KeyError from e

    def __iter__(self) -> Iterator[Key]:
        def traverse(node: Node23[Key, Value]) -> Iterator[Key]:
            match node.data:
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

    def minimum_node(
        self, root: Optional[Node23[Key, Value]] = None
    ) -> Node23[Key, Value]:
        current = self.nonnull_root if root is None else root
        while not current.is_leaf:
            current = current.left
        return current

    def maximum_node(
        self, root: Optional[Node23[Key, Value]] = None
    ) -> Node23[Key, Value]:
        current = self.nonnull_root if root is None else root
        while not current.is_leaf:
            current = current.right
        return current

    def minimum(self) -> tuple[Key, Value]:
        return self.minimum_node().data[0]

    def maximum(self) -> tuple[Key, Value]:
        return self.maximum_node().data[-1]

    def successor_node(self, key: Key) -> Node23[Key, Value]:
        raise NotImplementedError()

    def predecessor_node(self, key: Key) -> Union[Node23[Key, Value], Sentinel]:
        raise NotImplementedError()

    def successor(self, key: Key) -> tuple[Key, Value]:
        try:
            node = self.access(key)
            if node.is_leaf:
                if node.is_3node and key == node.min_key():
                    return node.data[1]
                p = node.parent
                while self.is_node(p) and p.right is node:
                    node = p
                    p = p.parent
                if self.is_node(p):
                    if p.left is node:
                        return p.data[0]
                    elif p.is_3node and p.middle is node:
                        return p.data[1]
                else:
                    if node.is_3node and key == node.min_key():
                        return node.data[1]
                raise KeyError(f"{key=} has no successor")
            else:
                if node.is_3node and key == node.min_key():
                    return node.middle.minimum()
                else:
                    return node.right.minimum()
        except SentinelReferenceError as e:
            raise KeyError(f"{key=} not found") from e

    def predecessor(self, key: Key) -> tuple[Key, Value]:
        try:
            node = self.access(key)
            if node.is_leaf:
                if node.is_3node and key == node.max_key():
                    return node.data[0]
                p = node.parent
                while self.is_node(p) and p.left is node:
                    node = p
                    p = p.parent
                if self.is_node(p):
                    if p.right is node:
                        return p.data[-1]
                    elif p.is_3node and p.middle is node:
                        return p.data[0]
                else:
                    if node.is_3node and key == node.max_key():
                        return node.data[0]
                raise KeyError(f"{key=} has no successor")
            else:
                if node.is_3node and key == node.max_key():
                    return node.middle.maximum()
                else:
                    return node.left.maximum()
        except SentinelReferenceError as e:
            raise KeyError(f"{key=} not found") from e

    def preorder(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def postorder(self) -> Iterator[Node23[Key, Value]]:
        raise NotImplementedError()

    def inorder(self) -> Iterator[Node23[Key, Value]]:
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
        raise NotImplementedError(
            "insert node not implemented for this class "
            "because of the ambiguity of a node"
        )

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
                    children = iter(node.nonnull_children)
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

    def dot(self, output_file_path: str = "tree23.pdf"):
        return draw_tree_23(self.nonnull_root, output_file_path)


def draw_tree_23(
    root: Union[Sentinel, Node23],
    output_filename: str = "tree23.pdf",
):
    from nodes import create_graph_pdf, escape, graph_epilogue, graph_prologue

    graph = [graph_prologue()]
    edges = []
    nodes = []

    if isinstance(root, Sentinel):
        nodes.append(
            f"   {id(root)} [shape=record, style=filled, fillcolor=black, "
            f'fontcolor=white, label="{escape(str(root))}"];'
        )
    else:
        for node in root.yield_nodes():
            nodes.append(
                f"   {id(node)} [shape=record, style=filled, fillcolor=black, "
                f'fontcolor=white, label="{escape(str(node))}"];'
            )

        for src, dst in root.yield_edges():
            edges.append(
                f"{(id(src))}:from_false -> {(id(dst))}:from_node [arrowhead=vee] "
            )

    graph.extend(edges)
    graph.extend(nodes)
    graph.append(graph_epilogue())

    create_graph_pdf(graph, output_filename=output_filename)
