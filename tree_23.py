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


use_key = itemgetter(0)


def insort_lists(A, B, key=None):
    A.extend(B)
    if key is None:
        A.sort()
    else:
        A.sort(key=key)


@dataclass(slots=True, eq=True)
class Node23(Node[Key, Value, "Node23[Key, Value]", Sentinel]):
    content: list[tuple[Key, Value]]
    children: list[Node23[Key, Value]] | Sentinel = field(default_factory=Sentinel)
    parent: Node23[Key, Value] | Sentinel = field(default_factory=Sentinel)

    def is_2node(self):
        return len(self.content) == 1

    def is_3node(self):
        return len(self.content) == 2

    def is_hole(self):
        return self.content == [] and len(self.children) <= 1

    def to_hole(self, node: Optional[Node23[Key, Value]] = None) -> Node23:
        self.content = []
        if node is not None:
            self.children = [node]
        else:
            self.children = []
        return self

    def remove_key(self, key: Key) -> Optional[Value]:
        new_content = []
        to_ret = None
        for k, v in self.content:
            if k != key:
                new_content.append((k, v))
            else:
                to_ret = v
        self.content = new_content
        return to_ret

    def remove_child(self, child: Node23[Key, Value]):
        self.nonnull_children.remove(child)
        if not self.children:
            self.children = Sentinel()

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
        if self.is_hole():
            return "hole"
        if self.is_2node():
            return f"({self.content[0][0]})"
        else:
            assert self.is_3node()
            return f"({self.content[0][0]} : {self.content[1][0]})"

    def contains_key(self, key: Key) -> bool:
        for k, _ in self.content:
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
        if self.parent.is_3node():
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
        if self.is_2node():
            return self.nonnull_children[1]
        elif self.is_3node():
            return self.nonnull_children[2]
        raise ValueError("left only defined for 2- and 3-nodes")

    @property
    def middle(self) -> Node23[Key, Value]:
        if self.is_2node():
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
        return self.content[-1][0]

    def min_key(self) -> Key:
        return self.content[0][0]

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

        for (my_key, _), child in zip(self.content, self.nonnull_children):
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
        return self.minimum_node().content[0]

    def maximum(self) -> tuple[Key, Value]:
        return self.maximum_node().content[-1]

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
            prefixes = ("L", "R") if self.is_2node() else ("L", "M", "R")
            for prefix, child in zip(prefixes, self.nonnull_children):
                yield from child.yield_line(indent, prefix)

    def pretty_str(self) -> str:
        raise NotImplementedError()

    def is_unsupported_4node(self):
        return len(self.content) == 3

    def access_value(self, key: Key) -> Value:
        try:
            for k, v in self.content:
                if k == key:
                    return v
            raise KeyError(f"Key {key} not found locally")
        except SentinelReferenceError as e:
            raise KeyError from e


class Tree23(Tree[Key, Value, Node23[Key, Value], Sentinel]):
    def _try_overwrite(
        self, key: Key, new_value: Value, allow_overwrite: bool
    ) -> Optional[Node23[Key, Value]]:
        try:
            node = self.access(key)
            assert node.contains_key(key)
            if allow_overwrite:
                new_content = [
                    (k, new_value if k == key else val) for k, val in node.content
                ]
                node.content = new_content
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

    def _fixup_case2(self, hole: Node23[Key, Value]) -> Optional[Node23[Key, Value]]:
        assert hole.is_hole()
        parent = hole.parent
        assert self.is_node(parent) and parent.is_2node()
        if hole is parent.left:
            sibling = parent.right
            hole.content = parent.content
            parent.content = sibling.content[:1]
            sibling.content = sibling.content[1:]
            if isinstance(sibling.children, list):
                insort_lists(hole.children, sibling.children[:1], key=Node23.min_key)
                sibling.children[0].parent = hole
                sibling.children = sibling.children[1:]
        else:
            sibling = parent.left
            hole.content = parent.content
            parent.content = sibling.content[-1:]
            sibling.content = sibling.content[:-1]
            if isinstance(sibling.children, list):
                insort_lists(hole.children, sibling.children[-1:], key=Node23.min_key)
                sibling.children[-1].parent = hole
                sibling.children = sibling.children[:-1]
        if not hole.children:
            hole.children = self.sentinel()

        return None

    def _fixup_case1(self, hole: Node23[Key, Value]) -> Optional[Node23[Key, Value]]:
        parent = hole.parent
        assert self.is_node(parent) and parent.is_2node()

        sibling = hole.sibling()
        insort_lists(sibling.content, parent.content, key=use_key)
        if sibling.children:
            insort_lists(sibling.children, hole.children, key=Node23.min_key)
        for child in hole.nonnull_children:
            child.parent = sibling
        return parent.to_hole(sibling)

    def _maybe_transfer_child_from_hole(
        self, hole: Node23[Key, Value], dest: Node23[Key, Value], *, at_start: bool
    ) -> None:
        assert hole.is_hole()
        if not self.is_sentinel(dest.children):
            child = hole.nonnull_children.pop()
            if at_start:
                dest.nonnull_children.insert(0, child)
            else:
                dest.nonnull_children.append(child)
            child.parent = dest

    def _fixup_case3(self, hole: Node23[Key, Value]) -> Optional[Node23[Key, Value]]:
        parent = hole.parent
        assert self.is_node(parent) and parent.is_3node()
        left, middle, right = parent.left, parent.middle, parent.right

        # we have about 6 cases
        if hole is left:
            if middle.is_2node():
                # R----(287 : 830)
                #      L----hole -> (a)
                #      M----(466) -> (b, c)
                #      R----X
                # R----(830)
                #      M----(287: 466) -> (a, b, c)
                #      R----X
                middle.content.insert(0, parent.content.pop(0))
                self._maybe_transfer_child_from_hole(hole, middle, at_start=True)
            else:
                assert right.is_2node()
                # R----(303 : 785)
                #      L----hole -> a
                #      M----(374 : 700) -> (b, c, d)
                #      R----(980) -> (e, f)

                # R----(700)
                #      L----(303 : 374) -> (a, b, c)
                #      R----(785 : 980) -> (d, e, f)

                right.content.insert(0, parent.content.pop())
                middle.content.insert(0, parent.content.pop())
                parent.content.append(middle.content.pop())
                if hole.children:
                    child_mid = hole.nonnull_children.pop()
                    child_mid.parent = middle
                    middle.nonnull_children.insert(0, child_mid)

                    child_right = middle.nonnull_children.pop()
                    child_right.parent = right
                    right.nonnull_children.insert(0, child_right)
        elif hole is middle:
            if left.is_2node():
                # R----(317 : 754)
                #      L----(194) -> (a, b)
                #      M----hole -> (c)
                #      R----X
                # R----
                left.content.append(parent.content.pop(0))
                self._maybe_transfer_child_from_hole(hole, left, at_start=False)
            else:
                assert right.is_2node()
                # R----(269 : 376)
                #      L----X
                #      M----hole -> (a)
                #      R----(535) -> (b, c)
                right.content.insert(0, parent.content.pop())
                self._maybe_transfer_child_from_hole(hole, right, at_start=True)
        else:
            assert hole is right
            if middle.is_2node():
                # R----(306 : 721)
                #      L---- X
                #      M----(504) -> (a, b)
                #      R----hole -> (c)

                # R----(306 : 721)
                #      L---- X
                #      R ----(504: 721) -> (a, b, c)
                middle.content.append(parent.content.pop())
                self._maybe_transfer_child_from_hole(hole, middle, at_start=False)
            else:
                assert left.is_2node()
                # R----(317 : 754)
                #      L----(200) -> (a, b)
                #      M----(350 : 541) -> (c, d, e)
                #      R----hole -> f

                # R----(350)
                #      L----(200: 317) -> (a, b, c)
                #      M----(541: 754) -> (d, e, f)

                middle.content.append(parent.content.pop())
                left.content.append(parent.content.pop())
                parent.content.append(middle.content.pop(0))
                if hole.children:
                    child_mid = middle.nonnull_children.pop(0)
                    child_mid.parent = left
                    left.nonnull_children.append(child_mid)

                    child = hole.nonnull_children.pop()
                    child.parent = middle
                    middle.nonnull_children.append(child)
        parent.remove_child(hole)
        return None

    def _fixup_case4(self, hole: Node23[Key, Value]) -> Optional[Node23[Key, Value]]:
        # a < 47 < b < 259 < c < 362 < d < 527 < e < 552 < f < 964 < g
        # R----(362 : 965)
        #      L----(47 : 259) => (a, b, c)
        #      M----(527 : 552) => (d, e, f)
        #      R---- 'hole' => (g)

        # R----(362 : 552)
        #      L----(47 : 259) => (a, b, c)
        #      M----(527) => (d, e)
        #      R---- (964) => (f, g)
        parent = hole.parent
        assert self.is_node(parent)
        left, middle, right = parent.left, parent.middle, parent.right

        if hole is parent.right:
            right.content.append(parent.content.pop())
            parent.content.append(middle.content.pop())
            if isinstance(middle.children, list) and isinstance(right.children, list):
                child = middle.children.pop()
                right.children.insert(0, child)
                child.parent = right
        # a < 211 < b < 476 < c < 757 < d < 849 < e < 872 < f < 923 < g
        # R----(757 : 849)
        #      L----(211 : 476) => (a, b, c)
        #      M---- 'hole' => (d)
        #      R----(872 : 923) => (e, f, g)

        # R----(757 : 872)
        #      L---- (211 : 476) => (a, b, c)
        #      M---- (849) => (d, e)
        #      R---- (923) => (f, g)

        elif hole is parent.middle:
            middle.content.append(parent.content.pop())
            parent.content.append(right.content.pop(0))
            if isinstance(middle.children, list) and isinstance(right.children, list):
                child = right.children.pop(0)
                middle.children.append(child)
                child.parent = middle
        # a < 172 < b < 275 < c < 409 < d < 678 < e < 684 < f < 926 < g
        # R----(172 : 678)
        #      L---- 'hole' => (a)
        #      M----(275 : 409) => (b, c, d)
        #      R----(684 : 926) => (e, f, g)

        # R----(275 : 678)
        #      L----(172) => (a, b)
        #      M----(409) => (c, d)
        #      R----(684 : 926) => (e, f, g)
        else:
            assert hole is parent.left
            left.content.append(parent.content.pop(0))
            parent.content.insert(0, middle.content.pop(0))
            if isinstance(left.children, list) and isinstance(middle.children, list):
                child = middle.children.pop(0)
                left.children.append(child)
                child.parent = left
        if not hole.children:
            hole.children = self.sentinel()

        return None

    def delete(self, key: Key) -> Value:
        target_node = self.access(key)
        value = target_node.access_value(key)

        if not target_node.is_leaf:
            # replace node with successor
            if target_node.is_3node():
                if key == target_node.min_key():
                    root_successor = target_node.middle.minimum_node()
                else:
                    root_successor = target_node.right.minimum_node()
            else:
                root_successor = target_node.right.minimum_node()
            # assert root.successor.is_leaf and is_sorted(root.successor.content, key=use_key)
            successor_key, successor_value = root_successor.content.pop(0)
            target_node.remove_key(key)
            insort(target_node.content, (successor_key, successor_value), key=use_key)
            target_node = root_successor

            if target_node.is_2node():
                self.size -= 1
                return value

        assert target_node.is_leaf

        if target_node.is_3node():
            # Easy case. Replace 3-node by 2-node
            target_node.remove_key(key)
            self.size -= 1
        else:
            assert target_node.is_2node() or target_node.is_hole()
            self._remove_hole(target_node.to_hole())
        return value

    def _remove_hole(self, hole: Optional[Node23[Key, Value]]) -> None:
        # hole is a leaf and is the root
        if hole is self.root:
            # case 0: The hole is the root
            self._clear_root()
            return
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
                    if parent.is_2node() and hole.sibling().is_2node():
                        hole = self._fixup_case1(hole)

                    # case 2: The hole has a 2-node as a parent and a 3-node as a sibling
                    elif parent.is_2node() and hole.sibling().is_3node():
                        hole = self._fixup_case2(hole)

                    # case 3: The hole has a 3-node as a parent with a 2-node as a sibling
                    elif parent.is_3node() and any(
                        node.is_2node() for node in parent.nonnull_children
                    ):
                        hole = self._fixup_case3(hole)
                    # case 4: The hole has a 3-node as a parent with only 3-node siblings
                    else:
                        hole = self._fixup_case4(hole)
        self.size -= 1

    def extract_min(self) -> tuple[Key, Value]:
        min_node = self.minimum_node()
        assert min_node.is_leaf

        keyval = min_node.content.pop(0)
        if not min_node.content:
            self._remove_hole(min_node.to_hole())
        else:
            self.size -= 1

        return keyval

    def extract_max(self) -> tuple[Key, Value]:
        max_node = self.maximum_node()
        assert max_node.is_leaf

        keyval = max_node.content.pop()
        if not max_node.content:
            self._remove_hole(max_node.to_hole())
        else:
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
            for k, v in self.access(key).content:
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
        return self.minimum_node().content[0]

    def maximum(self) -> tuple[Key, Value]:
        return self.maximum_node().content[-1]

    def successor_node(self, key: Key) -> Node23[Key, Value]:
        raise NotImplementedError()

    def predecessor_node(self, key: Key) -> Union[Node23[Key, Value], Sentinel]:
        raise NotImplementedError()

    def successor(self, key: Key) -> tuple[Key, Value]:
        try:
            node = self.access(key)
            if node.is_leaf:
                if node.is_3node() and key == node.min_key():
                    return node.content[1]
                p = node.parent
                while self.is_node(p) and p.right is node:
                    node = p
                    p = p.parent
                if self.is_node(p):
                    if p.left is node:
                        return p.content[0]
                    elif p.is_3node() and p.middle is node:
                        return p.content[1]
                else:
                    if node.is_3node() and key == node.min_key():
                        return node.content[1]
                raise KeyError(f"{key=} has no successor")
            else:
                if node.is_3node() and key == node.min_key():
                    return node.middle.minimum()
                else:
                    return node.right.minimum()
        except SentinelReferenceError as e:
            raise KeyError(f"{key=} not found") from e

    def predecessor(self, key: Key) -> tuple[Key, Value]:
        try:
            node = self.access(key)
            if node.is_leaf:
                if node.is_3node() and key == node.max_key():
                    return node.content[0]
                p = node.parent
                while self.is_node(p) and p.left is node:
                    node = p
                    p = p.parent
                if self.is_node(p):
                    if p.right is node:
                        return p.content[-1]
                    elif p.is_3node() and p.middle is node:
                        return p.content[0]
                else:
                    if node.is_3node() and key == node.max_key():
                        return node.content[0]
                raise KeyError(f"{key=} has no successor")
            else:
                if node.is_3node() and key == node.max_key():
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


if __name__ == "__main__":
    pass
