from __future__ import annotations

from bisect import insort
from dataclasses import dataclass, field
from operator import itemgetter
from sys import maxsize
from typing import Any, Iterator, Optional, Type, TypeGuard, Union

from core import Key, Node, NodeValidationError, SentinelReferenceError, Tree, Value
from nodes import Sentinel

from more_itertools import one


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
            return 'hole'
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

    def minimum(self) -> Node23[Key, Value]:
        if self.is_leaf:
            return self
        return self.left.minimum()

    def maximum(self) -> Node23[Key, Value]:
        if self.is_leaf:
            return self
        return self.right.maximum()

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

    def _fixup_case3(self, hole: Node23[Key, Value]) -> Optional[Node23[Key, Value]]:
        parent = hole.parent
        assert self.is_node(parent) and parent.is_3node()
        # we have about 6 cases
        if hole is parent.left:
            if parent.middle.is_2node():
                middle = parent.middle
                # R----(287 : 830)
                #      L----hole -> (a)
                #      M----(466) -> (b, c)
                #      R----X
                # R----(830)
                #      M----(287: 466) -> (a, b, c)
                #      R----X
                middle.content.insert(0, parent.content.pop(0))
                if hole.children:
                    child = hole.nonnull_children.pop()
                    middle.nonnull_children.insert(0, child)
                    child.parent = middle
                parent.remove_child(hole)
            else:
                right = parent.right
                middle = parent.middle
                assert parent.right.is_2node()
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
                parent.remove_child(hole)
        elif hole is parent.middle:
            if parent.left.is_2node():
                left = parent.left
                assert self.is_node(left)
                # R----(317 : 754)
                #      L----(194) -> (a, b)
                #      M----hole -> c
                #      R----(941, 1000) -> (d, e, f?)
                # merge hole into a 2 node with L
                left.content.append(parent.content.pop(0))
                if left.children:
                    hole_child = hole.nonnull_children[0]
                    left.nonnull_children.append(hole_child)
                    hole_child.parent = left
                parent.remove_child(hole)
            else:
                right = parent.right
                assert parent.right.is_2node()
                # R----(269 : 376)
                #      L----X
                #      M----hole -> (a)
                #      R----(535) -> (b,c)
                right.content.insert(0, parent.content.pop())
                if hole.children:
                    child = hole.nonnull_children.pop()
                    right.nonnull_children.insert(0, child)
                    child.parent = right
                parent.remove_child(hole)
        else:
            assert hole is parent.right
            middle = parent.middle
            if parent.left.is_2node():
                assert self.is_node(middle)
                if parent.middle.is_3node():
                    # R----(317 : 754)
                    #      L----X
                    #      M----(350 : 541) -> (c, d, e)
                    #      R----hole -> f

                    # R----(317 : 541)
                    #      L----X
                    #      M----(350) -> (c, d)
                    #      R----(754) -> (e, f)
                    hole.content.append(parent.content.pop())
                    parent.content.append(middle.content.pop())
                    if hole.children:
                        child = middle.nonnull_children.pop()
                        hole.nonnull_children.insert(0, child)
                        child.parent = hole
                else:
                    # R----(317 : 541)
                    #      L----(243) -> (a, b)
                    #      M----(350) -> (c, d)
                    #      R----hole -> e

                    # R----(317)
                    #      L----(243) -> (a, b)
                    #      R----(350: 541) -> (c, d, e)
                    middle.content.append(parent.content.pop())
                    if hole.children:
                        child = hole.nonnull_children.pop()
                        middle.nonnull_children.append(child)
                        child.parent = middle
                    parent.remove_child(hole)
            else:
                assert parent.middle.is_2node()
                # R----(306 : 721)
                #      L---- X
                #      M----(504) -> a, b
                #      R----hole -> c

                # R----(306 : 721)
                #      L---- X
                #      R ----(504: 721) -> (a, b, c)
                middle.content.append(parent.content.pop())
                if hole.children:
                    child = hole.nonnull_children.pop()
                    middle.nonnull_children.append(child)
                    child.parent = middle
                parent.remove_child(hole)
        if not hole.children:
            hole.children = self.sentinel()

        return None

        # # case 2.3
        # # we have a 3-node parent
        # # remove one child
        # data: list[tuple[Key, Value]] = sum(
        #     [c.content for c in parent.nonnull_children],
        #     parent.content,
        # )
        # mid = len(data) // 2
        # data.sort(key=use_key)
        # parent.content = data[mid : mid + 1]
        # parent.children = [
        #     Node23[Key, Value](content=data[:mid], parent=parent),
        #     Node23[Key, Value](content=data[mid + 1 :], parent=parent),
        # ]
        # for child in parent.nonnull_children:
        #     child.parent = parent
        # return None

    def delete(self, key: Key) -> Value:
        target_node = self.access(key)
        value = target_node.access_value(key)

        if not target_node.is_leaf:
            # replace node with successor
            if target_node.is_3node():
                if key == target_node.min_key():
                    root_successor = target_node.middle.minimum()
                else:
                    root_successor = target_node.right.minimum()
            else:
                root_successor = target_node.right.minimum()
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
        else:
            assert target_node.is_2node() or target_node.is_hole()
            hole: Optional[Node23[Key, Value]] = target_node.to_hole()
            # underflow
            # replacement the 2-node with a hole
            assert target_node.is_hole()
            # hole is a leaf and is the root
            if hole is self.root:
                # case 0: The hole is the root
                self._clear_root()
                return value
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
                        else:
                            assert parent.is_3node()
                            # a < 47 < b < 259 < c < 362 < d < 527 < e < 552 < f < 964 < g
                            # R----(362 : 965)
                            #      L----(47 : 259) => (a, b, c)
                            #      M----(527 : 552) => (d, e, f)
                            #      R---- 'hole' => (g)

                            # R----(362 : 552)
                            #      L----(47 : 259) => (a, b, c)
                            #      M----(527) => (d, e)
                            #      R---- (964) => (f, g)
                            if hole is parent.right:
                                middle = parent.middle
                                right = parent.right
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
                                middle = parent.middle
                                right = parent.right
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
                                left = parent.left
                                middle = parent.middle
                                left.content.append(parent.content.pop(0))
                                parent.content.insert(0, middle.content.pop(0))
                                if isinstance(left.children, list) and isinstance(middle.children, list):
                                    child = middle.children.pop(0)
                                    left.children.append(child)
                                    child.parent = left
                            if not hole.children:
                                hole.children = self.sentinel()
                            hole = None
        self.size -= 1
        return value

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

    def minimum(self) -> Node23[Key, Value]:
        raise NotImplementedError()

    def maximum(self) -> Node23[Key, Value]:
        raise NotImplementedError()

    def successor(self, key: Key) -> Union[Node23[Key, Value], Sentinel]:
        raise NotImplementedError()

    def predecessor(self, key: Key) -> Union[Node23[Key, Value], Sentinel]:
        raise NotImplementedError()

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
    from random import randint, shuffle

    try:
        for _ in range(10000):
            tree: Tree23[int, None] = Tree23[int, None]()
            num_values = 100
            values = list({randint(0, 100000) for _ in range(num_values)})
            # values = [6, 966, 860, 903, 138, 267, 175, 914, 791, 474, 604, 701, 735]
            # values = [928, 227, 415, 969, 104, 459, 146, 693, 949, 438, 761, 955, 607]
            # values = [806, 648, 969, 746, 237, 925, 174, 721, 306, 504, 61]
            # values = [736, 322, 104, 554, 76, 269, 110, 535, 376, 315, 255]
            # print(values)
            # values = [807, 872, 849, 211, 757, 923, 476]
            for V in values:
                tree.insert(V, None, allow_overwrite=True)
                # print(tree.pretty_str())
                tree.validate(-maxsize, maxsize)
                assert V in tree
            shuffle(values)
            # print(tree.pretty_str())
            for c, V in enumerate(values):
                tree.delete(V)
                # print(tree.pretty_str())
                assert tree.size == (len(values) - (c + 1))
                tree.validate(-maxsize, maxsize)
                assert V not in tree
    except Exception as e:
        print(values)
        raise

        # tree.size = 11
        # n45 = Node23[int, None]([(45, None)])
        # n30 = Node23[int, None]([(30, None)])
        # n6082 = Node23[int, None]([(60, None), (82, None)])
        # n45.children = [n30, n6082]
        # n30.parent = n6082.parent = n45
        # tree.root = n45
        #
        # n22 = Node23[int, None]([(22, None)])
        # n3540 = Node23[int, None]([(35, None), (40, None)])
        # n30.children = [n22, n3540]
        # n22.parent = n3540.parent = n30
        #
        # n5155 = Node23[int, None]([(51, None), (55, None)])
        # n75 = Node23[int, None]([(75, None)])
        # n87 = Node23[int, None]([(87, None)])
        # n5155.parent = n75.parent = n87.parent = n6082
        # n6082.children = [n5155, n75, n87]

        # n45 = Node23[int, None]([(45, None)])
        # n3032 = Node23[int, None]([(30, None), (32, None)])
        # n31 = Node23[int, None]([(31, None)])
        # n22 = Node23[int, None]([(22, None)])
        # n3540 = Node23[int, None]([(35, None), (40, None)])
        # n3032.children = [n22, n31, n3540]
        # n60 = Node23[int, None]([(60, None)])
        # n31.parent = n22.parent = n3540.parent = n3032
        # n45.children = [n3032, n60]
        # n3032.parent = n60.parent = n45
        #
        # n51 = Node23[int, None]([(51, None)])
        # n75 = Node23[int, None]([(75, None)])
        # n60.children = [n51, n75]
        # n51.parent = n75.parent = n60

        # tree.root = n45
        #
        # # print(tree.dot())
        #
        # # tree.delete(22)
        # #
        # # print(tree.pretty_str())
        # #
        # tree.validate(-maxsize, maxsize)
        # # tree.dot()
        # print(tree.pretty_str())
        # tree.delete(22)
        # print(tree.pretty_str())
        # tree.validate(-maxsize, maxsize)
        # tree.delete(30)
        # print(tree.pretty_str())
        # tree.validate(-maxsize, maxsize)
        # tree.delete(55)
        # print(tree.pretty_str())
        # tree.validate(-maxsize, maxsize)
        # tree.delete(45)
        # print(tree.pretty_str())
        # tree.validate(-maxsize, maxsize)
        #
        # tree.delete(82)
        # print(tree.pretty_str())
        # tree.validate(-maxsize, maxsize)
        #
        # tree.delete(35)
        # print(tree.pretty_str())
        # tree.validate(-maxsize, maxsize)

        # tree.delete(82)
        # tree.validate(-maxsize, maxsize)
        # print(tree.pretty_str())
        # tree.delete(51)
        #
        # tree.validate(-maxsize, maxsize)
        # print(tree.pretty_str())

        # for k in values:
        #     deleted = tree.delete(k)
        #     print(tree.pretty_str())
        #     assert k not in tree, values
