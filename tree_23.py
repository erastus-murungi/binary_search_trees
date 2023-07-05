from __future__ import annotations

from bisect import insort
from dataclasses import field, dataclass
from sys import maxsize
from typing import Any, Iterator, Optional, Type, TypeGuard, Union

from core import Key, Node, NodeValidationError, SentinelReferenceError, Tree, Value
from nodes import Sentinel
from more_itertools import partition


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

    def min_child(self) -> Node23[Key, Value]:
        if isinstance(self.children, list):
            return self.children[0]
        raise NodeValidationError("leaf")

    def max_child(self) -> Node23[Key, Value]:
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
        if self.is_leaf:
            return self
        return self.min_child().minimum()

    def maximum(self) -> Node23[Key, Value]:
        if self.is_leaf:
            return self
        return self.max_child().maximum()

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
        if target_node.is_leaf:
            if target_node.is_3_node():
                new_content = [(k, v) for k, v in target_node.content if k != key]
                target_node.content = new_content
            else:
                # underflow
                assert target_node.is_2node()
                # case 2.1
                # The removed node's sibling is a 3-node
                if self.is_sentinel(target_node.parent):
                    assert target_node is self.root
                    raise NotImplementedError()
                else:
                    parent = target_node.parent
                    assert self.is_node(parent)
                    if parent.is_2node():
                        if parent.min_child() is target_node:
                            if parent.max_child().is_3_node():
                                # case 2.1
                                #       30                               35
                                #     /   \        delete 22 -------->  /  \
                                #   22   35 : 40                       30  40

                                root_content, right_content = parent.max_child().content
                                (left_content,) = parent.content
                                parent.content = [root_content]
                                parent.children = [
                                    self.node(*left_content, parent=parent),
                                    self.node(*right_content, parent=parent),
                                ]
                            else:
                                # case 2.2
                                #
                                grand_parent = parent.parent
                                if self.is_node(grand_parent):
                                    if grand_parent.is_2node():
                                        # first replace the root with the minimum on the right
                                        if parent is grand_parent.min_child():
                                            root_successor = (
                                                grand_parent.max_child().minimum()
                                            )
                                            if root_successor.is_3_node():
                                                # make successor a 2-node
                                                parent.max_child().content.append(
                                                    grand_parent.content.pop(0)
                                                )
                                                # swap root with successor
                                                grand_parent.content.append(
                                                    root_successor.content.pop(0)
                                                )
                                                # back to case 2.1
                                                self.delete(key)
                                        else:
                                            assert parent is grand_parent.max_child()
                                            root_predecessor = (
                                                grand_parent.min_child().maximum()
                                            )
                                            if root_predecessor.is_3_node():
                                                # make predecessor a 2-node
                                                parent.min_child().content.insert(
                                                    0, grand_parent.content.pop(0)
                                                )
                                                # swap root with predecessor
                                                grand_parent.content.append(
                                                    root_predecessor.content.pop(-1)
                                                )
                                                # back to case 2.1
                                                self.delete(key)

                                    elif grand_parent.is_3_node():
                                        raise NotImplementedError()
                                    else:
                                        raise NotImplementedError()
                                else:
                                    raise NotImplementedError()
                        elif parent.max_child() is target_node:
                            if parent.min_child().is_2node():
                                # case 2.1
                                #       30                               20
                                #     /   \        delete 50 -------->  /  \
                                #  10: 20  50                         10    30
                                left_content, root_content = parent.min_child().content
                                (right_content,) = parent.content
                                parent.content = [root_content]
                                parent.children = [
                                    self.node(*left_content, parent=parent),
                                    self.node(*right_content, parent=parent),
                                ]
                            else:
                                raise NotImplementedError()
                    else:
                        assert parent.is_3_node()
                        # case 2.3
                        # we have a 3-node parent
                        # remove one child
                        data: list[tuple[Key, Value]] = sum(
                            [
                                c.content
                                for c in parent.nonnull_children()
                                if c is not target_node
                            ],
                            parent.content,
                        )
                        data.sort(key=lambda x: x[0])
                        parent.content = data[1:2]
                        parent.children = [
                            Node23[Key, Value](content=data[:1], parent=parent),
                            Node23[Key, Value](content=data[2:], parent=parent),
                        ]
        else:
            # replace node with successor
            root_successor = (
                target_node.max_child().minimum()
            )
            suc_cv = root_successor.content[0]
            self.delete(suc_cv[0])
            target_rest, _ = partition(
                lambda x: x[0] == key, target_node.content
            )
            target_node.content = list(target_rest)
            insort(target_node.content, suc_cv)
        return target_node

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

    def dot(self, output_file_path: str = "tree23.pdf"):
        return draw_tree_23(self.nonnull_root, output_file_path)


def draw_tree_23(
    root: Union[Sentinel, Node23],
    output_filename: str = "tree23.pdf",
):
    from nodes import graph_prologue, escape, graph_epilogue, create_graph_pdf

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

    for _ in range(1):
        tree: Tree23[int, None] = Tree23[int, None]()
        num_values = 11
        values = list({randint(0, 1000) for _ in range(num_values)})
        # values = [22, 30, 35, 40, 45, 60, 82, 51, 55, 75, 87]
        #
        # for V in values:
        #     tree.insert(V, None, allow_overwrite=True)
        #     # print(tree.pretty_str())
        #     tree.validate(-maxsize, maxsize)
        #     assert V in tree

        tree.size = 11
        n45 = Node23[int, None]([(45, None)])
        n30 = Node23[int, None]([(30, None)])
        n6082 = Node23[int, None]([(60, None), (82, None)])
        n45.children = [n30, n6082]
        n30.parent = n6082.parent = n45
        tree.root = n45

        n22 = Node23[int, None]([(22, None)])
        n3540 = Node23[int, None]([(35, None), (40, None)])
        n30.children = [n22, n3540]
        n22.parent = n3540.parent = n30

        n5155 = Node23[int, None]([(51, None), (55, None)])
        n75 = Node23[int, None]([(75, None)])
        n87 = Node23[int, None]([(87, None)])
        n5155.parent = n75.parent = n87.parent = n6082
        n6082.children = [n5155, n75, n87]

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

        tree.root = n45
        tree.validate(-maxsize, maxsize)

        # print(tree.dot())

        # tree.delete(22)
        #
        # print(tree.pretty_str())
        #
        tree.validate(-maxsize, maxsize)
        # tree.dot()
        # print(tree.pretty_str())
        tree.delete(22)
        tree.delete(30)
        print(tree.pretty_str())
        tree.delete(55)
        tree.delete(82)
        tree.validate(-maxsize, maxsize)
        print(tree.pretty_str())
        tree.delete(51)

        tree.validate(-maxsize, maxsize)
        print(tree.pretty_str())

        # for k in values:
        #     deleted = tree.delete(k)
        #     print(tree.pretty_str())
        #     assert k not in tree, values
