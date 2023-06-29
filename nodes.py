from __future__ import annotations

import os
import subprocess
import sys
from abc import ABC
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    Hashable,
    Iterator,
    Optional,
    TypeGuard,
    TypeVar,
    Union,
    cast,
)

from core import (
    AbstractSentinel,
    Key,
    Node,
    SentinelReferenceError,
    SentinelType,
    SupportsParent,
    Value,
)

BinaryNodeType = TypeVar("BinaryNodeType", bound="AbstractBSTNode")
BinaryNodeWithParentType = TypeVar(
    "BinaryNodeWithParentType", bound="AbstractBSTNodeWithParent"
)


class Sentinel(AbstractSentinel, Hashable):
    def __hash__(self):
        return hash("Sentinel")


class AbstractSentinelWithParent(
    AbstractSentinel,
    SupportsParent[BinaryNodeWithParentType, SentinelType],
    ABC,
):
    pass


@dataclass
class AbstractBSTNode(Node[Key, Value, BinaryNodeType, SentinelType], Hashable, ABC):
    key: Key
    value: Value
    left: Union[BinaryNodeType, SentinelType] = field(repr=False)
    right: Union[BinaryNodeType, SentinelType] = field(repr=False)

    def __hash__(self):
        return hash(self.key)

    def choose(self, key: Key) -> Union[BinaryNodeType, SentinelType]:
        if key > self.key:
            return self.right
        elif key < self.key:
            return self.left
        else:
            raise ValueError(f"Key {key} already exists in tree")

    def choose_set(self, key: Key, node: BinaryNodeType) -> None:
        if key > self.key:
            self.right = node
        elif key < self.key:
            self.left = node
        else:
            raise ValueError(f"Key {key} already exists in tree")

    def pretty_str(self) -> str:
        return "".join(self.yield_line("", "R"))

    def dot(self, output_file_path: str = "tree.pdf"):
        return draw_tree(self, output_file_path)

    def validate(self, lower_limit: Key, upper_limit: Key) -> None:
        if not (lower_limit < self.key < upper_limit):
            raise ValueError(f"Node {self} violates BST property")
        self.left.validate(lower_limit, self.key)
        self.right.validate(self.key, upper_limit)

    def __len__(self) -> int:
        return 1 + len(self.left) + len(self.right)

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        yield f"{indent}{prefix}----{self}\n"
        indent += "     " if prefix == "R" else "|    "
        yield from self.left.yield_line(indent, "L")
        yield from self.right.yield_line(indent, "R")

    def __contains__(self, key: Any) -> bool:
        node = self.access_no_throw(cast(Key, key))
        return isinstance(node, type(self)) and node.key == key

    def access(self, key: Key) -> BinaryNodeType:
        if self.key == key:
            return cast(BinaryNodeType, self)
        elif key < self.key:
            if self._is_node(self.left):
                return self.left.access(key)
            raise SentinelReferenceError(
                f"Key {key} not found, Sentinel {self.left} reached"
            )
        else:
            if self._is_node(self.right):
                return self.right.access(key)
            raise SentinelReferenceError(
                f"Key {key} not found, Sentinel {self.right} reached"
            )

    def minimum(self) -> BinaryNodeType:
        if self._is_node(self.left):
            return self.left.minimum()
        return cast(BinaryNodeType, self)

    def maximum(self) -> BinaryNodeType:
        if self._is_node(self.right):
            return self.right.maximum()
        return cast(BinaryNodeType, self)

    @property
    def nonnull_right(self) -> BinaryNodeType:
        if self._is_node(self.right):
            return self.right
        raise SentinelReferenceError(f"Node {self.right} is not a BinarySearchTreeNode")

    @property
    def nonnull_left(self) -> BinaryNodeType:
        if self._is_node(self.left):
            return self.left
        raise SentinelReferenceError(f"Node {self.left} is not a BinarySearchTreeNode")

    def successor(self, key: Key) -> BinaryNodeType:
        if key > self.key:
            return self.nonnull_right.successor(key)
        elif key < self.key:
            try:
                return self.nonnull_left.successor(key)
            except ValueError:
                return cast(BinaryNodeType, self)
        else:
            return self.nonnull_right.minimum()

    def predecessor(self, key: Key) -> BinaryNodeType:
        if key < self.key:
            return self.nonnull_left.predecessor(key)
        elif key > self.key:
            try:
                return self.nonnull_right.predecessor(key)
            except ValueError:
                return cast(BinaryNodeType, self)
        else:
            return self.nonnull_left.maximum()

    def insert_node(
        self, node: BinaryNodeType, allow_overwrite: bool = False
    ) -> BinaryNodeType:
        if self.key == node.key:
            if allow_overwrite:
                self.value = node.value
            else:
                raise ValueError(f"Overwrites of {node.key} not allowed")
        elif node.key < self.key:
            if self._is_node(self.left):
                self.left = self.left.insert_node(node, allow_overwrite)
            else:
                self.left = node
        else:
            if self._is_node(self.right):
                self.right = self.right.insert_node(node, allow_overwrite)
            else:
                self.right = node
        return cast(BinaryNodeType, self)

    def delete_key(self, key: Key) -> Union[BinaryNodeType, SentinelType]:
        return self._delete_key(key, None)

    def _delete_key(
        self, key: Key, parent: Optional[BinaryNodeType]
    ) -> Union[BinaryNodeType, SentinelType]:
        if key < self.key:
            self.left = self.nonnull_left._delete_key(key, self)
        elif key > self.key:
            self.right = self.nonnull_right._delete_key(key, self)
        else:
            if not self._is_node(self.left):
                return self.right
            elif not self._is_node(self.right):
                return self.left
            else:
                parent_successor = self
                successor = self.nonnull_right
                while successor._is_node(successor.left):
                    parent_successor = successor
                    successor = successor.nonnull_left
                assert not successor._is_node(successor.left)
                if successor is not self.right:
                    parent_successor.left = successor.right
                    successor.right = self.right
                successor.left = self.left
                if parent is not None:
                    if parent.left == self:
                        parent.left = successor
                    else:
                        parent.right = successor
                return successor
        return cast(BinaryNodeType, self)

    def delete_min(self):
        if not self._is_node(self.left):
            return self.right
        else:
            self.left = self.nonnull_left.delete_min()
            return cast(BinaryNodeType, self)

    def delete_max(self):
        if not self._is_node(self.right):
            return self.left
        else:
            self.right = self.nonnull_right.delete_max()
            return cast(BinaryNodeType, self)

    def extract_min(
        self,
    ) -> tuple[tuple[Key, Value], Union[BinaryNodeType, SentinelType]]:
        try:
            keyval, self.left = self.nonnull_left.extract_min()
            return keyval, cast(BinaryNodeType, self)
        except SentinelReferenceError:
            keyval = (self.key, self.value)
            if self._is_node(self.right):
                return keyval, self.right
            else:
                return keyval, self.left

    def extract_max(
        self,
    ) -> tuple[tuple[Key, Value], Union[BinaryNodeType, SentinelType]]:
        try:
            keyval, self.right = self.nonnull_right.extract_max()
            return keyval, cast(BinaryNodeType, self)
        except SentinelReferenceError:
            keyval = (self.key, self.value)
            if self._is_node(self.left):
                return keyval, self.left
            else:
                return keyval, self.right

    def inorder(self) -> Iterator[BinaryNodeType]:
        if self._is_node(self.left):
            yield from self.left.inorder()
        yield cast(BinaryNodeType, self)
        if self._is_node(self.right):
            yield from self.right.inorder()

    def preorder(self) -> Iterator[BinaryNodeType]:
        yield cast(BinaryNodeType, self)
        if self._is_node(self.left):
            yield from self.left.inorder()
        if self._is_node(self.right):
            yield from self.right.inorder()

    def postorder(self) -> Iterator[BinaryNodeType]:
        if self._is_node(self.left):
            yield from self.left.inorder()
        if self._is_node(self.right):
            yield from self.right.inorder()
        yield cast(BinaryNodeType, self)

    def yield_edges(self) -> Iterator[tuple[BinaryNodeType, BinaryNodeType]]:
        if self._is_node(self.left):
            yield from self.left.yield_edges()
            yield cast(BinaryNodeType, self), self.left
        if self._is_node(self.right):
            yield from self.right.yield_edges()
            yield cast(BinaryNodeType, self), self.right

    def level_order(self) -> Iterator[BinaryNodeType]:
        raise NotImplementedError

    def __setitem__(self, key: Key, value: Value) -> None:
        raise NotImplementedError("Use insert_node instead")

    def __delitem__(self, key: Key) -> None:
        self.delete_key(key)

    def __getitem__(self, key: Key) -> Value:
        raise NotImplementedError("Use access instead")

    def __iter__(self) -> Iterator[Key]:
        for node in self.inorder():
            yield node.key

    def children(self) -> Iterator[Union[BinaryNodeType, SentinelType]]:
        yield self.left
        yield self.right


class BSTNode(
    AbstractBSTNode[
        Key,
        Value,
        "BSTNode[Key, Value]",
        Sentinel,
    ]
):
    pass


@dataclass
class AbstractBSTNodeWithParent(
    Generic[Key, Value, BinaryNodeWithParentType, SentinelType],
    AbstractBSTNode[
        Key,
        Value,
        BinaryNodeWithParentType,
        SentinelType,
    ],
    SupportsParent[BinaryNodeWithParentType, SentinelType],
    ABC,
):
    pass


class BSTNodeWithParent(
    AbstractBSTNodeWithParent[
        Key,
        Value,
        "BSTNodeWithParent[Key, Value]",
        Sentinel,
    ]
):
    def __hash__(self):
        return hash(self.key)


DIR = "./trees/"
DOT_FILENAME = "tree.dot"
GRAPH_TYPE = "pdf"


def graph_prologue() -> str:
    return (
        'digraph G {  graph [fontname = "Courier New", engine="sfdp"];\n'
        + ' node [fontname = "Courier", style = rounded];\n'
        + ' edge [fontname = "Courier"];'
    )


def graph_epilogue() -> str:
    return "}"


def escape(s: str) -> str:
    return (
        s.replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\b", "\\b")
        .replace("\r", "\\r")
        .replace("\f", "\\f")
        .replace("'", "\\'")
        .replace('"', '\\"')
        .replace("<", "\\<")
        .replace(">", "\\>")
        # .replace("\n", "\\l")
        .replace("||", "\\|\\|")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def create_graph_pdf(
    graph,
    output_filename,
    dot_filename=DOT_FILENAME,
    output_filetype=GRAPH_TYPE,
):
    dot_exec_filepath = (
        "/usr/local/bin/dot" if sys.platform == "darwin" else "/usr/bin/dot"
    )

    output_filepath = DIR + output_filename
    dot_filepath = DIR + dot_filename

    os.makedirs(DIR, exist_ok=True)
    with open(dot_filepath, "w+") as f:
        f.write("\n".join(graph))

    args = [
        dot_exec_filepath,
        f"-T{output_filetype}",
        f"-Gdpi={96}",
        dot_filepath,
        "-o",
        output_filepath,
    ]
    subprocess.run(args)
    os.makedirs(output_filetype, exist_ok=True)
    subprocess.run(["open", output_filepath])
    # subprocess.run(["rm", output_filepath])


def draw_tree(
    root: Union[AbstractSentinel, AbstractBSTNode],
    output_filename: str = "tree.pdf",
):
    graph = [graph_prologue()]
    edges = []
    nodes = []

    if isinstance(root, AbstractSentinel):
        nodes.append(
            f"   {id(root)} [shape=record, style=filled, fillcolor=black, "
            f'fontcolor=white, label="{escape(str(root))}"];'
        )
    else:
        for node in root.inorder():
            if node.value:
                s = f"key={node.key}, value={node.value}"
            else:
                s = f"key={node.key}"
            nodes.append(
                f"   {id(node)} [shape=record, style=filled, fillcolor=black, "
                f'fontcolor=white, label="{escape(s)}"];'
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
    _ = BSTNode(0, None, Sentinel(), Sentinel())
