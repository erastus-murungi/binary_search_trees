from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Union

from bst import BinarySearchTreeIterative, Comparable, Internal, Leaf, Value

leaf_parents = {}


def get_parent(node: Node) -> Node:
    """Returns the parent of a node"""
    return (
        node.aux.parent if isinstance(node, Internal) else leaf_parents.get(node, node)
    )


def set_parent(node: InternalNode, parent: Node):
    """Sets the parent of a node"""
    if isinstance(node, Internal):
        node.aux.parent = parent
    else:
        leaf_parents[node] = parent


@dataclass(slots=True)
class SplayTreeAux:
    parent: Node = field(default_factory=Leaf, repr=False)


InternalNode = Internal[Comparable, Value, SplayTreeAux]
Node = Union[InternalNode, Leaf]


class SplayTree(BinarySearchTreeIterative[Comparable, Value, SplayTreeAux]):
    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        raise NotImplementedError()

    def insert_impl(
        self, key: Comparable, value: Value, aux: Optional[SplayTreeAux] = None
    ):
        parent, child = super().insert_parent(key, value, SplayTreeAux())
        set_parent(child, parent)
        self.splay(child)

    def access(self, key):
        if isinstance(self.root, Leaf):
            return self.root

        x = self.root
        y = get_parent(x)  # last non-null node
        while isinstance(x, Internal) and x.key != key:
            y = x
            x = x.choose(key)

        if isinstance(x, Leaf) or x.key != key:
            if isinstance(y, Internal):
                self.splay(y)
            return Leaf()
        else:
            self.splay(x)
            return x

    def splay(self, x):
        while isinstance(get_parent(x), Internal):
            x_p = get_parent(x)
            assert isinstance(x_p, Internal)
            x_pp = get_parent(x_p)
            if isinstance(x_pp, Leaf):
                if x_p.left is x:
                    self.right_rotate_with_parent(x_p)
                else:
                    self.left_rotate_with_parent(x_p)
            elif x_p.left is x and x_pp.left is x_p:
                self.right_rotate_with_parent(x_pp)
                self.right_rotate_with_parent(x_p)
            elif x_p.right is x and x_pp.right is x_p:
                self.left_rotate_with_parent(x_pp)
                self.left_rotate_with_parent(x_p)
            elif x_p.left is x and x_pp.right is x_p:
                self.right_rotate_with_parent(x_p)
                self.left_rotate_with_parent(x_pp)
            else:
                self.left_rotate_with_parent(x_p)
                self.right_rotate_with_parent(x_pp)

    def left_rotate_with_parent(self, node: InternalNode):
        parent = get_parent(node)
        right_child = node.right

        node.right = right_child.left
        if isinstance(right_child.left, Internal):
            set_parent(right_child.left, node)

        set_parent(right_child, parent)
        if isinstance(parent, Leaf):
            self.root = right_child
        elif node is parent.left:
            parent.left = right_child
        else:
            parent.right = right_child

        right_child.left = node
        set_parent(node, right_child)

    def right_rotate_with_parent(self, node: InternalNode):
        parent = get_parent(node)
        left_child = node.left

        node.left = left_child.right
        if isinstance(left_child.right, Internal):
            set_parent(left_child.right, node)

        set_parent(left_child, parent)
        if isinstance(parent, Leaf):
            self.root = left_child
        elif node is parent.left:
            parent.left = left_child
        else:
            parent.right = left_child

        left_child.right = node
        set_parent(node, left_child)

    def join(self, other):
        """Combines trees t1 and t2 into a single tree containing all items from
        both trees and return the resulting tree. This operation assumes that
        all items in t1 are less than all those in t2 and destroys both t1 and t2."""

        assert isinstance(self.root, Internal)
        assert isinstance(other.root, Internal)

        x = self.maximum()
        self.splay(x)
        self.root.right = other.root
        other.root.parent = self.root.right
        del other

        return self

    def split(self, i):
        """Construct and return two trees t1 and t2, where t1 contains all items
        in t less than or equal to i, and t2 contains all items in t greater than
        i. This operation destroys t."""

        assert isinstance(self.root, Internal)

        self.access(i)
        y = self.root.right
        assert isinstance(y, Internal)
        self.transplant(y, Leaf())
        set_parent(y, Leaf())
        return self, SplayTree(self.root.right, len(self.root.right))

    def transplant(self, u: Node, v: Node):
        # u is the initial node and v is the node to transplant u
        if isinstance(get_parent(u), Internal):
            #                g        g
            #                |        |
            #                u   =>   v
            #               / \      / \
            #             u.a  u.ß  v.a v.ß
            if u == get_parent(u).left:
                get_parent(u).left = v
            else:
                get_parent(u).right = v
        else:
            self.root = v

        set_parent(v, get_parent(u))

    def delete(self, target_key):
        """Deletes a key from the Splay Tree"""

        z = self.access(target_key)
        if isinstance(z, Internal):
            self.splay(z)
            left, right = self.root.left, self.root.right
            self.transplant(z, Leaf())

            m = Leaf()
            if isinstance(left, Internal):
                set_parent(left, Leaf())
                m = left.maximum()
                self.splay(m)
                self.root = m
            if isinstance(right, Internal):
                if isinstance(left, Internal):
                    m.right = right
                else:
                    self.root = right
                set_parent(right, m)
            self.size -= 1
        else:
            raise KeyError(f"Key = {target_key} not found {values}")


if __name__ == "__main__":
    from random import randint

    for _ in range(100):
        st = SplayTree()
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
        num_nodes = 1000
        values = list({randint(0, 100) for _ in range(num_nodes)})
        # values = [72, 49, 94]
        # values = [80, 93]
        # values = [1]

        for val in values:
            st.insert(val, None)
            assert val in st
            st.check_invariants(-10000, 100000)

        for val in values:
            st.delete(val)
            assert val not in st, values
            st.check_invariants(-10000, 100000)

        assert st.size == 0
        assert isinstance(st.root, Leaf)
