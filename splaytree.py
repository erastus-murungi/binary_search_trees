from __future__ import annotations

from typing import Any, Optional, TypeGuard, Union

from bst import AbstractBinarySearchTreeWithParentIterative, Comparable
from core import Value
from nodes import AbstractBinarySearchTreeInternalNodeWithParent, Sentinel


class SplayTreeInternalNode(
    AbstractBinarySearchTreeInternalNodeWithParent[
        Comparable,
        Value,
        "SplayTreeInternalNode[Comparable, Value]",
        Sentinel[Comparable],
    ]
):
    def is_node(self, node: Any) -> TypeGuard[SplayTreeInternalNode[Comparable, Value]]:
        return isinstance(node, SplayTreeInternalNode)

    def is_sentinel(self, node: Any) -> TypeGuard[Sentinel[Comparable]]:
        return isinstance(node, Sentinel)

    def sentinel(self) -> Sentinel[Comparable]:
        return Sentinel.default()

    def node(
        self,
        key: Comparable,
        value: Value,
        left: Union[
            SplayTreeInternalNode[Comparable, Value], Sentinel[Comparable]
        ] = Sentinel.default(),
        right: Union[
            SplayTreeInternalNode[Comparable, Value], Sentinel[Comparable]
        ] = Sentinel.default(),
        parent: Union[
            SplayTreeInternalNode[Comparable, Value], Sentinel[Comparable]
        ] = Sentinel.default(),
        *args,
        **kwargs,
    ) -> SplayTreeInternalNode[Comparable, Value]:
        return SplayTreeInternalNode(
            key=key, value=value, left=left, right=right, parent=parent
        )


class SplayTree(
    AbstractBinarySearchTreeWithParentIterative[
        Comparable, Value, SplayTreeInternalNode, Sentinel[Comparable]
    ]
):
    def insert(
        self, key: Comparable, value: Value, allow_overwrite: bool = True
    ) -> SplayTreeInternalNode[Comparable, Value]:
        node = super().insert(key, value, allow_overwrite)
        self.splay(node)
        return node

    # @property
    # def nonnull_root(self) -> SplayTreeInternalNode[Comparable, Value]:
    #     if self.is_sentinel(self.root):
    #         raise SentinelReferenceError("Root is sentinel")
    #     return self.root

    def access(self, key: Any) -> SplayTreeInternalNode[Comparable, Value]:
        if self.is_sentinel(self.root):
            raise KeyError(f"Key {key} not found; tree is empty")

        x = self.root
        y = self.nonnull_root.parent  # last non-null node
        while self.is_node(x) and x.key != key:
            y = x
            x = x.choose(key)

        if self.is_sentinel(x) or (self.is_node(x) and x.key != key):
            if self.is_node(y):
                self.splay(y)
            raise KeyError(f"Key {key} not found")
        else:
            assert self.is_node(x)
            self.splay(x)
            return x

    def splay(self, x: SplayTreeInternalNode[Comparable, Value]):
        while self.is_node(x.parent):
            x_p = x.parent
            x_pp = x_p.parent
            if self.is_sentinel(x_pp):
                if x_p.left is x:
                    self.right_rotate(x_p)
                else:
                    self.left_rotate(x_p)
            else:
                assert self.is_node(x_pp)
                if x_p.left is x and x_pp.left is x_p:
                    self.right_rotate(x_pp)
                    self.right_rotate(x_p)
                elif x_p.right is x and x_pp.right is x_p:
                    self.left_rotate(x_pp)
                    self.left_rotate(x_p)
                elif x_p.left is x and x_pp.right is x_p:
                    self.right_rotate(x_p)
                    self.left_rotate(x_pp)
                else:
                    self.left_rotate(x_p)
                    self.right_rotate(x_pp)

    def join(self, other):
        """Combines trees t1 and t2 into a single tree containing all items from
        both trees and return the resulting tree. This operation assumes that
        all items in t1 are less than all those in t2 and destroys both t1 and t2."""

        assert self.is_node(self.root)
        assert self.is_node(other.root)

        x = self.maximum()
        self.splay(x)
        self.root.right = other.root
        other.root.parent = self.root.right
        del other

        return self

    def split(
        self, i: Comparable
    ) -> tuple[SplayTree[Comparable, Value], SplayTree[Comparable, Value]]:
        """Construct and return two trees t1 and t2, where t1 contains all items
        in t less than or equal to i, and t2 contains all items in t greater than
        i. This operation destroys t."""

        assert self.is_node(self.root)

        self.access(i)
        y = self.root.right
        assert self.is_node(y)
        self.transplant(y, self.sentinel())
        y.parent = self.sentinel()
        return self, SplayTree[Comparable, Value](self.nonnull_root.nonnull_right)

    def delete(self, target_key):
        """Deletes a key from the Splay Tree"""

        z = self.access(target_key)
        if self.is_node(z):
            self.splay(z)
            left, right = self.root.left, self.root.right
            self.transplant(z, self.sentinel())

            m = self.sentinel()
            if self.is_node(left):
                left.parent = self.sentinel()
                left.parent = self.sentinel()
                m = left.maximum()
                self.splay(m)
                self.root = m
            if self.is_node(right):
                if self.is_node(left):
                    m.right = right
                else:
                    self.root = right
                right.parent = m
            self.size -= 1
        else:
            raise KeyError(f"Key = {target_key} not found {values}")

    def is_node(self, node: Any) -> TypeGuard[SplayTreeInternalNode[Comparable, Value]]:
        return isinstance(node, SplayTreeInternalNode)

    def is_sentinel(self, node: Any) -> TypeGuard[Sentinel[Comparable]]:
        return isinstance(node, Sentinel)

    def sentinel(self) -> Sentinel[Comparable]:
        return Sentinel.default()

    def node(
        self,
        key: Comparable,
        value: Value,
        left: Union[
            SplayTreeInternalNode[Comparable, Value], Sentinel[Comparable]
        ] = Sentinel.default(),
        right: Union[
            SplayTreeInternalNode[Comparable, Value], Sentinel[Comparable]
        ] = Sentinel.default(),
        parent: Union[
            SplayTreeInternalNode[Comparable, Value], Sentinel[Comparable]
        ] = Sentinel.default(),
        *args,
        **kwargs,
    ) -> SplayTreeInternalNode[Comparable, Value]:
        return SplayTreeInternalNode(
            key=key, value=value, left=left, right=right, parent=parent
        )


if __name__ == "__main__":
    from random import randint

    for _ in range(100):
        st = SplayTree[int, None]()
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
            st.validate(-10000, 100000)

        for val in values:
            st.delete(val)
            assert val not in st, values
            st.validate(-10000, 100000)

        assert not st
