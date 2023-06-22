from __future__ import annotations

from typing import Any, TypeGuard, Union

from bst import AbstractBSTWithParentIterative, Key
from core import Value
from nodes import BSTNodeWithParent, Sentinel


class SplayTree(
    AbstractBSTWithParentIterative[
        Key,
        Value,
        BSTNodeWithParent[Key, Value],
        Sentinel,
    ]
):
    def insert(
        self, key: Key, value: Value, allow_overwrite: bool = True
    ) -> BSTNodeWithParent[Key, Value]:
        node = super().insert(key, value, allow_overwrite)
        self.splay(node)
        return node

    def access(self, key: Any) -> BSTNodeWithParent[Key, Value]:
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

    def splay(self, x: BSTNodeWithParent[Key, Value]):
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

    def split(self, i: Key) -> tuple[SplayTree[Key, Value], SplayTree[Key, Value]]:
        """Construct and return two trees t1 and t2, where t1 contains all items
        in t less than or equal to i, and t2 contains all items in t greater than
        i. This operation destroys t."""

        assert self.is_node(self.root)

        self.access(i)
        y = self.root.right
        assert self.is_node(y)
        self.transplant(y, self.sentinel())
        y.parent = self.sentinel()
        return self, SplayTree[Key, Value](self.nonnull_root.nonnull_right)

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
            return z
        else:
            raise KeyError(f"Key = {target_key} not found")

    @staticmethod
    def is_node(node: Any) -> TypeGuard[BSTNodeWithParent[Key, Value]]:
        return isinstance(node, BSTNodeWithParent)

    @staticmethod
    def node(
        key: Key,
        value: Value,
        left: Union[
            BSTNodeWithParent[Key, Value],
            Sentinel,
        ] = Sentinel(),
        right: Union[
            BSTNodeWithParent[Key, Value],
            Sentinel,
        ] = Sentinel(),
        parent: Union[
            BSTNodeWithParent[Key, Value],
            Sentinel,
        ] = Sentinel(),
        *args,
        **kwargs,
    ) -> BSTNodeWithParent[Key, Value]:
        return BSTNodeWithParent(
            key=key, value=value, left=left, right=right, parent=parent
        )

    @classmethod
    def sentinel_class(cls):
        return Sentinel
