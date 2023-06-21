from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Generic, Optional, TypeGuard, Union

from bst import AbstractBSTWithParentIterative, Key, Value
from nodes import AbstractBSTNodeWithParent, AbstractSentinelWithParent, SupportsParent


class Color(IntEnum):
    RED = 0
    BLACK = 1


@dataclass(slots=True)
class SupportsColor(ABC):
    """Auxiliary data for a node in a red-black tree"""

    color: Color

    @abstractmethod
    def black_height(self) -> int:
        pass


@dataclass(slots=True)
class RBTSentinel(
    Generic[Key, Value],
    AbstractSentinelWithParent[
        "RBTNode[Key, Value]",
        "RBTSentinel[Key, Value]",
    ],
    SupportsColor,
    SupportsParent["RBTNode", "RBTSentinel"],
):
    """Sentinel node in a red-black tree"""

    parent: Union[RBTSentinel[Key, Value], RBTNode[Key, Value]]
    color: Color = Color.BLACK

    def __init__(
        self,
        parent: Optional[RBTNode[Key, Value]] = None,
        color: Color = Color.BLACK,
    ):
        if parent is None:
            self.parent = self
        else:
            self.parent = parent
        self.color = color

    def black_height(self) -> int:
        return 0

    @classmethod
    def default(cls) -> RBTSentinel[Key, Value]:
        return RBTSentinel()


@dataclass(slots=True)
class RBTNode(
    AbstractBSTNodeWithParent[
        Key,
        Value,
        "RBTNode[Key, Value]",
        RBTSentinel[Key, Value],
    ],
    SupportsColor,
):
    """Internal node in a red-black tree"""

    def black_height(self) -> int:
        if self.color == Color.BLACK:
            return 1 + self.right.black_height()
        else:
            return self.right.black_height()


class RedBlackTree(
    AbstractBSTWithParentIterative[
        Key,
        Value,
        RBTNode[Key, Value],
        RBTSentinel[Key, Value],
    ]
):
    """
    A Red-Black tree is a height-balanced binary search tree which supports queries and updates in O(log n) time.
    Red-Black Trees provide faster insertion and removal operations than AVL trees because they need fewer rotations.
    However, AVL trees provide faster lookups than Red-Black trees because they are more strictly balanced.
    A binary search tree is a Red-Black tree if:
        - Every node is either red or black.
        - Every leaf (nil) is black.
        - If a node is red, then both its children are black.
        - Every simple path from a node to a descendant leaf contains the same number of black nodes.
        The black-height of a node x, bh(x),
        is the number of black nodes on any path from x to a leaf, not counting x.
        A Red-Black tree with n internal nodes has height at most 2lg(n+1).
        See CLRS pg. 309 for the proof of this lemma.
    """

    def insert(
        self, key: Key, value: Value, allow_overwrite: bool = True
    ) -> RBTNode[Key, Value]:
        child = super().insert(key, value, allow_overwrite)
        child.color = Color.RED
        self.rbt_insert_fixup(child)
        return child

    def validate(self, lower_limit: Key, upper_limit: Key):
        super().validate(lower_limit, upper_limit)
        self.check_black_height_invariant()
        self.check_property_4()

    def rbt_insert_fixup(self, z):
        while z.parent.color == Color.RED:
            # let a = x's parent
            z_p = z.parent
            # if z's parent is a left child, the uncle will be z's grandparent right child
            z_pp = z_p.parent
            if self.is_sentinel(z_pp):
                break
            if z_p == z_pp.left:
                y = z_pp.right
                # if z's uncle is Color.RED (and z's parent is Color.RED), then we are in case 1
                if y.color == Color.RED:
                    z_p.color = Color.BLACK
                    y.color = Color.BLACK
                    z_pp.color = Color.RED
                    z = z.parent.parent
                # z's uncle is Color.BLACK (z's parent is Color.RED), we check for case 2 first
                else:
                    # if z is a right child (and remember z's parent is a left child), we left-rotate z.p
                    if z_p.right == z:
                        z = z_p
                        self.left_rotate(z)
                    z.parent.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    self.right_rotate(z.parent.parent)

            else:
                # z's parent is a right child, z's uncle is the left child of z's grandparent
                y = z_pp.left
                # check for case 1
                if y.color == Color.RED:
                    z_p.color = Color.BLACK
                    y.color = Color.BLACK
                    z_pp.color = Color.RED
                    z = z.parent.parent
                else:
                    # z's parent is already a right child, so check if z is a left child and rotate
                    if z_p.left == z:
                        z = z_p
                        self.right_rotate(z)
                    z.parent.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    self.left_rotate(z.parent.parent)

        # make the root black
        self.root.color = Color.BLACK

    def transplant(
        self,
        u: RBTNode[Key, Value],
        v: Union[RBTNode[Key, Value], RBTSentinel[Key, Value]],
        _: Optional[Union[RBTNode[Key, Value], RBTSentinel[Key, Value]]] = None,
    ) -> None:
        if self.is_sentinel(u.parent):
            self.root = v
        else:
            assert self.is_node(u.parent)
            if u is u.parent.left:
                u.parent.left = v
            else:
                u.parent.right = v
        v.parent = u.parent

    def delete(self, target_key: Key):
        z = self.access(target_key)
        if self.is_node(z):
            # y identifies the node to be deleted
            # We maintain node y as the node either removed from the tree or moved within the tree
            y = z
            y_original_color = y.color
            if self.is_sentinel(z.left):
                # if z.right is also null, then z => nil, the null node will also have a parent pointer
                x = z.right
                self.transplant(z, z.right)
            elif self.is_sentinel(z.right):
                x = z.left
                self.transplant(z, z.left)
            else:
                y = z.nonnull_right.minimum()  # find z's successor
                y_original_color = y.color
                x = y.right
                if y is not z.right:
                    self.transplant(y, y.right)
                    y.right = z.right
                    y.right.parent = y
                else:
                    x.parent = y
                self.transplant(z, y)
                y.left = z.left
                y.left.parent = y
                y.color = z.color
            if y_original_color == Color.BLACK:
                # the x might be a null pointer
                self.rbt_delete_fixup(x)
            self.size -= 1
            return z
        else:
            raise ValueError(f"key={target_key} not in tree")

    def rbt_delete_fixup(
        self,
        x: Union[RBTNode[Key, Value], RBTSentinel[Key, Value]],
    ):
        while x != self.root and x.color == Color.BLACK:
            x_p = x.parent
            if self.is_sentinel(x_p):
                break
            assert self.is_node(x_p)
            if x == x_p.left:
                w = x_p.right  # w is the sibling of x

                assert self.is_node(w)
                # case 1: w is red
                # Since w must have black children, we can switch the
                # colors of w and x: p and then perform a left-rotation on x: p without violating any
                # of the red-black properties. The new sibling of x, which is one of w’s children
                # prior to the rotation, is now black

                if w.color == Color.RED:
                    w.color = Color.BLACK
                    x_p.color = Color.RED
                    self.left_rotate(x_p)
                    w = x_p.right

                assert self.is_node(w)
                # case 2: x’s sibling w is black, and both of w’s children are black
                # the color of w is black, so we take of one black from both and add it to x.parent
                if w.left.color == Color.BLACK and w.right.color == Color.BLACK:
                    w.color = Color.RED
                    x = x_p
                    # if x was black, then the loop terminates, and x is colored black
                else:
                    # case 3: x’s sibling w is black, w’s left child is red, and w’s right child is black
                    # We can switch the colors of w and its left
                    # child w: left and then perform a right rotation on w without violating any of the
                    # red-black properties.
                    if w.right.color == Color.BLACK:
                        w.left.color = Color.BLACK
                        w.color = Color.RED
                        self.right_rotate(w)
                        w = x_p.right

                    # case 4: x’s sibling w is black, and w’s right child is red
                    # By making some color changes and performing a left rotation
                    # on x: p, we can remove the extra black on x, making it singly black, without
                    # violating any of the red-black properties. Setting x to be the root causes the while
                    # loop to terminate when it tests the loop condition
                    assert self.is_node(w)
                    w.color = x_p.color
                    x_p.color = Color.BLACK
                    w.right.color = Color.BLACK
                    self.left_rotate(x_p)
                    x = self.nonnull_root
            else:
                # symmetric to the cases 1 to case 4
                w = x_p.left
                assert self.is_node(w)
                if w.color == Color.RED:
                    w.color = Color.BLACK
                    x_p.color = Color.RED
                    self.right_rotate(x_p)
                    w = x_p.left

                assert self.is_node(w)
                if w.right.color == Color.BLACK and w.left.color == Color.BLACK:
                    w.color = Color.RED
                    x = x_p
                else:
                    if w.left.color == Color.BLACK:
                        w.right.color = Color.BLACK
                        w.color = Color.RED
                        self.left_rotate(w)
                        w = x_p.left

                    assert self.is_node(w)
                    w.color = x_p.color
                    x_p.color = Color.BLACK
                    w.left.color = Color.BLACK
                    self.right_rotate(x_p)
                    x = self.nonnull_root
        x.color = Color.BLACK

    def check_property_4(self):
        for node in self.inorder():
            if node.color == Color.RED:
                if node.left.color == Color.RED or node.right.color == Color.RED:
                    raise RuntimeError(f"Property 4 violated at {node}")

    def check_black_height_invariant(self):
        """Checks whether the black-height is the same for all paths starting at the root"""

        def recurse(node: Union[RBTSentinel[Key, Value], RBTNode[Key, Value]]) -> int:
            """Checks whether the black-height is the same for all paths starting at the root"""
            if self.is_sentinel(node):
                assert node.color == Color.BLACK
                return 0
            else:
                assert self.is_node(node)
                if (bh_right := recurse(node.right)) != recurse(node.left):
                    raise RuntimeError(f"Black height invariant violated at {node}")
                if node.color == Color.BLACK:
                    return 1 + bh_right
                else:
                    return bh_right

        recurse(self.root)

    def is_node(self, item: Any) -> TypeGuard[RBTNode[Key, Value]]:
        return isinstance(item, RBTNode)

    def is_sentinel(self, item: Any) -> TypeGuard[RBTSentinel[Key, Value]]:
        return isinstance(item, RBTSentinel)

    def node(
        self,
        key: Key,
        value: Value,
        left: Union[
            RBTNode[Key, Value], RBTSentinel[Key, Value]
        ] = RBTSentinel.default(),
        right: Union[
            RBTNode[Key, Value], RBTSentinel[Key, Value]
        ] = RBTSentinel.default(),
        parent: Union[
            RBTNode[Key, Value], RBTSentinel[Key, Value]
        ] = RBTSentinel.default(),
        color=Color.RED,
        *args,
        **kwargs,
    ) -> RBTNode[Key, Value]:
        return RBTNode(
            key=key, value=value, left=left, right=right, parent=parent, color=color
        )

    def sentinel(self, *args, **kwards) -> RBTSentinel[Key, Value]:
        return RBTSentinel()

    def extract_min(self) -> tuple[Key, Value]:
        z = self.minimum()
        y = z
        y_original_color = y.color
        if self.is_sentinel(z.right):
            x = z.left
            self.transplant(z, z.left)
        else:
            assert self.is_node(z.right)
            y = z.right.minimum()
            y_original_color = y.color
            x = y.right
            self.transplant(y, y.right)
            y.right = z.right
            y.right.parent = y
            self.transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        if y_original_color == Color.BLACK:
            self.rbt_delete_fixup(x)
        self.size -= 1
        return z.key, z.value

    def extract_max(self) -> tuple[Key, Value]:
        z = self.maximum()
        y = z
        y_original_color = y.color
        if self.is_sentinel(z.left):
            x = z.right
            self.transplant(z, z.right)
        else:
            assert self.is_node(z.left)
            y = z.left.maximum()
            y_original_color = y.color
            x = y.left
            self.transplant(y, y.left)
            y.left = z.left
            y.left.parent = y
            self.transplant(z, y)
            y.right = z.right
            y.right.parent = y
            y.color = z.color
        if y_original_color == Color.BLACK:
            self.rbt_delete_fixup(x)
        self.size -= 1
        return z.key, z.value


if __name__ == "__main__":
    pass
