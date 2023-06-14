from __future__ import annotations

from dataclasses import field, dataclass
from typing import Union, Optional, Iterator

from typeguard import typechecked
from enum import IntEnum

from bst import (
    BinarySearchTreeIterative,
    Comparable,
    Internal,
    Leaf,
    Value,
    KeyValue,
)


class Color(IntEnum):
    RED = 0
    BLACK = 1


@dataclass(slots=True)
class RBTAuxiliaryData:
    """Auxiliary data for a node in a red-black tree"""

    color: Color = Color.BLACK
    parent: InternalNode = field(default_factory=Leaf, repr=False, compare=False)


InternalNode = Internal[Comparable, Value, RBTAuxiliaryData]
Node = Union[InternalNode, Leaf]


def get_color(node: Node) -> Color:
    """Returns the color of a node"""
    return node.aux.color if isinstance(node, Internal) else Color.BLACK


leaf_parents = {}


def get_parent(node: Node) -> Node:
    """Returns the parent of a node"""
    return (
        node.aux.parent if isinstance(node, Internal) else leaf_parents.get(node, node)
    )


def set_parent(node: InternalNode, parent: InternalNode):
    """Sets the parent of a node"""
    if isinstance(node, Internal):
        node.aux.parent = parent
    else:
        leaf_parents[node] = parent


@typechecked
def set_color(node: Node, color: Color):
    """Sets the color of a node"""
    if isinstance(node, Internal):
        node.aux.color = color
    else:
        assert color == Color.BLACK


def black_height(node: Node) -> int:
    if isinstance(node, Internal):
        if get_color(node) == Color.BLACK:
            return 1 + black_height(node.right)
        else:
            return black_height(node.right)
    else:
        return 0


class RedBlackTree(BinarySearchTreeIterative[Comparable, Value, RBTAuxiliaryData]):
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

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        raise NotImplementedError()

    def parent(self, node: InternalNode) -> Node:
        return node.aux.parent

    def insert(
        self, key: Comparable, value: Value, aux: Optional[RBTAuxiliaryData] = None
    ) -> InternalNode:
        node = self.access(key)
        if isinstance(node, Internal):
            node.value = value
        else:
            ancestry = self.insert_ancestry(key, value, RBTAuxiliaryData())
            assert ancestry and len(ancestry) > 0
            node = ancestry.pop()
            if ancestry:
                set_parent(node, ancestry[-1])
            set_color(node, Color.RED)
            self.rbt_insert_fixup(node)
        return node

    def check_invariants(self, lower_limit: Comparable, upper_limit: Comparable):
        super().check_invariants(lower_limit, upper_limit)
        self.check_black_height_invariant()
        self.check_property_4()

    def rbt_insert_fixup(self, z):
        while get_color(get_parent(z)) == Color.RED:
            # let a = x's parent
            z_p = get_parent(z)
            # if z's parent is a left child, the uncle will be z's grandparent right child
            z_pp = get_parent(z_p)
            if isinstance(z_pp, Leaf):
                break
            if z_p == z_pp.left:
                y = z_pp.right
                # if z's uncle is Color.RED (and z's parent is Color.RED), then we are in case 1
                if get_color(y) == Color.RED:
                    set_color(z_p, Color.BLACK)
                    set_color(y, Color.BLACK)
                    set_color(z_pp, Color.RED)
                    z = get_parent(get_parent(z))
                # z's uncle is Color.BLACK (z's parent is Color.RED), we check for case 2 first
                else:
                    # if z is a right child (and remember z's parent is a left child), we left-rotate z.p
                    if z_p.right == z:
                        z = z_p
                        self.left_rotate_with_parent(z)
                    set_color(get_parent(z), Color.BLACK)
                    set_color(get_parent(get_parent(z)), Color.RED)
                    self.right_rotate_with_parent(get_parent(get_parent(z)))

            else:
                # z's parent is a right child, z's uncle is the left child of z's grandparent
                y = z_pp.left
                # check for case 1
                if get_color(y) == Color.RED:
                    set_color(z_p, Color.BLACK)
                    set_color(y, Color.BLACK)
                    set_color(z_pp, Color.RED)
                    z = get_parent(get_parent(z))
                else:
                    # z's parent is already a right child, so check if z is a left child and rotate
                    if z_p.left == z:
                        z = z_p
                        self.right_rotate_with_parent(z)
                    set_color(get_parent(z), Color.BLACK)
                    set_color(get_parent(get_parent(z)), Color.RED)
                    self.left_rotate_with_parent(get_parent(get_parent(z)))

        # make the root black
        set_color(self.root, Color.BLACK)

    def delete(self, target_key: Comparable):
        z = self.access(target_key)
        if isinstance(z, Internal):
            # y identifies the node to be deleted
            # We maintain node y as the node either removed from the tree or moved within the tree
            y = z
            y_original_color = get_color(y)
            if isinstance(z.left, Leaf):
                # if z.right is also null, then z => nil, the null node will also have a parent pointer
                x = z.right
                # replace the node with its right child
                self.transplant(z, z.right)
            elif isinstance(z.right, Leaf):
                x = z.left
                self.transplant(z, z.left)
            else:
                y = z.right.minimum()  # find z's successor
                y_original_color = get_color(y)
                x = y.right
                if y != z.right:
                    self.transplant(y, y.right)
                    y.right = z.right
                    set_parent(y.right, y)
                else:
                    set_parent(x, y)
                self.transplant(z, y)
                y.left = z.left
                set_parent(y.left, y)
                set_color(y, get_color(z))
            if y_original_color == Color.BLACK:
                # the x might be a null pointer
                self.delete_fixup(x)
            self.size -= 1
        else:
            raise ValueError(f"key={target_key} not in tree")

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

    def delete_fixup(self, x: Node):
        while x != self.root and get_color(x) == Color.BLACK:
            x_p = get_parent(x)
            if isinstance(x_p, Leaf):
                break
            if x == x_p.left:
                w = x_p.right  # w is the sibling of x
                # case 1: w is red
                # Since w must have black children, we can switch the
                # colors of w and x: p and then perform a left-rotation on x: p without violating any
                # of the red-black properties. The new sibling of x, which is one of w’s children
                # prior to the rotation, is now black
                if get_color(w) == Color.RED:
                    set_color(w, Color.BLACK)
                    set_color(x_p, Color.RED)
                    self.left_rotate_with_parent(get_parent(x))
                    w = get_parent(x).right

                # case 2: x’s sibling w is black, and both of w’s children are black
                # the color of w is black, so we take of one black from both and add it to x.parent
                if (
                    get_color(w.left) == Color.BLACK
                    and get_color(w.right) == Color.BLACK
                ):
                    set_color(w, Color.RED)
                    x = get_parent(x)
                    # if x was black, then the loop terminates, and x is colored black
                else:
                    # case 3: x’s sibling w is black, w’s left child is red, and w’s right child is black
                    # We can switch the colors of w and its left
                    # child w: left and then perform a right rotation on w without violating any of the
                    # red-black properties.

                    if get_color(w.right) == Color.BLACK:
                        set_color(w.left, Color.BLACK)
                        set_color(w, Color.RED)
                        self.right_rotate_with_parent(w)
                        w = get_parent(x).right

                    # case 4: x’s sibling w is black, and w’s right child is red
                    # By making some color changes and performing a left rotation
                    # on x: p, we can remove the extra black on x, making it singly black, without
                    # violating any of the red-black properties. Setting x to be the root causes the while
                    # loop to terminate when it tests the loop condition
                    set_color(w, get_color(x_p))
                    set_color(get_parent(x), Color.BLACK)
                    set_color(w.right, Color.BLACK)
                    self.left_rotate_with_parent(get_parent(x))
                    x = self.root
            else:
                # symmetric to the cases 1 to case 4
                w = x_p.left
                if get_color(w) == Color.RED:
                    set_color(w, Color.BLACK)
                    set_color(x_p, Color.RED)
                    self.right_rotate_with_parent(x_p)
                    assert get_parent(x) == x_p
                    w = x_p.left
                if (
                    get_color(w.left) == Color.BLACK
                    and get_color(w.right) == Color.BLACK
                ):
                    set_color(w, Color.RED)
                    x = get_parent(x)
                else:
                    if get_color(w.left) == Color.BLACK:
                        set_color(w.right, Color.BLACK)
                        set_color(w, Color.RED)
                        self.left_rotate_with_parent(w)
                        w = get_parent(x).left
                    assert get_parent(x) == x_p
                    set_color(w, get_color(x_p))
                    set_color(x_p, Color.BLACK)
                    set_color(w.left, Color.BLACK)
                    self.right_rotate_with_parent(x_p)
                    x = self.root
        set_color(x, Color.BLACK)

    def check_property_4(self):
        for node in self.inorder():
            if get_color(node) == Color.RED:
                if (
                    get_color(node.left) == Color.RED
                    or get_color(node.right) == Color.RED
                ):
                    raise RuntimeError(f"Property 4 violated at {node} {values}")

    def check_black_height_invariant(self):
        """Checks whether the black-height is the same for all paths starting at the root"""

        def recurse(node: Node):
            """Checks whether the black-height is the same for all paths starting at the root"""
            if isinstance(node, Leaf):
                return 0
            else:
                assert isinstance(node, Internal)
                if (bh_right := recurse(node.right)) != recurse(node.left):
                    raise RuntimeError(f"Black height invariant violated at {node}")
                if get_color(node) == Color.BLACK:
                    return 1 + bh_right
                else:
                    return bh_right

        recurse(self.root)

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

    def extract_min(self) -> tuple[KeyValue, Node]:
        min_node = self.minimum()
        keyval = (min_node.key, min_node.value)
        self.delete(min_node.key)
        return keyval, self.root

    def extract_max(self) -> tuple[KeyValue, Node]:
        max_node = self.maximum()
        keyval = (max_node.key, max_node.value)
        self.delete(max_node.key)
        return keyval, self.root


if __name__ == "__main__":

    # #
    # # # values = [3, 52, 31, 55, 93, 60, 81, 93, 46, 37, 47, 67, 34, 95, 10, 23, 90, 14, 13, 88, 88]
    # #
    from random import randint

    num_nodes = 100
    for _ in range(1000):
        values = list({randint(0, 100) for _ in range(num_nodes)})
        rb = RedBlackTree()
        for k in values:
            rb.insert(k, None)
            rb.check_invariants(-1, 100000)
            assert k in rb

        for k in values:
            rb.delete(k)
            rb.check_invariants(-1, 100000)
            assert k not in rb, values

        assert not rb
        assert isinstance(rb.root, Leaf)
