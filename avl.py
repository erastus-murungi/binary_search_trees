from dataclasses import dataclass
from typing import cast

from bst import Node, BST, InternalNode, Comparable, Value


@dataclass(slots=True)
class InternalAVLNode(InternalNode[Comparable, Value]):
    """As a reminder, the empty tree has height -1,
    the height of a nonempty tree whose root’s left child has height h₁ and
    root’s right child has height h₂ is 1 + max(h₁, h₂)"""

    left: "InternalAVLNode[Comparable, Value]" = None
    right: "InternalAVLNode[Comparable, Value]" = None
    h: int = 0

    def update_height(self):
        self.h = self.height()

    def choose_dir(self, right) -> "InternalAVLNode[Comparable, Value]":
        if right:
            return cast(InternalAVLNode, self.right)
        else:
            return cast(InternalAVLNode, self.left)

    def choose_set_dir(self, right, node):
        if right:
            self.right = node
        else:
            self.left = node

    def skew_direction(self) -> bool:
        hl = self.left.h if self.left else -1
        hr = self.right.h if self.right else -1
        return hl <= hr

    def is_skewed(self) -> bool:
        hl = self.left.h if self.left else -1
        hr = self.right.h if self.right else -1
        return abs(hr - hl) > 1

    def __update_heights(self):
        if self.left:
            self.left.update_height()
        if self.right:
            self.right.update_height()
        self.update_height()


class AVLTree(BST[Comparable, Value, InternalAVLNode]):
    """
    Little History:
        AVL trees, developed in 1962, were the first class of balanced binary search
        trees to be invented. They’re named after their inventors Адельсо́н-Ве́льский
        and Ла́ндис (Adelson-Velsky and Landis).
        A binary tree is an AVL tree if either:
            (1) the tree is empty or
            (2) the left and right children of the root are AVL trees and the heights of the left and right
                subtrees differ by at most one.
        AVL Trees are balanced because of the AVL condition that:
            For every node in the tree, the heights of its left subtree and right subtree
             differ by at most 1.
        Is this condition strong enough to guarantee O( log n) per operation?
        let N(h) denote the minimum number of nodes in an AVL Tree of height h, where h is the length of
        the longest root to leaf path.
        Clearly, N(0) = 1.
            N(h) = N(h-1) + N(h-2) + 1
        To make the recurrence well-defined, we add the case N(1) = 2
        This recurrence looks very similar to the Fibonacci recurrence.
        (by a little approximation and a little constructive induction). It can be proved that:
                N(h) ≈ (φ)^5, where φ is the golden ratio ≈ 1.618
                Solving for h results in h = O(log n)

        The class supports standard BST Queries and Updates:
        UPDATES:
            Insert(x, k)  : insert value x with key
            Delete(k)     : delete value with key k

        Queries:
            Search(k)     : return true if item with key k is in the dictionary
            Min()         : return the item with the minimum key
            Max()         : return the item with the maxium key
            Successor(n)  : return the node which is the successor of node n
            Predecessor(n): return the node which is the successor of node n

        The Class can also be used as a priority queue:
            Extract-min  : return a (key, value) pair where key is the min among all other keys in the Tree
            Extraxt-max  : return a (key, value) pair where key is the max among all other keys in the Tree

        Other methods:
            In-order-traversal: returns the element in sorted order
    """

    @property
    def node_class(self) -> type[InternalAVLNode]:
        return InternalAVLNode

    def insert(self, key: Comparable, value: Value) -> Node:
        # if the tree was empty, just insert the key at the root

        z = super().insert(key, value)
        parent = self.parent(z)

        if parent is not None:
            parent.update_height()
            grandparent = self.parent(parent)
            if grandparent is not None:
                grandparent.update_height()
                self.balance_after_insert(grandparent)
        return z

    def delete(self, target_key):
        z = self.find(target_key)
        if z is None:
            raise KeyError(str(target_key) + " not found in the Tree.")
        # if z has both subtrees
        if z.right and z.left:
            x = self.find_min(z.right)
            # if the successor x has a right child, then replace the successor with the right child
            # note that since x is a min, it cannot have a left child
            if x.right:
                # here the action point, i.e. where the first imbalance might occur is at the parent of x
                x.swap(x.right)
                y = self.parent(x)  # the action point is the parent of the deleted node
            else:
                # the successor is just a leaf node, replace it with None
                x.parent.child[x is not x.parent.left] = None
                y = self.parent(x)

            # copy values of x to z and start balancing at the action point y
            z.key = x.key
            z.item = x.item
            self.__balance_delete(y)

        # the node is a left child and right child exists:
        else:
            if z.right:
                z.swap(z.right)
                self.__balance_delete(z.parent)
            elif z.left:
                z.swap(z.left)
                self.__balance_delete(z.parent)
            else:
                if z is self.root:
                    self.root = None
                else:
                    z.parent.child[z is not z.parent.left] = None
                    self.__balance_delete(z.parent)

    def check_avl_property(self):
        return self.__check_avl_property_helper(self.root)

    @staticmethod
    def __height_node(node):
        if node is None:
            return -1
        else:
            return node.h

    def balance_after_insert(self, node: InternalAVLNode):
        """Re-balances AVL Tree starting assuming that the subtrees rooted at z have AVL property
        Non-recursive version"""

        while True:
            if node.is_skewed():
                dir_parent = node.skew_direction()
                w = node.choose_dir(dir_parent)
                dir_child = w.skew_direction()
                if dir_parent != dir_child:
                    self.__rotate(w, dir_parent)
                    self.__rotate(node, dir_child)
                else:
                    self.__rotate(node, not dir_child)

                node = w
                if node is not None:
                    node.update_height()
                else:
                    break
            else:
                node = self.parent(node)
                if node is not None:
                    node.update_height()
                else:
                    break

    def __balance_delete(self, y):
        """assumes node y is the first possibly unbalanced node
        checks for all the cases of possible imbalance and fixes them
        """
        if y is not None:
            y.update_height()
            while True:
                # find the first skewed node
                if self.__is_skewed(y):
                    y_initial_parent = y.parent
                    dir_sib = self.__skew_direction(y)
                    w = y.child[dir_sib]
                    if self.__height_node(w.child[not dir_sib]) > self.__height_node(
                        w.child[dir_sib]
                    ):
                        self.__rotate(w, dir_sib)
                        self.__rotate(y, not dir_sib)
                    else:
                        self.__rotate(y, not dir_sib)

                    y = y_initial_parent
                    if y is not None:
                        y.update_height()
                    else:
                        break
                else:
                    y = y.parent
                    if y is not None:
                        y.update_height()
                    else:
                        break

    def __rotate(self, y: InternalAVLNode, direction: int):
        """x is the node to be taken to the top
        y = x.parent
                y                              x
               / \                            / \
              x   Ω  = {right_rotate(y)} =>  𝛼   y
             / \                                / \
            𝛼   ß                              ß   Ω
        the comments use the specific case of a right-rotation which of course is symmetric to left-rotation
        """

        if y is None:
            raise ValueError("can't rotate null value")

        attrs = ["left", "right"]

        # move ß to the left child of y
        x = y.__getattribute__(attrs[not direction])
        beta = x.__getattribute__(attrs[direction])
        y.__setattr__(attrs[not direction], beta)

        # now we deal with replacing y with x in z (y.parent)
        z = self.parent(y)
        if z is None:
            # y was the root
            self.root = x
        # make the initial parent of y the parent of x, if y is not the root, i.e y.parent == z is not self.null
        else:
            z.__setattr__(attrs[y is not z.left], x)

        # make y x's subtree, and make sure to make y.parent x
        x.__setattr__(attrs[direction], y)

        # update height of children and heights, the order must be maintained
        x.update_heights()

    def __print_helper(self, node, indent, last):
        """Simple recursive tree printer"""
        if node is not None:
            print(indent, end="")
            if last:
                print("R----", end="")
                indent += "     "
            else:
                print("L----", end="")
                indent += "|    "
            print(
                "(" + str(node.key) + ", " + str(node.item) + ", " + str(node.h) + ")"
            )
            self.__print_helper(node.left, indent, False)
            self.__print_helper(node.right, indent, True)

    def __str__(self):
        if self.root is None:
            return "Empty"
        else:
            self.__print_helper(self.root, "", True)
            return ""

    def __check_avl_property_helper(self, node):
        if node is None:
            return -1
        else:
            hl = self.__check_avl_property_helper(node.left)
            hr = self.__check_avl_property_helper(node.right)
            assert abs(hr - hl) <= 1, repr(node)
            return max(hr, hl) + 1


if __name__ == "__main__":
    from random import randint
    from datetime import datetime

    # values = [3, 52, 31, 55, 93, 60, 81, 93, 46, 37, 47, 67, 34, 95, 10, 23, 90, 14, 13, 88]
    num_nodes = 1000
    values = [(randint(0, 100000), randint(0, 10000)) for _ in range(num_nodes)]
    t1 = datetime.now()
    avl = AVLTree()
    for k, val in values:
        avl.insert(k, val)
    # avl.check_avl_property()
    print(
        f"AVL tree ran in"
        f" {(datetime.now() - t1).total_seconds()} seconds for {num_nodes} insertions."
    )
    print(avl.root.h)
    # print(avl)
    # print(avl.root.height)

    avl.root.check_bst_property(0, 10110000101010)
