from dataclasses import dataclass
from sys import maxsize
from typing import Iterator, Optional

from typeguard import typechecked

from bst import BinarySearchTreeIterative, Comparable, Internal, KeyValue, Node, Value


@dataclass(slots=True)
class AVLAuxiliaryData:
    height: int = 0


def height(node: Node) -> int:
    if isinstance(node, Internal):
        return node.aux.height
    return -1


def balance_factor(node: Node) -> int:
    if isinstance(node, Internal):
        return height(node.right) - height(node.left)
    return 0


def update_height(node: Node):
    if isinstance(node, Internal):
        assert isinstance(node.aux, AVLAuxiliaryData)
        node.aux.height = max(height(node.left), height(node.right)) + 1


def right_rotate(node: Node) -> Internal:
    """
             y                              x
           // \\                          // \\
          x   Ω  = {right_rotate(y)} =>  𝛼   y
        // \\                              // \\
        𝛼   ß                              ß   Ω
    """
    assert isinstance(node, Internal)
    assert isinstance(node.left, Internal)
    left_child = node.left

    node.left = left_child.right
    left_child.right = node

    update_height(node)
    update_height(left_child)

    return left_child


def left_rotate(node: Node) -> Internal:
    assert isinstance(node, Internal)
    assert isinstance(node.right, Internal)
    right_child = node.right

    node.right = right_child.left
    right_child.left = node

    update_height(node)
    update_height(right_child)

    return right_child


def re_balance(node: Internal) -> Optional[Internal]:
    bf = balance_factor(node)
    if bf < -1:
        if balance_factor(node.left) > 0:
            node.left = left_rotate(node.left)
        return right_rotate(node)
    if bf > 1:
        if balance_factor(node.right) < 0:
            node.right = right_rotate(node.right)
        return left_rotate(node)
    return None


class AVLTreeIterative(BinarySearchTreeIterative[Comparable, Value, AVLAuxiliaryData]):
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

    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        raise NotImplementedError()

    def check_invariants(self, lower_limit: Comparable, upper_limit: Comparable):
        super().check_invariants(lower_limit, upper_limit)
        for node in self.inorder():
            assert abs(balance_factor(node)) in (-1, 0, 1)

    def insert(
        self, key: Comparable, value: Value, aux: Optional[AVLAuxiliaryData] = None
    ) -> Optional[Node]:
        ancestry = self.insert_ancestry(key, value, AVLAuxiliaryData(height=0))
        if ancestry:
            # fixup the height and re-balance the tree
            node = ancestry[-1]
            self.avl_fixup(ancestry)
            return node
        return None

    def avl_fixup(self, ancestry: list[Internal]):
        while ancestry:
            node = ancestry.pop()
            update_height(node)
            if (new_node := re_balance(node)) is not None:
                self.set_child(ancestry, new_node)

    @typechecked
    def extract_min_iterative(self, node: Internal) -> KeyValue:
        ancestry_min = self.access_ancestry_min(node)
        self.check_invariants(-float("inf"), float("inf"))
        keyval = super().extract_min_iterative(node)
        self.avl_fixup(ancestry_min)
        return keyval

    def delete(self, target_key: Comparable):
        if ancestry := self.access_ancestry(target_key):
            super().delete(target_key)
            self.avl_fixup(ancestry)


if __name__ == "__main__":
    from random import randint

    for _ in range(1000):
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
        num_nodes = 12
        values = list({randint(0, 1000) for _ in range(num_nodes)})
        # values = [608, 612, 583, 234, 875, 618, 689, 625, 376, 409, 701, 671]
        # values = [610, 511]
        avl = AVLTreeIterative()
        for i, k in enumerate(values):
            avl.insert(k, None)
            assert k in avl, values
            avl.check_invariants(-maxsize, maxsize)
            assert len(avl) == i + 1, values
        # print(avl.pretty_str())
        for i, k in enumerate(values):
            del avl[k]
            assert k not in avl, values
            avl.check_invariants(-maxsize, maxsize)
            assert len(avl) == len(values) - i - 1, values
