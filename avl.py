from dataclasses import dataclass

from typeguard import typechecked

from bst import Node, BST, InternalNode, Comparable, Value


def height(node: Node) -> int:
    return node.height if node else -1


def balance_factor(node: Node) -> int:
    if node:
        return height(node.right) - height(node.left)
    return 0


def update_height(node: Node):
    node.height = max(height(node.left), height(node.right)) + 1


@dataclass(slots=True)
class InternalAVLNode(InternalNode[Comparable, Value]):
    """As a reminder, the empty tree has height -1,
    the height of a nonempty tree whose root’s left child has height h₁ and
    root’s right child has height h₂ is 1 + max(h₁, h₂)"""

    left: "InternalAVLNode[Comparable, Value]" = None
    right: "InternalAVLNode[Comparable, Value]" = None
    height: int = 0

    def right_rotate(self):
        """
                 y                              x
               / \                            / \
              x   Ω  = {right_rotate(y)} =>  𝛼   y
             / \                                / \
            𝛼   ß                              ß   Ω
        """
        assert self.left
        left_child = self.left

        self.left = left_child.right
        left_child.right = self

        update_height(self)
        update_height(left_child)

        return left_child

    def left_rotate(self):
        assert self.right
        right_child = self.right

        self.right = right_child.left
        right_child.left = self

        update_height(self)
        update_height(right_child)

        return right_child

    def re_balance(self):
        bf = balance_factor(self)
        if bf < -1:
            if balance_factor(self.left) > 0:
                self.left = self.left.left_rotate()
            return self.right_rotate()
        if bf > 1:
            if balance_factor(self.right) < 0:
                self.right = self.right.right_rotate()
            return self.left_rotate()
        return None


class AVL(BST[Comparable, Value, InternalAVLNode]):
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

    def check_invariants(self, lower_limit: Comparable, upper_limit: Comparable):
        super().check_invariants(lower_limit, upper_limit)
        for node in self.inorder():
            assert abs(balance_factor(node)) <= 1

    @property
    def node_class(self) -> type[InternalAVLNode]:
        return InternalAVLNode

    def insert(self, key: Comparable, value: Value) -> Node:
        ancestry, node = [], self.root

        while True:
            if node is None:
                node = self.node_class(key, value)
                if ancestry:
                    ancestry[-1].choose_set(key, node)
                else:
                    self.root = node
                ancestry.append(node)
                break
            elif node.key == key:
                node.value = value
                return node
            else:
                ancestry.append(node)
                node = node.choose(key)

        self.size += 1

        # fixup the height and re-balance the tree
        self.avl_fixup(ancestry)

    def avl_fixup(self, ancestry: list[Node]):
        while ancestry:
            node = ancestry.pop()
            update_height(node)
            if (new_node := node.re_balance()) is not None:
                if ancestry:
                    ancestry[-1].choose_set(new_node.key, new_node)
                else:
                    self.root = new_node

    @typechecked
    def extract_min(self, node: InternalAVLNode):
        current: InternalNode = node
        ancestry = self.access_ancestry(current.key)
        while current.left is not None:
            ancestry.append(current.left)
            current = current.left

        min_key_value = (current.key, current.value)

        if current.right:
            current.swap(current.right)
        elif current is self.root and current.is_leaf:
            assert self.size == 1
            self.root = None
        else:
            assert len(ancestry) >= 2
            ancestry[-2].choose_set(current.key, None)
        self.avl_fixup(ancestry)
        return min_key_value

    def delete(self, target_key: Comparable):
        if (ancestry := self.access_ancestry(target_key)) is None:
            raise ValueError(f"Key {target_key} not found")
        target_node = ancestry[-1]

        if target_node.right and target_node.left:
            # swap key with successor and delete using extract_min
            key, value = self.extract_min(target_node.right)
            target_node.key, target_node.value = key, value

        elif target_node.right:
            target_node.swap(target_node.right)

        elif target_node.left:
            target_node.swap(target_node.left)
        else:
            # the target node is the root, and it is a leaf node, then setting the root to None will suffice
            if target_node is self.root:
                self.root = None
            else:
                assert len(ancestry) >= 2
                assert target_node.is_leaf
                # else the target is a leaf node which has a parent
                parent = ancestry[-2]
                parent.choose_set(target_node.key, None)

        self.avl_fixup(ancestry)
        self.size -= 1


if __name__ == "__main__":

    for _ in range(1):
        values = [
            3,
            52,
            31,
            55,
            93,
            60,
            81,
            93,
            46,
            37,
            47,
            67,
            34,
            95,
            10,
            23,
            90,
            14,
            13,
            88,
        ]
        avl = AVL()
        for i, k in enumerate(values):
            avl.insert(k, None)
            assert k in avl, values
            avl.check_avl_invariant()
            assert len(avl) == i + 1, values
        for i, k in enumerate(values):
            del avl[k]
            assert k not in avl, values
            avl.check_avl_invariant()
            assert len(avl) == len(values) - i - 1, values
