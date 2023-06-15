from dataclasses import dataclass
from typing import Iterator, Optional

from typeguard import typechecked

from bst import (BinarySearchTreeIterative, Comparable, Internal, InternalNode,
                 Node, Value)


@dataclass(slots=True)
class AVLAuxiliaryData:
    height: int = 0


def height(node: Node) -> int:
    if isinstance(node, Internal):
        assert isinstance(node.aux, AVLAuxiliaryData)
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
    def yield_line(self, indent: str, prefix: str) -> Iterator[str]:
        raise NotImplementedError()

    def check_invariants(self, lower_limit: Comparable, upper_limit: Comparable):
        super().check_invariants(lower_limit, upper_limit)
        for node in self.inorder():
            if abs(balance_factor(node)) not in (-1, 0, 1):
                raise RuntimeError(
                    f"Invalid AVL Tree, balance factor = {balance_factor(node)} not in [-1, 0, 1]"
                )

    def insert_impl(
        self, key: Comparable, value: Value, aux: Optional[AVLAuxiliaryData] = None
    ) -> InternalNode:
        ancestry = self.insert_ancestry(key, value, AVLAuxiliaryData(height=0))
        node = ancestry[-1]
        self.avl_fixup(ancestry)
        return node

    def avl_fixup(self, ancestry: list[Internal]):
        while ancestry:
            node = ancestry.pop()
            update_height(node)
            if (new_node := re_balance(node)) is not None:
                self.set_child(ancestry, new_node)

    @typechecked
    def extract_min_iterative(self, node: Internal) -> tuple[Comparable, Value]:
        ancestry_min = self.access_ancestry_min(node)
        keyval = super().extract_min_iterative(node)
        self.avl_fixup(ancestry_min)
        return keyval

    def delete(self, target_key: Comparable):
        if ancestry := self.access_ancestry(target_key):
            super().delete(target_key)
            self.avl_fixup(ancestry)

    def extract_min(self) -> tuple[tuple[Comparable, Value] :, Node]:
        min_node = self.minimum()
        keyval = (min_node.key, min_node.value)
        self.delete(min_node.key)
        return keyval, self.root

    def extract_max(self) -> tuple[tuple[Comparable, Value] :, Node]:
        max_node = self.maximum()
        keyval = (max_node.key, max_node.value)
        self.delete(max_node.key)
        return keyval, self.root


if __name__ == "__main__":
    pass
