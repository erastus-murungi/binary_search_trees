from random import randint
from functools import total_ordering

LEFT = 0  # False
RIGHT = 1  # True

__author__ = 'Erastus Murungi' \
             'email: murungi@mit.edu' \

@total_ordering
class Node:
    """A class representing a node in a binary search tree
        Quick note/reminder on the total_ordering decorator:
        "While this decorator makes it easy to create well behaved totally ordered types,
        it does come at the cost of slower execution and more complex stack traces for the derived comparison methods.
        If performance benchmarking indicates this is a bottleneck for a given application,
        implementing all six rich comparison methods instead is likely to provide an easy speed boost."
        Source: https://docs.python.org/3.8/library/functools.html
        """

    def __init__(self, key, item, parent=None):
        """constructor"""
        self.key = key
        self.item = item
        self.child = [None, None]
        self.parent = parent

    def __iter__(self):
        yield self.key
        yield self.item

    def __eq__(self, other):
        return self.key == other.key and self.item == other.item

    def __lt__(self, other):
        # the key is given a higher priority in comparisons
        return self.key < self.key or self.item < other.item

    def __repr__(self):
        return "Node(({}, {}), children: {}])".format(str(self.key), str(self.item), repr(self.child))


class BST:
    """A simple full implementation of a binary search tree"""

    def __init__(self):
        self.root = None

    def find(self, key):
        """Returns the node with the current key if key exists else None
        We impose the >= condition instead of > because a node with a similar key to the current node in
        the traversal to be placed in the right subtree"""

        # boilerplate code
        if self.root is None:
            raise ValueError("empty tree")

        # traverse to the lowest node possible
        current = self.root
        while current.key != key and current is not None:
            # the expression 'key >= current.key' evaluates to True => 1 => RIGHT or False => 0 => LEFT
            current = current.child[key >= current.key]

        # the while stopped because we reached the node with the desired key
        if current.key == key:
            return current
        # the while loop stopped because we reached an external node
        else:
            return None

    def search(self, key):
        """Returns True if the node with the key is in the BST and False otherwise"""

        # boilerplate code
        if self.root is None:
            raise ValueError("empty tree")

        current = self.root
        while current is not None and current.key != key:
            current = current.child[key >= current.key]

        if current is None:
            return False
        else:
            return key == current.key

    def in_order(self):
        """Creates an In-order traversal generator of the BST"""
        def helper(node):
            # visit left node's subtree first if that subtree is not an external node
            if node.child[LEFT]:
                yield from helper(node.child[LEFT])
            # then visit node
            yield node
            # lastly visit the right subtree
            if node.child[RIGHT]:
                yield from helper(node.child[RIGHT])

        return helper(self.root)

    def is_empty(self):
        return self.root is self.null

    @staticmethod
    def is_leaf(node):
        """Returns True if node is a leaf"""
        return node.child[LEFT] is None and node.child[RIGHT] is None

    def insert(self, key, item=0):
        """Inserts a node with a key and item
        Assumes key is comparable with the other keys"""

        # if the tree was empty, just insert the key at the root
        if self.root is None:
            self.root = Node(key, item)
        else:
            # create x and y for readability
            x = self.root
            y = x.parent

            # the while loop can only stop if we are at an external node or we are at a leaf node
            while x is not None and not self.is_leaf(x):
                y = x
                x = x.child[key >= x.key]

            if x is None:
                # while loop broke we are at an external node, insert the node in the parent y
                # if left child is not None, expression evaluates to True => 1 => RIGHT
                y.child[y.child[LEFT] is not None] = Node(key, item, y)
            else:
                # while loop broke because x is a leaf, just insert the node in the leaf x
                x.child[key >= x.key] = Node(key, item, x)

    @property
    def minimum(self):
        """Returns a tuple of the (min_key, item) """
        x = self.find_min(self.root)
        return x.key, x.item

    @property
    def maximum(self):
        """Returns a tuple of the (max_key, item) """
        x = self.find_max(self.root)
        return x.key, x.item

    @staticmethod
    def find_min(x):
        """Return the node with minimum key in x's subtree"""

        if x is None:
            raise ValueError("x can't be none")

        # traverse to the leftmost node
        while x.child[LEFT] is not None:
            x = x.child[LEFT]
        return x

    @staticmethod
    def find_max(x):
        """Return the node maximum key in x's subtree"""

        if x is None:
            raise ValueError("x can't be none")

        # traverse to the rightmost node
        while x.child[RIGHT] is not None:
            x = x.child[RIGHT]
        return x

    def extract_max(self, current=None):
        """Returns a tuple of (max_key, item) and deletes it from the BST"""

        # preprocessing starts
        if self.root is None:
            raise ValueError("empty tree")
        if current is None:
            current = self.root
        # preprocessing ends

        while current.child[RIGHT] is not None:
            current = current.child[RIGHT]
        ret = current.key, current.item

        # if node with max key has a left child, replace that node with its left child
        if current.child[LEFT]:
            self.__replace(current, current.child[LEFT])
        else:
            # if the root is the only remaining node, current.parent.child is None existent
            if current is self.root and self.is_leaf(current):
                self.root = None
                return ret
            # node must be a left node, so just remove it
            current.parent.child[current.key >= current.parent.key] = None

        return ret

    def extract_min(self, current=None):
        """Returns a tuple of (min_key, item) and deletes it from the BST"""

        # pre-processing starts
        if self.root is None:
            raise ValueError("empty tree")
        if current is None:
            current = self.root
        # pre-processing ends

        # traverse to the leftmost node and get its key and satellite data
        while current.child[LEFT] is not None:
            current = current.child[LEFT]
        ret = current.key, current.item

        # if node with min key has a right child, replace the node with its right child
        if current.child[RIGHT]:
            self.__replace(current, current.child[RIGHT])
        else:
            # if the root is the only remaining node, current.parent.child is None existent
            if current is self.root and self.is_leaf(current):
                self.root = None
                return ret
            # node must be a left node, so just remove it
            current.parent.child[current.key >= current.parent.key] = None

        return ret

    def successor(self, current: Node) -> Node:
        """Find the node whose key immediately succeeds current.key"""
        # boilerplate
        if current is None:
            raise ValueError("can't find the node with the key")

        # case 1: if node has right subtree, then return the min in the subtree
        if current.child[RIGHT] is not None:
            y = self.find_min(current.child[RIGHT])
            return y

        # case 2: traverse to the first instance where there is a right edge and return the node incident on the edge
        while current.parent is not None and current is current.parent.child[RIGHT]:
            current = current.parent

        y = current.parent
        return y

    def predecessor(self, current: Node) -> Node:
        """Find the node whose key immediately precedes current.key
        It is important to deal with nodes and note their (key, item) pair because
        the pairs are not unique but the nodes identities are unique.
        That us why the comparisons use is rather than '=='.
        """

        # check that the type is correct
        if current is None:
            raise ValueError("can't find the node with the given key")

        # case 1: if node has a left subtree, then return the max in the subtree
        if current.child[LEFT] is not None:
            y = self.find_max(current.child[LEFT])
            return y

        # case 2: traverse to the first instance where there is a left edge and return the node incident on the edge
        while current.parent is not None and current is current.parent.child[LEFT]:
            current = current.parent

        y = current.parent
        return y

    @staticmethod
    def __replace(x: Node, y: Node):
        """Simple method to cody data from one node to the other"""
        x.key = y.key
        x.item = y.item
        x.child = y.child

    def delete(self, target_key):
        """Deletes a key from the BST"""
        target_node = self.find(target_key)
        if target_node is None:
            return False

        if target_node.child[RIGHT] and target_node.child[LEFT]:
            # swap key with successor and delete using extract_min
            key, item = self.extract_min(target_node.child[RIGHT])
            target_node.key, target_node.item = key, item

        elif target_node.child[LEFT] is None and target_node.child[RIGHT]:
            # replace node with its right child
            self.__replace(target_node, target_node.child[RIGHT])

        elif target_node.child[RIGHT] is None and target_node.child[LEFT]:
            # replace node with its left child
            self.__replace(target_node, target_node.child[LEFT])

        else:
            # the target node is the root and it is a leaf node, then setting the root to None will suffice
            if target_node == self.root:
                self.root = None
            else:
                # else the target is a leaf node which has a parent
                target_node.parent.child[target_key >= target_node.parent.key] = None

    def clear(self):
        """delete all the elements in the tree,
        this time without maintaining red-black tree properties """

        self.__clear_helper(self.root.child[LEFT])
        self.__clear_helper(self.root.child[RIGHT])
        self.root = None

    def __clear_helper(self, node):
        if node.child[LEFT] is None and node.child[RIGHT] is None:
            del node
        else:
            if node.child[LEFT] is not None:
                self.__clear_helper(node.child[LEFT])
            if node.child[RIGHT] is not None:
                self.__clear_helper(node.child[RIGHT])
            del node

    def __print_helper(self, node, indent, last):
        """Simple recursive tree printer"""
        if node is not None:
            print(indent, end='')
            if last:
                print("R----", end='')
                indent += "     "
            else:
                print("L----", end='')
                indent += "|    "
            print('(' + str(node.key) + ', ' + str(node.item) + ")")
            self.__print_helper(node.child[LEFT], indent, False)
            self.__print_helper(node.child[RIGHT], indent, True)

    def __str__(self):
        if self.root is None:
            return str(None)
        else:
            self.__print_helper(self.root, "", True)
            return ''

    def __repr__(self):
        """Simple repr method"""
        return repr(self.root)


if __name__ == '__main__':
    bst = BST()
    values = [randint(1, 100) for _ in range(20)]
    # values = [3, 52, 31, 55, 93, 60, 81, 93, 46, 37, 47, 67, 34, 95, 10, 23, 90, 14, 13, 88]

    for val in values:
        bst.insert(val)
    print(bst)
    
    for val in values:
        bst.delete(val)
    print(bst)
