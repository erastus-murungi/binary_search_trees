from random import randint

LEFT = 0  # False
RIGHT = 1  # True

__author__ = 'Erastus Murungi'
__email__ = 'murungi@mit.edu'


class BSTNode:
    def __init__(self, key, item, parent=None):
        """constructor"""
        self.key = key
        self.item = item
        self.child = [None, None]
        self.parent = parent

    def __repr__(self):
        return "Node(({}, {}), children: {}])".format(str(self.key), str(self.item), repr(self.child))


class BST:
    """A simple full implementation of a binary search tree"""

    def __init__(self):
        self.root = None
        self.size = 0

    def insert(self, key, item=0):
        """Inserts a node with a key and item
        Assumes key is comparable with the other keys"""

        # if the tree was empty, just insert the key at the root
        if self.root is None:
            self.root = BSTNode(key, item)
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
                y.child[y.child[LEFT] is not None] = BSTNode(key, item, y)
            else:
                # while loop broke because x is a leaf, just insert the node in the leaf x
                x.child[key >= x.key] = BSTNode(key, item, x)
        self.size += 1

    def find(self, key):
        """Returns the node with the current key if key exists else None
        We impose the >= condition instead of > because a node with a similar key to the current node in
        the traversal to be placed in the right subtree"""

        # boilerplate code
        if self.root is None:
            return None

        # traverse to the lowest node possible
        current = self.root
        while current is not None and current.key != key:
            # the expression 'key >= current.key' evaluates to True => 1 => RIGHT or False => 0 => LEFT
            current = current.child[key >= current.key]

        if current is None or current.key != key:
            return None
        else:
            return current

    def __contains__(self, item):
        """Returns True if the node with the key is in the BST and False otherwise"""

        return self.find(item) is not None

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

        if self.root is not None:
            yield from helper(self.root)
        else:
            yield ''

    def iteritems(self):
        def helper(node):
            if node.child[LEFT]:
                yield from helper(node.child[LEFT])
            yield node.key, node.item
            if node.child[RIGHT]:
                yield from helper(node.child[RIGHT])

        if self.root is not None:
            yield from helper(self.root)
        else:
            yield ''

    def is_empty(self):
        return self.root is None

    @staticmethod
    def is_leaf(node):
        """Returns True if node is a leaf"""
        return node.child[LEFT] is None and node.child[RIGHT] is None

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
            return None

        # traverse to the leftmost node
        while x.child[LEFT] is not None:
            x = x.child[LEFT]
        return x

    @staticmethod
    def find_max(x):
        """Return the node maximum key in x's subtree"""

        if x is None:
            return None

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

        self.size -= 1
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

        self.size -= 1
        return ret

    def successor(self, current: BSTNode) -> BSTNode:
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

    def height(self):
        """Get the height of the tree"""
        def helper(node):
            if node is None:
                return -1
            else:
                return max(helper(node.child[LEFT]), helper(node.child[RIGHT])) + 1
        return helper(self.root)

    def s_value(self):
        def helper(node):
            if node is None:
                return - 1
            else:
                return min(helper(node.child[LEFT]), helper(node.child[RIGHT])) + 1
        return helper(self.root)

    def predecessor(self, current: BSTNode) -> BSTNode:
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

    def check_weak_search_property(self):
        """Recursively checks's whether x.left.key <= x.key >= x.right.key
        I have no formal reason to call it 'weak' search other than that this method does not
        check whether 'all_keys_in_left_subtree' <= x.key >= 'all_keys_in_right_subtree"""
        return self.__check_weak_search_property_helper(self.root)

    def __check_weak_search_property_helper(self, node):
        """Helper method"""
        if node is None:
            return True
        else:
            if node.child[LEFT] is not None:
                assert node.key >= node.child[LEFT].key, repr(node)
            if node.child[RIGHT] is not None:
                assert node.key <= node.child[RIGHT].key, repr(node)

            self.__check_weak_search_property_helper(node.child[LEFT])
            self.__check_weak_search_property_helper(node.child[RIGHT])
        return True

    @staticmethod
    def __replace(x: BSTNode, y: BSTNode):
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

        self.size -= 1

    def clear(self):
        """delete all the elements in the tree,
        this time without maintaining red-black tree properties """

        self.__clear_helper(self.root.child[LEFT])
        self.__clear_helper(self.root.child[RIGHT])
        self.root = None
        self.size = 0

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

    def __len__(self):
        return self.size

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

    for i, val in enumerate(values):
        bst.insert(val)
        print(values[i] in bst)

    print(bst)
