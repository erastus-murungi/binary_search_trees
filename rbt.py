#!/usr/bin/env python

from random import randint

BLACK = 1
RED = 0
LEFT = 0
RIGHT = 1

__author__ = 'Erastus Murungi'


class RBNull:
    """Universal null in red-black trees"""
    def __init__(self):
        self.color = BLACK
        self.parent = None

    def __repr__(self):
        return "NIL"


class RBNode:
    """A template for a node in a red-black tree
    Every node has a 'bit' for color and other field's found in regular BSTs"""
    null = RBNull()

    def __init__(self, key, item, parent, color=RED):
        self.key = key
        self.item = item
        self.parent = parent
        self.color = color
        self.child = [self.null, self.null]

    def __repr__(self):
        return "Node(({}, {}, {}), children: {}])".format(str(self.key), str(self.item),
                                                          str(self.color), ';'.join([repr(c) for c in self.child]))


class RedBlackTree:
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

    def __init__(self):
        self.null = RBNode.null
        self.root = self.null

    def insert(self, key, item=0):
        # perform insertion just like in a normal BST

        # if the tree was empty, just insert the key at the root
        if self.root is self.null:
            z = RBNode(key, item, self.null)
            self.root = RBNode(key, item, self.null)
        else:
            # create x and y for readability
            x = self.root
            y = x.parent

            # the while loop can only stop if we are at an external node or we are at a leaf node
            # the expression 'key >= current.key' evaluates to True => 1 => RIGHT or False => 0 => LEFT
            while x is not self.null and not self.is_leaf(x):
                y = x
                x = x.child[key >= x.key]

            if x is self.null:
                # while loop broke we are at an external node, insert the node in the parent y
                # if left child is not None, expression evaluates to True => 1 => RIGHT
                z = RBNode(key, item, y)
                y.child[y.child[LEFT] is not self.null] = z

            else:
                # while loop broke because x is a leaf, just insert the node in the leaf x
                z = RBNode(key, item, x)
                x.child[key >= x.key] = z

        self.__insert_fix(z)

    def delete(self, target_key):

        # find the key first
        z = self.access(target_key)
        if z is None:
            raise ValueError("key not in tree")

        # y identifies the node to be deleted
        # We maintain node y as the node either removed from the tree or moved within the tree
        y = z
        y_original_color = y.color
        if z.child[LEFT] == self.null:
            # if z.right is also null, then z => nil, the null node will also have a parent pointer
            x = z.child[RIGHT]
            # replace the node with its right child
            self.__replace(z, z.child[RIGHT])
        elif z.child[RIGHT] is self.null:
            x = z.child[LEFT]
            self.__replace(z, z.child[LEFT])
        else:
            y = self.access_min(z.child[RIGHT])  # find z's successor
            y_original_color = y.color
            x = y.child[RIGHT]  # we will start fixing violations from the successor's right child
            # z might be the minimum's parent
            if y.parent == z:
                # x might be null node, in which case we need to give it a parent pointer
                x.parent = y
            else:
                # transplant y with its right child, which might be a null
                self.__replace(y, y.child[RIGHT])
                y.child[RIGHT] = z.child[RIGHT]
                # if y does have a right subtree we need the subtree to point to the new y
                y.child[RIGHT].parent = y
            self.__replace(z, y)
            y.child[LEFT] = z.child[LEFT]
            y.child[LEFT].parent = y
            y.color = z.color
        if y_original_color == BLACK:
            # the x might be a null pointer
            self.__delete_fix(x)

    def is_empty(self):
        return self.root is self.null

    def access(self, key):
        """Returns the node with the current key if key exists else None
        We impose the >= condition instead of > because a node with a similar key to the current node in
        the traversal to be placed in the right subtree"""

        # boilerplate code
        if self.root is self.null:
            raise ValueError("empty tree")

        # traverse to the lowest node possible
        current = self.root
        while current is not self.null and current.key != key:
            # here we use > than because we want to find the first occurrence of a node with a certain key
            current = current.child[key > current.key]

        # the while stopped because we reached the node with the desired key
        if current is self.null:
            return None
        # the while loop stopped because we reached an external node
        else:
            return current

    def search(self, key):
        """Returns true if the key is found in the tree and false otherwise"""
        return self.access(key) is not None

    @staticmethod
    def is_leaf(node):
        """Returns true if an internal node is a leaf node and false otherwise"""
        return node.child[LEFT] == node.child[RIGHT]  # == self.null

    @property
    def minimum(self):
        """Returns a tuple of the (min_key, item) """
        x = self.access_min(self.root)
        return x.key, x.item

    @property
    def maximum(self):
        """Returns a tuple of the (max_key, item) """
        x = self.access_max(self.root)
        return x.key, x.item

    def access_min(self, x: RBNode):
        """Return the node with minimum key in x's subtree"""

        if x is None:
            raise ValueError("x can't be none")

        # traverse to the leftmost node
        while x.child[LEFT] is not self.null:
            x = x.child[LEFT]
        return x

    def access_max(self, x):
        """Return the node maximum key in x's subtree"""

        if x is None:
            raise ValueError("x can't be none")

        # traverse to the rightmost node
        while x.child[RIGHT] is not self.null:
            x = x.child[RIGHT]
        return x

    def extract_max(self):
        """can be used a max priority queue"""
        node = self.access_max(self.root)
        ret = node.key, node.item
        self.delete(node.key)
        return ret

    def extract_min(self):
        """can be used as a min priority queue"""
        node = self.access_min(self.root)
        ret = node.key, node.item
        self.delete(node.key)
        return ret

    def successor(self, current: RBNode) -> RBNode:
        """Find the node whose key immediately succeeds current.key"""
        # boilerplate
        if current is self.null:
            raise ValueError("can't find the node with the key")

        # case 1: if node has right subtree, then return the min in the subtree
        if current.child[RIGHT] is not self.null:
            y = self.access_min(current.child[RIGHT])
            return y

        # case 2: traverse to the first instance where there is a right edge and return the node incident on the edge
        while current.parent is not self.null and current is current.parent.child[RIGHT]:
            current = current.parent

        y = current.parent
        return y

    def predecessor(self, current: RBNode) -> RBNode:
        """Find the node whose key immediately precedes current.key
        It is important to deal with nodes and note their (key, item) pair because
        the pairs are not unique but the nodes identities are unique.
        That us why the comparisons use is rather than '=='.
        """

        # check that the type is correct
        if current is self.null:
            raise ValueError("can't find the node with the given key")

        # case 1: if node has a left subtree, then return the max in the subtree
        if current.child[LEFT] is not self.null:
            y = self.access_max(current.child[LEFT])
            return y

        # case 2: traverse to the first instance where there is a left edge and return the node incident on the edge
        while current.parent is not self.null and current is current.parent.child[LEFT]:
            current = current.parent

        y = current.parent
        return y

    def in_order(self):
        """In-order traversal generator of the BST"""

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

    def clear(self):
        """delete all the elements in the tree,
        this time without maintaining red-black tree properties """

        self.__clear_helper(self.root.child[LEFT])
        self.__clear_helper(self.root.child[RIGHT])
        self.null.parent = None
        self.root = self.null

    def check_black_height_property(self):
        if self.root is not self.null:
            bh = self.__check_black_height_helper(self.root)
            print("Tree black-height =", bh)
        else:
            print("Empty tree")

    def check_weak_search_property(self):
        """Recursively checks's whether x.left.key <= x.key >= x.right.key
        I have no formal reason to call it 'weak' search other than that this method does not
        check whether 'all_keys_in_left_subtree' <= x.key >= 'all_keys_in_right_subtree"""
        return self.__check_weak_search_property_helper(self.root)

    def __rotate(self, y: RBNode, direction: float):
        """x is the node to be taken to the top
        y = x.parent
                y                              x
               / \                            / \
              x   Ω   ={right_rotate(y)} =>   𝛼  y
             / \                               / \
            𝛼   ß                             ß  Ω
        the comments use the specific case of a right-rotation which of course is symmetric to left-rotation
        """

        if y == self.null:
            raise ValueError("can't rotate null value")

        # move ß to the left child of y
        x = y.child[not direction]
        beta = x.child[direction]
        y.child[not direction] = beta

        # set ß's parent to y
        # sometimes ß might be an external node, we need to be careful about that
        if beta is not self.null:
            beta.parent = y

        # now we deal with replacing y with x in z (y.parent)
        z = y.parent
        if z is self.null:
            # y was the root
            self.root = x
        # make the initial parent of y the parent of x, if y is not the root, i.e y.parent == z is not self.null
        else:
            # this part must be >, because of cases like:
            #    z=93
            #     /  \
            #  y=93   95
            #
            # y.key > z.key should evaluate to false so that y goes in the left subtree
            z.child[y is not z.child[LEFT]] = x
        x.parent = z

        # make y x's subtree, and make sure to make y.parent x
        x.child[direction] = y
        y.parent = x

    @staticmethod
    def __insert_first_case(a, y):
        """a is z's parent, y is z's uncle"""
        a.color = BLACK
        y.color = BLACK
        a.parent.color = RED

    def __insert_third_case(self, parent, grandparent, direction):
        """Case three fixup"""
        self.__rotate(grandparent, direction)
        parent.color = BLACK
        grandparent.color = RED

    def __insert_fix(self, z):

        while z.parent.color == RED:
            # let a = x's parent
            a = z.parent
            # if z's parent is a left child, the uncle will be z's grandparent right child
            if a.parent.child[LEFT] == a:
                y = a.parent.child[RIGHT]
                # if z's uncle is RED (and z's parent is RED), then we are in case 1
                if y.color == RED:
                    self.__insert_first_case(a, y)
                    z = z.parent.parent
                # z's uncle is BLACK (z's parent is RED), we check for case 2 first
                else:
                    # if z is a right child (and remember z's parent is a left child), we left-rotate z.p
                    if a.child[RIGHT] == z:
                        z = a
                        self.__rotate(a, LEFT)
                        # z with be back to a child node after the rotation
                    # now we are in case 3
                    self.__insert_third_case(z.parent, z.parent.parent, RIGHT)

            else:
                # z's parent is a right child, z's uncle is the left child of z's grandparent
                y = a.parent.child[LEFT]
                # check for case 1
                if y.color == RED:
                    self.__insert_first_case(a, y)
                    z = z.parent.parent

                else:
                    # z's parent is already a right child, so check if z is a left child and rotate
                    if a.child[LEFT] == z:
                        z = a
                        self.__rotate(z, RIGHT)
                    # now case 3
                    self.__insert_third_case(z.parent, z.parent.parent, LEFT)

        # make the root black
        self.root.color = BLACK

    def __replace(self, u, v):
        # u is the initial node and v is the node to transplant u
        if u.parent is self.null:
            #                g        g
            #                |        |
            #                u   =>   v
            #               / \      / \
            #             u.a  u.ß  v.a v.ß
            self.root = v
        else:
            u.parent.child[u is not u.parent.child[LEFT]] = v
        if v:
            v.parent = u.parent

    def __delete_fix(self, x):
        while x != self.root and x.color == BLACK:
            if x == x.parent.child[LEFT]:
                w = x.parent.child[RIGHT]  # w is the sibling of x
                # case 1: w is red
                # Since w must have black children, we can switch the
                # colors of w and x: p and then perform a left-rotation on x: p without violating any
                # of the red-black properties. The new sibling of x, which is one of w’s children
                # prior to the rotation, is now black
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self.__rotate(x.parent, LEFT)
                    w = x.parent.child[RIGHT]

                # case 2: x’s sibling w is black, and both of w’s children are black
                # the color of w is black, so we take of one black from both and add it to x.parent
                if w.child[LEFT].color == BLACK and w.child[RIGHT].color == BLACK:
                    w.color = RED
                    x = x.parent
                    # if x was black, then the loop terminates, and x is colored black
                else:
                    # case 3: x’s sibling w is black, w’s left child is red, and w’s right child is black
                    # We can switch the colors of w and its left
                    # child w: left and then perform a right rotation on w without violating any of the
                    # red-black properties.

                    if w.child[RIGHT].color == BLACK:
                        w.child[LEFT].color = BLACK
                        w.color = RED
                        self.__rotate(w, RIGHT)
                        w = x.parent.child[RIGHT]

                    # case 4: x’s sibling w is black, and w’s right child is red
                    # By making some color changes and performing a left rotation
                    # on x: p, we can remove the extra black on x, making it singly black, without
                    # violating any of the red-black properties. Setting x to be the root causes the while
                    # loop to terminate when it tests the loop condition
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.child[RIGHT].color = BLACK
                    self.__rotate(x.parent, LEFT)
                    x = self.root
            else:
                # symmetric to the cases 1 to case 4
                w = x.parent.child[LEFT]
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self.__rotate(x.parent, RIGHT)
                    w = x.parent.child[LEFT]
                if w.child[LEFT].color == BLACK and w.child[RIGHT].color == BLACK:
                    w.color = RED
                    x = x.parent
                else:
                    if w.child[LEFT].color == BLACK:
                        w.child[RIGHT].color = BLACK
                        w.color = RED
                        self.__rotate(w, LEFT)
                        w = x.parent.child[LEFT]
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.child[LEFT].color = BLACK
                    self.__rotate(x.parent, RIGHT)
                    x = self.root
        x.color = BLACK

    def __check_weak_search_property_helper(self, node):
        """Helper method"""
        if node is self.null:
            return True
        else:
            if node.child[LEFT] is not self.null:
                assert node.key >= node.child[LEFT].key, repr(node)
            if node.child[RIGHT] is not self.null:
                assert node.key <= node.child[RIGHT].key, repr(node)

            self.__check_weak_search_property_helper(node.child[LEFT])
            self.__check_weak_search_property_helper(node.child[RIGHT])
        return True

    def __check_black_height_helper(self, node):
        """Checks whether the black-height is the same for all paths starting at the root"""
        if node is self.null:  # we are at an external node
            return 0
        else:
            bh_right = self.__check_black_height_helper(node.child[RIGHT])
            bh_left = self.__check_black_height_helper(node.child[LEFT])
            assert bh_left == bh_right, repr(node)
            if node.color == BLACK:
                return 1 + bh_right
            else:
                return bh_right

    def __print_helper(self, node, indent, last):
        """Simple recursive tree printer"""
        if node is not self.null:
            print(indent, end='')
            if last:
                print("R----", end='')
                indent += "     "
            else:
                print("L----", end='')
                indent += "|    "
            color_to_string = "BLACK" if node.color == 1 else "RED"
            print('<' + str(node.key), node.item, color_to_string + ">")
            self.__print_helper(node.child[LEFT], indent, False)
            self.__print_helper(node.child[RIGHT], indent, True)

    def __clear_helper(self, node):
        if node.child[LEFT] is self.null and node.child[RIGHT] is self.null:
            del node
        else:
            if node.child[LEFT] is not self.null:
                self.__clear_helper(node.child[LEFT])
            if node.child[RIGHT] is not self.null:
                self.__clear_helper(node.child[RIGHT])
            del node

    def __str__(self):
        if self.root is self.null:
            return repr(self.root)
        else:
            self.__print_helper(self.root, "", True)
            return ''

    def __repr__(self):
        return repr(self.root)


if __name__ == '__main__':
    rb = RedBlackTree()
    # values = [3, 52, 31, 55, 93, 60, 81, 93, 46, 37, 47, 67, 34, 95, 10, 23, 90, 14, 13, 88, 88]
    values = [randint(0, 2000000) for _ in range(1_000)]
    for val in values:
        rb.insert(val)

    for val in values:
        rb.delete(val)

    print(rb)
