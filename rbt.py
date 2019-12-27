from random import randint

BLACK = 1
RED = 0
LEFT = 0
RIGHT = 1


class RBNull:
    """Universal null in red-black trees"""
    def __init__(self):
        self.color = BLACK

    def __repr__(self):
        return "NIL"


class RBNode:
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
    def __init__(self):
        self.null = RBNode.null
        self.root = self.null

    def rotate(self, y: RBNode, direction: float):
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
            z.child[y.key >= z.key] = x
        x.parent = z

        # make y x's subtree, and make sure to make y.parent x
        x.child[direction] = y
        y.parent = x

    def find(self, key):
        """Returns the node with the current key if key exists else None
        We impose the >= condition instead of > because a node with a similar key to the current node in
        the traversal to be placed in the right subtree"""

        # boilerplate code
        if self.root is self.null:
            raise ValueError("empty tree")

        # traverse to the lowest node possible
        current = self.root
        while current.key != key and current is not self.null:
            # the expression 'key >= current.key' evaluates to True => 1 => RIGHT or False => 0 => LEFT
            current = current.child[key >= current.key]

        # the while stopped because we reached the node with the desired key
        if current.key == key:
            return current
        # the while loop stopped because we reached an external node
        else:
            return None

    @staticmethod
    def is_leaf(node):
        return node.child[LEFT] == node.child[RIGHT]  # == self.null

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

    @staticmethod
    def case_one(a, y):
        """a is z's parent, y is z's uncle"""
        a.color = BLACK
        y.color = BLACK
        a.parent.color = RED

    def case_three(self, parent, grandparent, direction):
        """Case three fixup"""
        parent.color = BLACK
        grandparent.color = RED
        self.rotate(grandparent, direction)

    def __insert_fix(self, z):

        while z.parent.color == RED:
            # let a = x's parent
            a = z.parent
            # if z's parent is a left child, the uncle will be z's grandparent right child
            if a.parent.child[LEFT] == a:
                y = a.parent.child[RIGHT]
                # if z's uncle is RED (and z's parent is RED), then we are in case 1
                if y.color == RED:
                    self.case_one(a, y)
                    z = z.parent.parent
                # z's uncle is BLACK (z's parent is RED), we check for case 2 first
                else:
                    # if z is a right child (and remember z's parent is a left child), we left-rotate z.p
                    if a.child[RIGHT] == z:
                        z = a
                        self.rotate(a, LEFT)
                        # z with be back to a child node after the rotation
                    # now we are in case 3
                    self.case_three(z.parent, z.parent.parent, RIGHT)

            else:
                # z's parent is a right child, z's uncle is the left child of z's grandparent
                y = a.parent.child[LEFT]
                # check for case 1
                if y.color == RED:
                    self.case_one(a, y)
                    z = z.parent.parent

                else:
                    # z's parent is already a right child, so check if z is a left child and rotate
                    if a.child[LEFT] == z:
                        z = a
                        self.rotate(z, RIGHT)
                    # now case 3
                    self.case_three(z.parent, z.parent.parent, LEFT)

        # make the root black
        self.root.color = BLACK

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
    def find_min(x: RBNode):
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

    def __rb_transplant(self, u, v):
        # u is the initial node and v is the node to transplant u
        if u.parent is self.null:
            self.root = v
        u.parent.child[u is not u.parent.child[LEFT]] = v
        v.parent = u.parent

    def delete(self, target_key):

        # find the key first
        z: RBNode = self.find(target_key)
        if z is None:
            raise ValueError("key not in tree")

        # y identifies the node to be deleted
        # We maintain node y as the node either removed from the tree or moved within the tree
        y = z
        y_original_color = y.color
        if z.child[LEFT] == self.null:
            x = z.child[RIGHT]
            self.__rb_transplant(z, z.child[RIGHT])
        elif z.child[RIGHT] is self.null:
            x = z.child[LEFT]
            self.__rb_transplant(z, z.child[LEFT])
        else:
            # partly resembles an extract min procedure
            y = self.find_min(z.child[RIGHT])  # find z's successor
            y_original_color = y.color
            x = y.child[RIGHT]
            # z might be the minimum's parent
            if y.parent == z:
                x.parent = y
            else:
                self.__rb_transplant(y, y.child[RIGHT])
                y.child[RIGHT] = z.child[RIGHT]
                y.child[RIGHT].parent = y
            self.__rb_transplant(z, y)
            y.child[LEFT] = z.child[LEFT]
            y.child[LEFT].parent = y
            y.color = z.color
        if y_original_color == BLACK:
            self.__delete_fix(x)

    def __delete_fix(self, x):
        while x != self.root and x.color == BLACK:
            if x == x.parent.child[LEFT]:
                s = x.parent.child[RIGHT]
                if s.color == RED:
                    s.color = BLACK
                    x.parent.color = RED
                    self.rotate(x.parent, LEFT)
                    s = x.parent.child[RIGHT]
                if s.child[LEFT].color == BLACK and s.child[RIGHT].color == BLACK:
                    s.color = RED
                    x = x.parent
                else:
                    if s.child[RIGHT].color == BLACK:
                        s.child[LEFT].color = BLACK
                        s.color = RED
                        self.rotate(s, RIGHT)
                        s = x.parent.child[RIGHT]
                    s.color = x.parent.color
                    x.parent.color = BLACK
                    s.child[RIGHT].color = BLACK
                    self.rotate(x.parent, LEFT)
                    x = self.root
            else:
                s = x.parent.child[LEFT]
                if s.color == RED:
                    s.color = BLACK
                    x.parent.color = RED
                    self.rotate(x.parent, RIGHT)
                    s = x.parent.child[LEFT]
                if s.child[RIGHT].color == BLACK and s.child[RIGHT].color == BLACK:
                    s.color = RED
                    x = x.parent
                else:
                    if s.child[LEFT].color == BLACK:
                        s.child[RIGHT].color = BLACK
                        s.color = RED
                        self.rotate(s, LEFT)
                        s = x.parent.child[LEFT]
                    s.color = x.parent.color
                    x.parent.color = BLACK
                    s.child[LEFT].color = BLACK
                    self.rotate(x.parent, RIGHT)
                    x = self.root
        x.color = BLACK

    def successor(self, current: RBNode) -> RBNode:
        """Find the node whose key immediately succeeds current.key"""
        # boilerplate
        if current is self.null:
            raise ValueError("can't find the node with the key")

        # case 1: if node has right subtree, then return the min in the subtree
        if current.child[RIGHT] is self.null:
            y = self.find_min(current.child[RIGHT])
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
            y = self.find_max(current.child[LEFT])
            return y

        # case 2: traverse to the first instance where there is a left edge and return the node incident on the edge
        while current.parent is not self.null and current is current.parent.child[LEFT]:
            current = current.parent

        y = current.parent
        return y

    def __print_helper(self, node, indent, last):
        if node != self.null:
            print(indent, end='')
            if last:
                print("R----", end='')
                indent += "     "
            else:
                print("L----", end='')
                indent += "|    "
            color_to_string = "BLACK" if node.color == 1 else "RED"
            print('(', node.key, node.item, color_to_string, ")")
            self.__print_helper(node.child[LEFT], indent, False)
            self.__print_helper(node.child[RIGHT], indent, True)

    def __str__(self):
        self.__print_helper(self.root, "", True)
        return 'Done'

    def __repr__(self):
        return repr(self.root)


if __name__ == '__main__':
    rb = RedBlackTree()
    values = [3, 52, 31, 55, 93, 60, 81, 93, 46, 37, 47, 67, 34, 95, 10, 23, 90, 14, 13, 88]
    print(values)
    for val in values:
        rb.insert(val)

    print(rb)

    for val in values:
        rb.delete(val)
