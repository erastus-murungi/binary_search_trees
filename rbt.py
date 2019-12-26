from random import randint

BLACK = 1
RED = 0
LEFT = 0
RIGHT = 1


class RBNull:
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

        self.fix_ranks(z)

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

    def fix_ranks(self, z):

        while z.parent.color == RED:
            # let a = x's parent
            a = z.parent
            if a.parent is not self.null:
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

    def __repr__(self):
        return repr(self.root)

if __name__ == '__main__':
    rb = RedBlackTree()
    values = [3, 52, 31, 55, 93, 60, 81, 93, 46, 37, 47, 67, 34, 95, 10, 23, 90, 14, 13, 88]
    for val in values:
        rb.insert(val)
    x = rb.find(52)
    print(x)
    rb.rotate(x, LEFT)
