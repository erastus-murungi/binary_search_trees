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
                                                          str(self.color), repr(self.child))

class RedBlackTree:
    def __init__(self):
        self.null = RBNode.null
        self.root = self.null

    def rotate(self, x: RBNode, direction: float):
        """x is the node to be taken to the top
        y = x.parent

                y                              x
               / \                            / \
              x   Ω   ={right_rotate(x)} =>   𝛼  y
             / \                               / \
            𝛼   ß                             ß  Ω

        the comments use the specific case of a right-rotation which of course is symmetric to left-rotation
        """

        if x == self.root:
            raise ValueError("can't rotate the rotate, instead rotate the children of the root")

        # move ß to the right subtree of y, make sure ß.parent is y
        y = x.parent
        a = x.child[direction]
        y.child[not direction] = a

        # sometimes x.left might be an external node, we need to be careful about that
        # since we are not using a universal null pointer

        if a is not None:
            a.parent = y

        # make y x's subtree, and make sure to make y.parent x
        z = y.parent
        x.child[direction] = y.parent
        y.parent = x

        # make the initial parent of y the parent of x, if y is not the root, i.e y.parent == z is not None
        if z is not self.null:
            # z must be the root
            z.child[y.key >= z.key] = x

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

    def fix_ranks(self, z):
        p = z.parent
        while p.color == RED:
            if p == p.parent.child[LEFT]:  # case 1
                y = p.parent.child[RIGHT]
                if y.color == RED:
                    # color z's parent, z's uncle BLACK and z's grandparent RED
                    p.color = BLACK
                    y.color = BLACK
                    x.p.color = RED
                    # update x to the grandparent
                    z = p.parent

                else:
                    if z == z.parent.child[RIGHT]:
                        p = z.parent
                        # bring z to the subtree root
                        self.rotate(z, LEFT)
                        z = p

                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self.rotate(z.parent, RIGHT)
            else:
                # p == p.parent.child[RIGHT]
                y = p.parent.child[LEFT]
                if y.color == RED:
                    # color z's parent, z's uncle BLACK and z's grandparent RED
                    p.color = BLACK
                    y.color = BLACK
                    x.p.color = RED
                    # update x to the grandparent
                    z = p.parent

                else:
                    if z == z.parent.child[LEFT]:
                        p = z.parent
                        # bring z to the subtree root
                        self.rotate(z, RIGHT)
                        z = p

                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self.rotate(z.parent, BLACK)



if __name__ == '__main__':
    rb = RedBlackTree()
    values = [3, 52, 31, 55, 93, 60, 81, 93, 46, 37, 47, 67, 34, 95, 10, 23, 90, 14, 13, 88]
    for val in values:
        rb.insert(val)
    x = rb.find(52)
    print(x)
    rb.rotate(x, LEFT)
