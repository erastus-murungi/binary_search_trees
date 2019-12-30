LEFT = 0  # False
RIGHT = 1  # True


class Node:
    def __init__(self, key, item, parent=None):
        self.key = key
        self.item = item
        self.child = [None, None]
        self.parent = parent

    def __repr__(self):
        return "Node(({}, {}), children: {}])".format(str(self.key), str(self.item), repr(self.child))


class SplayTree:
    def __init__(self, root=None):
        self.root = root

    def insert(self, key, item=0):
        if self.root is None:
            self.root = Node(key, item)
        else:
            x = self.root
            y = x.parent

            while x is not None and not self.is_leaf(x):
                y = x
                x = x.child[key >= x.key]

            if x is None:
                z = Node(key, item, y)
                y.child[y.child[LEFT] is not None] = z

            else:
                z = Node(key, item, x)
                x.child[key >= x.key] = z

            self.splay(z)

    def access(self, key):
        if self.root is None:
            raise ValueError("empty tree")

        x = self.root
        y = x.parent  # last non-null node
        while x is not None and x.key != key:
            y = x
            x = x.child[key >= x.key]

        if x is None or x.key != key:
            if y is not None:
                self.splay(y)
            return None
        else:
            self.splay(x)
            return x

    def splay(self, x):
        while x.parent is not None:
            p = x.parent
            g = p.parent
            dirx = x is not p.child[LEFT]
            if g is None:  # zig
                self.__rotate(p, not dirx)
            else:
                dirp = p is not g.child[LEFT]
                if dirp == dirx:  # zig-zig
                    self.__rotate(g, not dirp)
                    self.__rotate(p, not dirp)
                else:  # zig-zag
                    self.__rotate(p, dirp)
                    self.__rotate(g, not dirp)

    def join(self, other):
        """Combines trees t1 and t2 into a single tree containing all items from
            both trees and return the resulting tree. This operation assumes that
            all items in t1 are less than all those in t2 and destroys both t1 and t2."""

        assert self.root is not None
        assert other.root is not None

        x = self.find_max(self.root)
        self.splay(x)
        self.root.child[RIGHT] = other.root
        other.root.parent = self.root.child[RIGHT]
        del other

        return self

    def split(self, i):
        """Construct and return two trees t1 and t2, where t1 contains all items
            in t less than or equal to i, and t2 contains all items in t greater than
            i. This operation destroys t. """

        assert self.root is not None

        self.access(i)
        y = self.root.child[RIGHT]
        self.__replace(y, None)
        y.parent = None
        return self, SplayTree(self.root.child[RIGHT])

    def __rotate(self, y, direction):

        if y is None:
            raise ValueError("can't rotate null value")

        x = y.child[not direction]
        beta = x.child[direction]
        y.child[not direction] = beta

        if beta is not None:
            beta.parent = y

        z = y.parent
        if z is None:
            self.root = x
        else:
            z.child[y is not z.child[LEFT]] = x
        x.parent = z

        x.child[direction] = y
        y.parent = x

    def in_order(self):
        """Creates an In-order traversal generator of the BST"""

        def helper(node):
            if node.child[LEFT]:
                yield from helper(node.child[LEFT])
            yield node
            if node.child[RIGHT]:
                yield from helper(node.child[RIGHT])

        return helper(self.root)

    def is_empty(self):
        return self.root is None

    @staticmethod
    def is_leaf(node):
        return node.child[LEFT] is None and node.child[RIGHT] is None

    @property
    def minimum(self):
        x = self.find_min(self.root)
        return x.item

    @property
    def maximum(self):
        x = self.find_max(self.root)
        return x.item

    @staticmethod
    def find_min(x):
        if x is None:
            raise ValueError("Empty Tree!")

        while x.child[LEFT] is not None:
            x = x.child[LEFT]
        return x

    @staticmethod
    def find_max(x):
        if x is None:
            raise ValueError("x can't be none")

        while x.child[RIGHT] is not None:
            x = x.child[RIGHT]
        return x

    def __replace(self, u, v):
        # u is the initial node and v is the node to transplant u
        if u.parent is None:
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

    def delete(self, target_key):
        """Deletes a key from the Splay Tree"""

        z = self.access(target_key)
        if z is None:
            raise KeyError("Key not found")
        left, right = self.root.child[LEFT], self.root.child[RIGHT]

        m = None
        if left:
            m = self.find_max(left)
            left.parent = None
            left = SplayTree(left)
            left.splay(m)
            self.root = left.root
        if right:
            if left:
                self.root.child[RIGHT] = right
            else:
                self.root = right
            right.parent = m

    def clear(self):
        """delete all the elements in the tree """

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
            return 'Empty'
        else:
            self.__print_helper(self.root, "", True)
            return ''

    def __repr__(self):
        """Simple repr method"""
        return repr(self.root)


if __name__ == '__main__':
    from random import choice
    st = SplayTree()
    values = [3, 52, 31, 55, 93, 60, 81, 93, 46, 37, 47, 67, 34, 95, 10, 23, 90, 14, 13, 88]
    values1 = [1, 1, 2]
    st1 = SplayTree()

    for val in values:
        st.insert(val)

    for val in values1:
        st1.insert(val)

    print(st)
    print(st1)
    st2 = st1.join(st)
    print(st2)

    values.extend(values1)
    for val in values:
        st2.delete(val)

    print(st2)



    # for val in values:
    #     st.delete(val)
    #     print(st)
