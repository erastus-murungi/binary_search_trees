from bst import BSTNode, BST

LEFT, RIGHT = 0, 1


class AVLNode(BSTNode):
    def __init__(self, key, item=0, parent=None):
        super().__init__(key, item, parent)
        self.height = 0

    def update_height(self):
        hl = -1 if self.child[LEFT] is None else self.child[LEFT].height
        hr = -1 if self.child[RIGHT] is None else self.child[RIGHT].height
        self.height = max(hr, hl) + 1

    def __repr__(self):
        return "Node(({}, {}, {}), children: {}])".format(str(self.key), str(self.item),
                                                          str(self.height), ';'.join([repr(c) for c in self.child]))


class AVLTree(BST):
    def __init__(self):
        super(AVLTree, self).__init__()

    def insert(self, key, item=0):
        """Inserts a node with a key and item
        Assumes key is comparable with the other keys"""

        # if the tree was empty, just insert the key at the root
        if self.root is None:
            self.root = AVLNode(key, item)
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
                z = AVLNode(key, item, y)
                y.child[y.child[LEFT] is not None] = z
            else:
                # while loop broke because x is a leaf, just insert the node in the leaf x
                z = AVLNode(key, item, x)
                x.child[key >= x.key] = z

            # re-balancing should start from the grandparent of z

            if z.parent is not None:
                z.parent.update_height()
                if z.parent.parent is not None:
                    z.parent.parent.update_height()
                    self.__balance(z.parent.parent)

    def delete(self, target_key):
        z = self.find(target_key)
        if z is None:
            raise KeyError(str(target_key) + ' not found in the Tree.')

        if z.child[RIGHT] and z.child[LEFT]:
            x = self.find_min(z.child[RIGHT])
            if x.child[RIGHT]:
                self.__replace(x, x.child[RIGHT])
                self.__balance(x.child[RIGHT])
            else:
                if x is self.root and self.is_leaf(x):
                    self.root = None
                else:
                    x.parent.child[LEFT] = None
                    x.parent.update_height()
                    self.__balance(x.parent)

            self.__replace(z, x)
            x.child = z.child
            x.height = z.height
            self.__balance(x)

        # free (z)
        # the node is a left child and right child exists:
        else:
            if z.child[RIGHT]:
                self.__replace(z, z.child[RIGHT])
                self.__balance(z.child[RIGHT])
                # balance starting from the new root which is z.child[RIGHT]
                # free z
            elif z.child[LEFT]:
                self.__replace(z, z.child[LEFT])
                self.__balance(z.child[LEFT])
                # balance starting from the new root which is z.child[RIGHT]
            else:
                z.parent.child[z is not z.parent.child[LEFT]] = None
                z.parent.update_height()
                self.__balance(z.parent)

    @staticmethod
    def __skew_direction(node):
        hl = node.child[LEFT].height if node.child[LEFT] else -1
        hr = node.child[RIGHT].height if node.child[RIGHT] else -1
        return hl < hr

    def check_avl_property(self):
        return self.__check_avl_property_helper(self.root)

    def __balance(self, node):
        """Re-balances AVL Tree starting assuming that the subtrees rooted at z have AVL property
        Non-recursive version"""

        while True:
            if self.__is_skewed(node):
                dir_parent = self.__skew_direction(node)
                w = node.child[dir_parent]
                dir_child = self.__skew_direction(w)
                if dir_parent != dir_child:
                    self.__rotate(w, dir_parent)
                    self.__rotate(node, dir_child)
                else:
                    self.__rotate(node, not dir_child)

                # z.parent is the new root
                node = w
                if node is not None:
                    node.update_height()
                else:
                    break
            else:
                node = node.parent
                if node is not None:
                    node.update_height()
                else:
                    break

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
        v.parent = u.parent

    def __balance_delete(self):
        pass

    @staticmethod
    def __update_heights(node):
        if node.child[LEFT]:
            node.child[LEFT].update_height()
        if node.child[RIGHT]:
            node.child[RIGHT].update_height()
        node.update_height()

    @staticmethod
    def __is_skewed(node):
        hl = node.child[LEFT].height if node.child[LEFT] else -1
        hr = node.child[RIGHT].height if node.child[RIGHT] else -1
        return abs(hr - hl) > 1

    def __rotate(self, y: AVLNode, direction: float):
        """x is the node to be taken to the top
        y = x.parent
                y                              x
               / \                            / \
              x   Ω  = {right_rotate(y)} =>  𝛼   y
             / \                                / \
            𝛼   ß                              ß   Ω
        the comments use the specific case of a right-rotation which of course is symmetric to left-rotation
        """

        if y is None:
            raise ValueError("can't rotate null value")

        # move ß to the left child of y
        x = y.child[not direction]
        beta = x.child[direction]
        y.child[not direction] = beta

        # set ß's parent to y
        # sometimes ß might be an external node, we need to be careful about that
        if beta is not None:
            beta.parent = y

        # now we deal with replacing y with x in z (y.parent)
        z = y.parent
        if z is None:
            # y was the root
            self.root = x
        # make the initial parent of y the parent of x, if y is not the root, i.e y.parent == z is not self.null
        else:
            z.child[y is not z.child[LEFT]] = x
        x.parent = z

        # make y x's subtree, and make sure to make y.parent x
        x.child[direction] = y
        y.parent = x

        # update height of children and heights, the order must be maintained
        self.__update_heights(x)

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
            print('(' + str(node.key) + ', ' + str(node.item) + ', ' + str(node.height) + ")")
            self.__print_helper(node.child[LEFT], indent, False)
            self.__print_helper(node.child[RIGHT], indent, True)

    def __str__(self):
        if self.root is None:
            return str(None)
        else:
            self.__print_helper(self.root, "", True)
            return ''

    def __check_avl_property_helper(self, node):
        if node is None:
            return -1
        else:
            hl = self.__check_avl_property_helper(node.child[LEFT])
            hr = self.__check_avl_property_helper(node.child[RIGHT])
            assert abs(hr - hl) <= 1, repr(node)
            return max(hr, hl) + 1


if __name__ == '__main__':
    from pprint import pprint
    from random import randint
    avl = AVLTree()
    values = [3, 52, 31, 55, 93, 60, 81, 93, 46, 37, 47, 67, 34, 95, 10, 23, 90, 14, 13, 88]

    for val in values:
        avl.insert(val)
        avl.check_avl_property()

    for val in values:
        avl.delete(val)
        print("deleted: ", val)
        avl.check_avl_property()

    print(avl)
