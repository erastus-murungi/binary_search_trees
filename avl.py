from bst import BSTNode, BST

LEFT, RIGHT = 0, 1


class AVLNode(BSTNode):
    """ As a reminder, the empty tree has height -1,
        the height of a nonempty tree whose root’s left child has height h₁ and
        root’s right child has height h₂ is 1 + max(h₁, h₂)"""

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
        longest root to leaf path.
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
                    self.__balance_insert(z.parent.parent)

    def delete(self, target_key):
        z = self.find(target_key)
        if z is None:
            raise KeyError(str(target_key) + ' not found in the Tree.')
        # if z has both subtrees
        if z.child[RIGHT] and z.child[LEFT]:
            x = self.find_min(z.child[RIGHT])
            # if the successor x has a right child, then replace the successor with the right child
            # note that since x is a min, it cannot have a left child
            if x.child[RIGHT]:
                # here the action point, i.e where the first imbalance might occur is at the parent of x
                self.__replace(x, x.child[RIGHT])
                y = x.parent  # the action point is the parent of the deleted node
            else:
                # the successor is just a leaf node, replace it with None
                x.parent.child[x is not x.parent.child[LEFT]] = None
                y = x.parent

            # copy values of x to z and start balancing at the action point y
            z.key = x.key
            z.item = x.item
            self.__balance_delete(y)

        # the node is a left child and right child exists:
        else:
            if z.child[RIGHT]:
                self.__replace(z, z.child[RIGHT])
                self.__balance_delete(z.parent)
            elif z.child[LEFT]:
                self.__replace(z, z.child[LEFT])
                self.__balance_delete(z.parent)
            else:
                if z is self.root:
                    self.root = None
                else:
                    z.parent.child[z is not z.parent.child[LEFT]] = None
                    self.__balance_delete(z.parent)

    def check_avl_property(self):
        return self.__check_avl_property_helper(self.root)

    @staticmethod
    def __height_node(node):
        if node is None:
            return -1
        else:
            return node.height

    @staticmethod
    def __skew_direction(node):
        hl = node.child[LEFT].height if node.child[LEFT] else -1
        hr = node.child[RIGHT].height if node.child[RIGHT] else -1
        return hl <= hr

    def __balance_insert(self, node):
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

    def __balance_delete(self, y):
        """assumes node y is the first possibly unbalanced node
        checks for all the cases of possible imbalance and fixes them
        """
        if y is not None:
            y.update_height()
            while True:
                # find the first skewed node
                if self.__is_skewed(y):
                    y_initial_parent = y.parent
                    dir_sib = self.__skew_direction(y)
                    w = y.child[dir_sib]
                    if self.__height_node(w.child[not dir_sib]) > self.__height_node(w.child[dir_sib]):
                        self.__rotate(w, dir_sib)
                        self.__rotate(y, not dir_sib)
                    else:
                        self.__rotate(y, not dir_sib)

                    y = y_initial_parent
                    if y is not None:
                        y.update_height()
                    else:
                        break
                else:
                    y = y.parent
                    if y is not None:
                        y.update_height()
                    else:
                        break

    @property
    def height(self):
        return self.__height_node(self.root)

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
        if v is not None:
            v.parent = u.parent

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
            return 'Empty'
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
    from random import randint
    from pympler import asizeof
    from datetime import datetime

    # values = [3, 52, 31, 55, 93, 60, 81, 93, 46, 37, 47, 67, 34, 95, 10, 23, 90, 14, 13, 88]
    num_nodes = 1000_000
    values = [(randint(0, 100000000), randint(0, 10000)) for _ in range(num_nodes)]
    t1 = datetime.now()
    avl = AVLTree()
    for key, val in values:
        avl.insert(key, val)
    avl.check_avl_property()
    print(f"AVL tree used {asizeof.asizeof(avl) / (1 << 20):.2f} MB of memory and ran in"
          f" {(datetime.now() - t1).total_seconds()} seconds for {num_nodes} insertions.")
    print(avl.height)
    # print(avl)
    # print(avl.root.height)

    avl.check_weak_search_property()
