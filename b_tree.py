from bisect import bisect_right, bisect_left, insort

__author__ = "Erastus Murungi"
__email__ = "erastusmurungi@gmail.com"


class BNode:
    # t is the minimum degree of the tree, which is >= 2.
    t = 2
    def __init__(self, val=None):
        if val is None:
            self.key = []
        else:
            self.key = [val]
        self.children = []

    @property
    def isleaf(self) -> bool:
        """Returns True if this BTree Node is a leaf and false if it is an internal node."""
        return len(self.children) == 0

    @property
    def isempty(self) -> bool:
        """Returns True if this BTree node has no keys and false otherwise."""
        return len(self.key) == 0

    def pos(self, k) -> int:
        """Get the the number of elements strictly less than k quickly in O(log n) time."""
        return bisect_left(self.key, k)

    @property
    def n(self):
        """Returns the number of keys in this B-Tree node."""
        return len(self.key)

    def rank(self, k):
        """Get the number of elements less than or equal to k in O(log n) time."""
        return bisect_right(self.key, k)

    def inorder(self):
        """Return the inorder traversal of the keys in the tree."""
        if self.isleaf:
            yield from self.key
        else:
            for i, key in enumerate(self.key):
                yield from self.children[i].inorder()
                yield self.key[i]
            yield from self.children[-1].inorder()

    def linear_rank(self, k):
        """Return the number of strictly less than k in O(n) time"""
        i = 0
        while i < self.n and k > self.key[i]:
            i += 1
        return i

    def recursive_search(self, k):
        """ Recursively searches for the given keys in the subtree rooted at self.
            If the key is found, the function returns the node containing the key
            and the index of k in the array of keys stored at the node. Otherwise returns None.
        """
        i = self.pos(k)
        if i < self.n and k == self.key[i]:
            return self, i
        if self.isleaf:
            return None
        else:
            return self.children[i].recursive_search(k)

    def iterative_search(self, k):
        """Iterative version of the search method."""
        node = self
        while True:
            i = node.pos(k)
            if i < node.n and k == node.key[i]:
                return node, i
            if node.isleaf:
                return None
            else:
                node = node.children[i]

    @property
    def isfull(self):
        return len(self.key) == (self.t << 1) - 1

    def insert_nonfull(self, k):
        if self.isleaf:
            insort(self.key, k)
        else:
            # rank is the number of keys larger than or equal to k
            poschild = self.rank(k)
            child = self.children[poschild]
            if child.isfull:
                self.split(poschild)
                if k > self.key[poschild]:
                    child = self.children[poschild + 1]
            child.insert_nonfull(k)

    def split(self, i):
        """splits a child node. Assuming the child node has 2t - 1 keys"""
        # num_nodes = 2t - 1

        t, z, y = self.t, BNode(), self.children[i]

        self.children.insert(i + 1, z)
        self.key.insert(i, y.key[t - 1])

        z.key.extend(y.key[t:])  # split the keys
        y.key = y.key[:t - 1]

        if not y.isleaf:
            z.children.extend(y.children[t:])
            y.children = y.children[:t]

    def nsmallest(self, n):
        if n < 0:
            return None
        if self.isleaf:
            if n >= self.n:
                return None
            else:
                return self.key[n]
        j, s = 0, 0
        while j < self.n and s <= n:
            j, s = j + 1, s + self.children[j].numkeys() + 1  # plus 1 for plus the element itself
        if s - 1 == n:
            return self.key[j - 1]
        elif s > n and j > 0:
            return self.children[j - 1].nsmallest(n - (s - self.children[j - 1].numkeys() - 1))
        else:  # s >= n
            return self.children[j].nsmallest(n - s)

    def delete(self, key):
        """Deletes a key from the btree. Returns true was present and has been deleted. else returns False."""
        node = self
        while True:
            i = node.pos(key)
            if i < node.n and node.key[i] == key:
                # we have found it
                if node.isleaf:
                    node.key.pop(i)
                    return True
                else:
                    return node._delete_from_internal(i)
            else:
                if node.isleaf:
                    print("Key not found!")
                    return False
                flag = i == node.n
                if node.children[i].n < self.t:  # not full
                    node._fill(i)
                node = node.children[i - 1] if flag and i > node.n else node.children[i]

    def _borrow_prev(self, i):
        child, _prev = self.children[i], self.children[i - 1]
        # Shift the keys and children to the right to create space for one key and extra child pointer.
        child.key.insert(0, self.key[i - 1])
        if not _prev.isleaf:
            child.children.insert(0, _prev.children.pop())
        # Move the last key of the prev child to the parent.
        self.key[i - 1] = _prev.key.pop()

    def _borrow_next(self, i):
        child, _next = self.children[i], self.children[i + 1]
        child.key.append(self.key[i])
        if not _next.isleaf:
            child.children.append(_next.children.pop(0))
        self.key[i] = _next.key.pop(0)

    def _fill(self, i):
        """Fill the child at index which has less that t - 1 keys."""
        if i > 0 and self.children[i - 1].n >= self.t:
            self._borrow_prev(i)
        elif i < self.n and self.children[i + 1].n >= self.t:
            self._borrow_next(i)
        else:
            if i < self.n:
                self._merge(i)
            else:
                self._merge(i - 1)

    def _delete_from_internal(self, i):
        """Assumes that self is an internal node from which we are deleting the key at index i."""
        assert not self.isleaf, "This function deletes from internal nodes only."

        k = self.key[i]

        if self.children[i].n >= self.t:
            pred = self._predecessor(i)
            self.key[i] = pred
            self.children[i].delete(pred)

        elif self.children[i + 1].n >= self.t:
            succ = self._successor(i)
            self.key[i] = succ
            self.children[i + 1].delete(succ)

        else:
            self._merge(i)
            self.children[i].delete(k)

    def _merge(self, i):
        """Merges self.children[i] and self.children[i + 1]."""

        child = self.children[i]
        sibling = self.children[i + 1]

        # pulling the key we wanted to delete from the current node and inserting
        child.key.append(self.key[i])
        child.key.extend(sibling.key)
        child.children.extend(sibling.children)

        # delete child pointer and key from parent
        self.children.pop(i + 1)
        self.key.pop(i)

    def _successor(self, i):
        """Returns the successor of an element once the index is known.
            Doesn't handle edge cases because its a private method."""

        current = self.children[i + 1]
        while not current.isleaf:
            current = current.children[0]
        return current.key[0]

    def _predecessor(self, i):
        """Returns the predecessor of the key and index i in the current node."""

        current = self.children[i]
        while not current.isleaf:
            current = current.children[-1]
        return current.key[-1]

    def successor(self, k):
        """Return the next-larger key. This method does not handle duplicates. """
        if self.isempty:
            return None
        node, parent_key = self, None
        i = node.pos(k)
        while not node.isleaf or (i < node.n and node.key[i] != k):
            parent_key = node.key[i - 1] if i == node.n else node.key[i]
            if node.children[i].maximum() < k:
                break
            else:
                node = node.children[i]
            i = node.pos(k)

        if node.isleaf:
            if i == 0:
                if node.key[i] > k:
                    return node.key[i]
                else:
                    return parent_key
            elif i < node.n - 1:
                return node.key[i + 1]
            else:
                return None
        else:
            # the key was found and the node must be an internal node
            return node._successor(i)

    def predecessor(self, k):
        """Return the next-smaller key."""
        raise NotImplemented

    def minimum(self):
        if self.isleaf:
            return self.key[0]
        else:
            return self.children[0].minimum()

    def maximum(self):
        if self.isleaf:
            return self.key[-1]
        else:
            return self.children[-1].maximum()

    def __repr__(self):
        return repr(self.key)

    def __str__(self):
        return self.__repr__()

    def numkeys(self):
        if self.isleaf:
            return len(self.key)
        else:
            return sum(child.numkeys() for child in self.children) + len(self.key)

    def height(self):
        if self.isleaf:
            return 0
        else:
            return max(child.height() for child in self.children) + 1


class BTree:
    def __init__(self, t):
        assert t >= 2, "The minimum degree t must be >= 2."
        self.root = BNode()
        self.t = BNode.t = t

    def __len__(self):
        return self.root.numkeys()

    @property
    def isempty(self):
        return self.root.isempty

    def insert(self, k):
        r = self.root
        if r.isfull:
            s = BNode()
            self.root = s
            s.children.append(r)
            s.split(0)
            s.insert_nonfull(k)
        else:
            r.insert_nonfull(k)

    def delete(self, key):
        self.root.delete(key)

        if self.root.isempty:
            if not self.root.isleaf:
                self.root = self.root.children[0]

    @staticmethod
    def combine(t1: "BTree", t2: "BTree", key: float) -> "BTree":
        """Pass"""
        assert t1.t == t2.t, "the degree t of the two trees must be the same"
        raise NotImplemented

    def os_rank(self, key):
        """Returns the number of elements less than or equal to the current key."""

        r, s, node = 0, 0, self.root
        while not node.isleaf:
            r = node.rank(key)
            s = s + sum(c.numkeys() for c in node.children[:r]) + r
            node = node.children[r]
        r = node.rank(key)
        if r > 0:
            s = s + sum(c.numkeys() for c in node.children[:r]) + r
        return s

    def __repr__(self):
        return repr(self.root)

    def nsmallest(self, n):
        if n < 0:
            return self.root.nsmallest(self.root.numkeys() + n)
        return self.root.nsmallest(n)

    def __contains__(self, item):
        return self.root.iterative_search(item) is not None

    @property
    def inorder(self):
        return self.root.inorder()

    @property
    def minimum(self):
        if self.root.n == 0:
            return None
        else:
            return self.root.minimum()

    @property
    def maximum(self):
        if self.root.n == 0:
            return None
        else:
            return self.root.maximum()

    def nlargest(self, i):
        return self.root.nsmallest(self.root.numkeys() - i - 1)

    @property
    def height(self):
        return self.root.height()

    def _check(self, node):
        """Helper method for check btree properties."""
        if node.isleaf:
            assert self.t - 1 <= node.n <= ((self.t << 1) - 1), repr(node)
            return 0
        else:
            assert self.t - 1 <= node.n <= ((self.t << 1) - 1), repr(node)
            assert self.t <= len(node.children) <= (self.t << 1), repr(node)

            h = self._check(node.children[0])
            assert all(self._check(node.children[j]) == h for j in range(1, len(node.children)))
            return h + 1

    def check_btree_properties(self):
        """ Checks that for every node other than the root:
            1.  a) t - 1 <= x.n <= 2t - 1
                b) t <= x.num_children <= 2t
            2.  The height of every subtree rooted at self is the same.
        """

        h = self._check(self.root.children[0])
        for j in range(1, len(self.root.children)):
            assert h == self._check(self.root.children[j]), repr(self.root.children[j])
        return True


if __name__ == '__main__':
    import numpy as np
    from random import shuffle, randint

    num_iter = 1

    for _ in range(num_iter):
        values = np.random.randint(0, 10000, 2000)
        # values = [2, 32, 32, 41, 46, 50, 52, 69, 71, 71, 73, 74, 89, 93, 98]
        btree = BTree(t=2)
        for v in values:
            btree.insert(v)

        x = tuple(btree.inorder)
        # print(x)
        assert (btree.minimum == min(values))
        assert (btree.maximum == max(values))
        assert (len(btree) == len(x))

        btree.check_btree_properties()

        shuffle(values)
        for v in values:
            assert v in btree
        for v in values:
            btree.delete(v)

        assert btree.isempty
