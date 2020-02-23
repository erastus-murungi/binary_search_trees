from bisect import bisect, bisect_left, insort


class BNode:
    t = 2

    def __init__(self, val=None):
        if val is None:
            self.key = []
        else:
            self.key = [val]
        self.children = []

    @property
    def isleaf(self):
        return len(self.children) == 0

    @property
    def isempty(self):
        return len(self.key) == 0

    def pos(self, k) -> int:
        """Get the rank of an element quickly in O(lg n)"""
        return bisect_left(self.key, k)

    @property
    def n(self):
        return len(self.key)

    def rank(self, k):
        return bisect(self.key, k)

    def inorder(self):
        if self.isleaf:
            yield from self.key
        else:
            for i, key in enumerate(self.key):
                yield from self.children[i].inorder()
                yield self.key[i]
            yield from self.children[-1].inorder()

    def linear_rank(self, k):
        """return the number of elements less than or equal to k in O(n) time"""
        i = 0
        while i < self.n and k > self.key[i]:
            i += 1
        return i

    def search(self, k):
        i = self.pos(k)
        if i < self.n and k == self.key[i]:
            return self, i
        if self.isleaf:
            return None
        else:
            return self.children[i].search(k)

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

    def nsmallest(self, i):
        raise NotImplemented

    def nlargest(self, i):
        raise NotImplemented

    def delete(self, key):

        node = self
        while True:
            i = node.pos(key)
            if i < node.n and node.key[i] == key:
                # we have found it
                if node.isleaf:
                    node.key.pop(i)
                    return True
                else:
                    return node.delete_from_internal(i)
            else:
                if node.isleaf:
                    raise ValueError("Key not found!")
                    # return False
                flag = i == node.n
                if node.children[i].n < self.t:  # not full
                    node.fill(i)
                node = node.children[i - 1] if flag and i > node.n else node.children[i]

    def borrow_prev(self, i):
        child, _prev = self.children[i], self.children[i - 1]
        # shift the keys and children to the right to create space for one key and extra child pointer
        child.key.insert(0, self.key[i - 1])
        if not _prev.isleaf:
            child.children.insert(0, _prev.children.pop())
        # move the last key of the prev child to the parent
        self.key[i - 1] = _prev.key.pop()

    def borrow_next(self, i):
        child, _next = self.children[i], self.children[i + 1]

        child.key.append(self.key[i])
        if not _next.isleaf:
            child.children.append(_next.children.pop(0))
        self.key[i] = _next.key.pop(0)

    def fill(self, i):
        """fill the child at index which has less that t - 1 keys"""
        if i > 0 and self.children[i - 1].n >= self.t:
            self.borrow_prev(i)
        elif i < self.n and self.children[i + 1].n >= self.t:
            self.borrow_next(i)
        else:
            if i < self.n:
                self.merge(i)
            else:
                self.merge(i - 1)

    def delete_from_internal(self, i):
        """Assumes that self is an internal node from which we are deleting the key at index i."""
        assert not self.isleaf, "This function deletes from internal nodes."

        k = self.key[i]

        if self.children[i].n >= self.t:
            pred = self.predecessor(i)
            self.key[i] = pred
            self.children[i].delete(pred)

        elif self.children[i + 1].n >= self.t:
            succ = self.successor(i)
            self.key[i] = succ
            self.children[i + 1].delete(succ)

        else:
            self.merge(i)
            self.children[i].delete(k)

    def merge(self, i):
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

    def successor(self, i):
        # the index of the item in the current node
        current = self.children[i + 1]
        while not current.isleaf:
            current = current.children[0]
        return current.key[0]

    def predecessor(self, i):
        current = self.children[i]
        while not current.isleaf:
            current = current.children[-1]
        return current.key[-1]

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

    def num_items(self):
        if self.isleaf:
            return len(self.key)
        else:
            return sum(child.num_items() for child in self.children) + len(self.key)

    def height(self):
        if self.isleaf:
            return 0
        else:
            return max(child.height() for child in self.children) + 1


class BTree:
    def __init__(self, t):
        assert t >= 2, "t >= 2"
        self.root = BNode()
        self.t = BNode.t = t

    def __len__(self):
        return self.root.num_items()

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

        r, s = 0, 0
        node = self.root

        while not node.isleaf:
            r = node.rank(key)
            s += sum(c.numitems() for c in node.children[:r])
            node = node.children[r]

        r = node.rank(key)
        if r > 0:
            s += sum(c.numitems() for c in node.children[:r])

        return s

    def __repr__(self):
        return repr(self.root)

    def __contains__(self, item):
        return self.root.search(item) is not None

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

    @property
    def height(self):
        return self.root.height()

    def check_btree_properties(self):
        pass


if __name__ == '__main__':
    import numpy as np
    from random import shuffle, randint
    for _ in range(3):
        values = np.random.randint(0, 1000000, 10000)
        # values = [2, 32, 32, 41, 46, 50, 52, 69, 71, 71, 73, 74, 89, 93, 98]
        btree = BTree(t=2)
        for val in values:
            btree.insert(val)

        x = tuple(btree.inorder)
        print(btree.minimum, btree.maximum)
        print(np.min(values), np.max(values))
        assert(btree.minimum == min(values))
        assert(btree.maximum == max(values))
        assert (len(btree) == len(x))

        shuffle(values)
        for val in values:
            btree.delete(val)
        assert btree.isempty