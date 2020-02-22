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
        n = len(self.key)
        i = 0
        while i < n and k > self.key[i]:
            i += 1
        return i

    def search(self, k):
        i, n = self.pos(k), len(self.key)
        if i < n and k == self.key[i]:
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

    def delete(self, key):
        pair = self.search(key)
        if pair is None:
            print("Key not found!")
            return False
        else:
            node, pos = pair
            if node.isleaf:
                node.key.pop(pos)
            else:
                node.delete_from_internal(pos)

    def delete_from_internal(self, i):
        """Assumes that self is an internal node from which we are deleting the key at index i."""
        assert not self.isleaf, "This function deletes from internal nodes."

        k = self.key[i]
        if len(self.children[i].key) >= self.t:
            pred = self.predecessor(i)
            self.key[i] = pred
            self.children[i].delete(pred)

        elif len(self.children[i + 1].key) >= self.t:
            succ = self.successor(i)
            self.key[i] = succ
            self.children[i + 1].delete(succ)

        else:
            # Both the left and right kids have less than < t children. So merge both the predecessor and successor.
            self.merge(i)
            self.children[i].delete(k)
        return True

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
        BNode.t = t

    def __len__(self):
        return self.root.num_items()

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

    def __repr__(self):
        return repr(self.root)

    def __contains__(self, item):
        return self.root.search(item) is not None

    @property
    def inorder(self):
        return self.root.inorder()

    @property
    def minimum(self):
        if len(self.root.key) == 0:
            return None
        else:
            return self.root.minimum()

    @property
    def maximum(self):
        if len(self.root.key) == 0:
            return None
        else:
            return self.root.maximum()

    @property
    def height(self):
        return self.root.height()


if __name__ == '__main__':
    # from numpy import random

    # values = random.randint(0, 100, 15)
    values = (2, 32, 32, 41, 46, 50, 52, 69, 71, 71, 73, 74, 89, 93, 98)
    btree = BTree(2)
    for val in values:
        btree.insert(val)

    x = tuple(btree.inorder)
    print(x)
    print(btree.minimum, min(values))
    print(btree.maximum, max(values))
    print(len(btree), len(x))
    btree.delete(69)
    print(tuple(btree.inorder))

