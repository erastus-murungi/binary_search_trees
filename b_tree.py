from bisect import bisect, bisect_left, insort


class BNode:
    t = 2

    def __init__(self, val=None):
        if val is None:
            self.key = []
        else:
            self.key = [val]
        self.children = []
        self.size = len(self.key)

    @property
    def isleaf(self):
        return len(self.children) == 0

    def pos(self, k) -> int:
        """Get the rank of an element quickly in O(lg n)"""
        return bisect_left(self.key, k)

    def rank(self, k):
        return bisect(self.key, k)

    def linear_rank(self, k):
        """return the number of elements less than or equal to k in O(n) time"""
        n = len(self.children)
        i = 0
        while i < n and k > self.children[i]:
            i += 1
        return i

    def search(self, k):
        i, n = self.pos(k), len(self.children)
        if i < n and k == self.key[i]:
            return self, i
        if self.isleaf:
            return None
        else:
            return self.children[i].search(k)

    def isfull(self):
        return len(self.key) == (self.t << 1) - 1

    def insert_nonfull(self, k):
        if self.isleaf:
            insort(self.key, k)
        else:
            poschild = bisect(self.key, k)
            child = self.children[poschild]
            if child.isfull:
                self.split(poschild)
                if k > self.key[poschild]:
                    child = self.children[poschild + 1]
            child.insert_nonfull(k)

    def split(self, i):
        """splits a child node. Assuming the child node has 2t - 1 keys"""

        t = self.t
        z = BNode()
        y = self.children[i]

        z.key.extend(y.key[t:])
        if not y.isleaf:
            z.children.extend(y.children[t:])
            y.children = y.children[:t]
        y.key = y.key[:t]

        self.children.insert(i + 1, z)
        self.key.insert(i, y.key[t - 1])

    def __repr__(self):
        return repr(self.key)

    def __str__(self):
        return self.__repr__()

    @property
    def num_items(self):
        if self.isleaf:
            return len(self.key)
        else:
            return sum(child.num_items for child in self.children) + len(self.key)


class BTree:
    def __init__(self, t):
        assert t >= 2, "t >= 2"
        self.root = BNode()
        BNode.t = t

    def __len__(self):
        return self.root.num_items

    def insert(self, k):
        r = self.root
        if r.isfull():
            s = BNode()
            self.root = s
            s.children.append(r)
            s.split(0)
            s.insert_nonfull(k)
        else:
            r.insert_nonfull(k)

    def __repr__(self):
        return repr(self.root)


if __name__ == '__main__':
    from random import randint
    values = [randint(0, 100) for _ in range(10)]
    btree = BTree(2)
    for val in values:
        btree.insert(val)
    print(repr(btree))
