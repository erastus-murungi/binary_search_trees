from bisect import insort, bisect


class BNode:
    def __init__(self, val=None):
        if val is None:
            self.key = []
        else:
            self.key = [val]
        self.children = []

    def insert(self):
        pass

    @property
    def isleaf(self):
        return len(self.children) == 0

    def rank(self, k) -> int:
        """Get the rank of an element quickly in O(lg n)"""
        low, high, m = 0, len(self.children), 0
        if k >= self.children[-1]:
            return high
        while low <= high:
            mid = (low + high) >> 1
            if self.children[mid + 1] > k >= self.children[mid]:
                return mid + 1
            elif self.children[mid] < k:
                low = mid + 1
            else:
                high = mid - 1
        return 0

    def linear_rank(self, k):
        """return the number of elements less than or equal to k in O(n) time"""
        n = len(self.children)
        i = 0
        while i < n and k > self.children[i]:
            i += 1
        return i + 1

    def search(self, k):
        i, n = self.rank(k), len(self.children)
        if i < n and k == self.key[i]:
            return self, i
        if self.isleaf:
            return None
        else:
            return self.children[i].search(k)

    def split(self, i, t):
        """splits a child node. Assuming the child node has 2t - 1 keys"""
        z = BNode()
        y = self.children[i]
        assert len(y.data) == (t << 1) - 1
        for j in range(t - 1):
            z.key[j] = y.key[j + t]
        if not y.isleaf:
            for j in range(t):
                z.children[j] = y.children[j + t]

        pos = bisect(self.key, y.key[t])
        self.children.insert(pos, z)
        self.key.insert(pos, y.key[t])


class BTree:
    def __init__(self, t):
        assert t >= 2, "t >= 2"
        self.root = None
        self._t = t



if __name__ == '__main__':
    bnode = BNode()
    bnode.children.extend(list(range(10)))
    bnode.children.extend([23, 67, 100])
    # x = bnode.linear_rank(100)
    x = bisect(bnode.children, 0)
    print(x)
    print(bnode.children)
