from bst import BST, LEFT, RIGHT
from random import randint
import numpy as np
from collections import Counter


class ZipNode:
    def __init__(self, key, item, rank):
        self.key = key
        self.item = item
        self.rank = rank
        self.child = [None, None]

    def __repr__(self):
        return f"Node(({self.key}, {self.item}), children: {repr(self.child)}])"


class ZipTree(BST):
    """duplicate keys not allowed"""
    def __init__(self):
        super().__init__()

    @staticmethod
    def random_rank():
        rank = 0
        y = np.random.random()
        while y > 0.5:
            rank += 1
            y = np.random.random()
        return rank

    def insert(self, key, item=0):
        # generate a random rank
        assert key not in self
        rank = self.random_rank()
        z = ZipNode(key, item, rank)
        self.size += 1

        prev, curr, fix = None, self.root, None

        while curr is not None and (rank < curr.rank or (rank == curr.rank and key > curr.key)):
            prev = curr
            curr = curr.child[key >= curr.key]

        if curr == self.root:
            self.root = z
        else:
            prev.child[key >= prev.key] = z

        if curr is None:
            z.child[RIGHT] = z.child[LEFT] = None
            return
        z.child[key < curr.key] = curr
        prev = z

        # ZIPPING LEFT AND RIGHT PARTS
        while curr is not None:
            fix = prev
            if curr.key < key:
                while curr is not None and curr.key <= key:
                    prev = curr
                    curr = curr.child[RIGHT]
            else:
                while curr is not None and curr.key >= key:
                    prev = curr
                    curr = curr.child[LEFT]
            if fix.key > key or (fix == z and prev.key > key):
                fix.child[LEFT] = curr
            else:
                fix.child[RIGHT] = curr

    def delete(self, key):
        if self.root is None:
            raise KeyError("Empty Tree.")
        curr, prev = self.root, None

        while key != curr.key:
            prev = curr
            curr = curr.child[key >= curr.key]

        left, right = curr.child[LEFT], curr.child[RIGHT]
        if left is None:
            curr = right
        elif right is None:
            curr = left
        elif left.rank >= right.rank:
            curr = left
        else:
            curr = right

        if self.root == key:
            self.root = curr
        else:
            prev.child[key >= prev.key] = curr

        while left is not None and right is not None:
            if left.rank >= right.rank:
                while left is not None and left.rank >= right.rank:
                    prev = left
                    left = left.child[RIGHT]
                prev.child[RIGHT] = right
            else:
                while right is not None and left.rank < right.rank:
                    prev = right
                    right = right.child[LEFT]
                prev.child[LEFT] = left
        self.size -= 1


if __name__ == '__main__':
    from datetime import datetime
    from pympler import asizeof

    # check that rank function produces geometric distribution
    ranks = [ZipTree.random_rank() for _ in range(100)]
    # print(Counter(ranks))
    while True:
        num_nodes = 100000
        nodes = list(set([(randint(0, 1000000), None) for _ in range(num_nodes)]))
        # nodes = [(77, None), (48, None), (56, None)]
        print(len(nodes))

        t1 = datetime.now()
        zp = ZipTree()
        for key, item in nodes:
            zp.insert(key, item)
        print(f"Zip tree used {asizeof.asizeof(zp) / (1 << 20):.2f} MB of memory and ran in"
              f" {(datetime.now() - t1).total_seconds()} seconds for {num_nodes} insertions.")

        print(zp.height())
        # print(95858277 in zp)
        # print(zp)
        # print(zp.check_weak_search_property())

        for key, _ in nodes:
            assert(key in zp), (nodes, key)
        print(len(zp))
