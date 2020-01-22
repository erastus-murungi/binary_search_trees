from bst import BST, BSTNode, LEFT, RIGHT
from random import randint, choice, seed
import numpy as np
from collections import Counter

# NUM_BITS = 64
# secret_key = hash(randint(0, 0xFFFFFFFFF))
# ROTATE_ROUNDS = choice([randint(0, NUM_BITS) for _ in range(NUM_BITS)])


# def left_rotate(b, n):
#     return (b >> n) | (b << (NUM_BITS - n))


class ZipNode(BSTNode):
    def __init__(self, key, item, rank, parent=None):
        super().__init__(key, item, parent)
        self.rank = rank


class ZipTree(BST):
    def __init__(self):
        super().__init__()

    @staticmethod
    def random_rank(key):
        # key = left_rotate(key, ROTATE_ROUNDS)
        # key ^= secret_key
        # seed(key)
        
        rank = 0
        y = np.random.random()
        while y < 0.5:
            rank += 1
            y = np.random.random()
        return rank

    def insert(self, key, item=0):
        # generate a random rank
        rank = self.random_rank(key)

        prev, curr, = None, self.root

        while curr is not None and (rank < curr.rank or (rank == curr.rank and key > curr.key)):
            prev = curr
            curr = curr.child[key < curr.key]

        z = ZipNode(key, item, rank)
        if curr == self.root:
            self.root = z
        else:
            prev.child[key < prev.key] = z

        if curr is None:
            return
        else:
            z.child[key < curr.key] = curr
        prev = z

        # ZIPPING LEFT AND RIGHT PARTS
        while curr is not None:
            fix = prev
            if key < curr.key:
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


if __name__ == '__main__':
    # keys = [randint(0, 0xFFFFFFF) for _ in range(100)]
    # ranks = list(map(lambda k: ZipTree.random_rank(k), keys))
    # print(Counter(ranks))

    nodes = [(randint(0, 15), randint(0, 10000)) for _ in range(5)]
    zp = ZipTree()
    for key, item in nodes:
        zp.insert(key, item)
