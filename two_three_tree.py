from bisect import insort
LEFT, MIDDLE, RIGHT = 0, 1, 2


class Node:
    def __init__(self, key, val, parent=None):
        self.parent = parent
        self.data = [(key, val)]
        self.children = []

    @property
    def isfull(self):
        return len(self.data) >= 3

    def overflowing(self):
        return len(self.children) >= 3

    @property
    def isleaf(self):
        return len(self.children) == 0

    def __repr__(self):
        return f"{self.__class__.__qualname__}({repr(self.data)})"

    def height(self):
        if self is None:
            return -1
        if self.isleaf:
            return 0
        else:
            return max(node.height() for node in self.children) + 1

    def size(self):
        if self is None:
            return 0
        if self.isleaf:
            return len(self.data)
        else:
            return sum(node.size() for node in self.children) + len(self.data)


class Tree:
    def __init__(self):
        self.root = None
        self.size = 0

    @property
    def isempty(self):
        return self.root is None

    def access(self, key):
        if self.isempty:
            raise KeyError(f"{repr(key)}")

        curr = self.root
        while True:
            for k, v in curr.data:
                if k == key:
                    return k, v

            if curr.isleaf:
                return None
            else:
                curr = self.trickle_down(curr, key)

    def __contains__(self, item):
        return self.access(item) is not None

    @staticmethod
    def trickle_down(curr, key):
        if key > curr.data[-1][0]:
            curr = curr.children[-1]
            # last item
        else:
            for i, (k, _) in enumerate(curr.data):
                if key < k:
                    curr = curr.children[i]
                    break
        return curr

    def insert(self, args):
        if len(args) == 2:
            self._insert(*args)
        else:
            for arg in args:
                self._insert(arg)

    def _insert(self, key, val=None):
        if self.isempty:
            self.root = Node(key, val)
            self.size += 1
            return

        curr = self.root
        while True:
            if curr.isleaf:
                break
            else:
                curr = self.trickle_down(curr, key)

        assert curr.isleaf, "Insertion should happen at a leaf node."
        insort(curr.data, (key, val))

        if curr.isfull:
            self.split(curr)

    def split(self, node: Node):

        # when dealing with a leaf node
        p = node.parent
        # promote the middle node
        mid = node.data.pop(MIDDLE)
        if p is None:
            # just create a new node
            self.root = Node(*mid)
            self.root.children = list(Node(k, v, self.root) for k, v in node.data)
            for child in self.root.children:
                child.parent = self.root
        else:
            insort(p.data, mid)

        # if the parent has three nodes. Check if the parent has three children
        if p is not None and p.isfull:
            mid = p.data.pop(MIDDLE)
            if p.parent is None:
                x = Node(*mid)
                self.root = x
            else:
                x = p.parent
                insort(x.data, mid)

            l, r = (Node(k, v, x) for k, v in p.data)
            l.children, r.children = [p.data.children[0]], [p.data.children[1]]
            l.children[0].parent = l
            r.children[0].parent = r
            x.children.extend([l, r])
            x.children.sort()

            left, right = (Node(k, v, x) for k, v in p.data)

            x.children = [right, left]

            assert len(p.children) == 4
            x1, x2 = p.children[:2]
            y1, y2 = p.children[2:]

            x1.parent = x2.parent = left
            y1.parent = y2.parent = right

            left.children = [x1, x2]
            right.children = [y1, y2]

            node = p
            p = x

    def remove(self, key):
        pass

    def __repr__(self):
        return repr(self.root)

    def __str__(self):
        if self.root is None:
            return str(None)
        else:
            self.__print_helper(self.root, "", True)
            return ''

    def __print_helper(self, node, indent, last, pos="R"):
        """Simple recursive tree printer"""
        if node is not None:
            print(indent, end='')
            if last:
                print("R----", end='')
                indent += "     "
            else:
                print(f"{pos}----", end='')
                indent += "|    "
            print(*node.data)
            if node.children:
                self.__print_helper(node.children[LEFT], indent, False, "L")
                if len(node.children) == 2:
                    self.__print_helper(node.children[MIDDLE], indent, True)
                else:
                    self.__print_helper(node.children[MIDDLE], indent, False, "M")
                    self.__print_helper(node.children[RIGHT], indent, True)

    def __len__(self):
        return self.size


if __name__ == "__main__":
    from random import randint

    LIM = 20
    NUM = 20
    values = [randint(0, LIM) for _ in range(NUM)]
    st = Tree()
    st.insert(values)
    print(st)
