from bisect import insort

LEFT, MIDDLE, RIGHT = 0, 1, 2
KEY, ITEM = 0, 1


class Node:
    def __init__(self, key, val, parent=None):
        self.parent = parent
        self.data = [(key, val)]
        self.children = []

    @property
    def isfull(self):
        return len(self.data) >= 3

    def __le__(self, other):
        return self.data[KEY] <= other.data[KEY]

    def __eq__(self, other):
        return self.data[KEY] == other.data[KEY]

    def __ge__(self, other):
        return self.data[KEY] >= other.data[KEY]

    def __lt__(self, other):
        return self.data[KEY] < other.data[KEY]

    def __gt__(self, other):
        return self.data[KEY] > other.data[KEY]

    @property
    def isleaf(self):
        return len(self.children) == 0

    def __repr__(self):
        return f"{self.__class__.__qualname__}({repr(self.data)})"

    @property
    def height(self):
        if self is None:
            return -1
        if self.isleaf:
            return 0
        else:
            return max(node.height for node in self.children) + 1

    @property
    def size(self):
        if self is None:
            return 0
        if self.isleaf:
            return len(self.data)
        else:
            return sum(node.size for node in self.children) + len(self.data)

    def is_balanced(self):
        if self.isleaf:
            return True
        else:
            if len(self.data) == 1:
                assert self.children[LEFT] <= self.data[0][0] <= self.children[MIDDLE]
            if len(self.data) == 2:
                assert self.data[0][0] >= self.children[LEFT] and \
                       self.data[0][0] >= self.children[MIDDLE] <= self.data[0][1] <= self.children[RIGHT]

    def height_is_equal(self):
        if self.isleaf:
            return 0
        h = self.children[0].height
        if len(self.children) == 1:
            return h + 1
        heights = [self.children[i].height for i in range(1, len(self.children))]
        assert all(h == hn for hn in heights)
        return h + 1

    def minimum(self):
        if self.isleaf:
            return self.data[LEFT]
        return self.children[LEFT].minimum()

    def maximum(self):
        if self.isleaf:
            return self.data[-1]
        else:
            return self.children[-1].maximum()

    def successor(self, key):
        x, j = None, None
        if len(self.children) == 0 or key > self.data[-1][0]:
            x = None
        else:
            for i, (k, _) in enumerate(self.data):
                if key <= k:
                    j = i + 1
                    break

        if j is not None:
            x = self.children[j].minimum()

        if x is None:
            # regular traversal
            current = self
            while current.parent is not None and current is current.parent.children[-1]:
                current = current.parent
            if current.parent is not None:
                x = current.parent.data[-1]

            # successor might be in the same node
            if j is not None and j + 1 < len(self.data):
                return self.data[j + 1]
        return x

    def find_in_children(self, data):
        for i, child in enumerate(self.children):
            if data in child.data:
                return i
        return None


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
                    return curr

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
                if key <= k:
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
        self.size += 1

    def split(self, node: Node):
        # when dealing with a leaf node
        p = node.parent
        # promote the middle node
        mid = node.data.pop(MIDDLE)
        if p is None:
            self.root = p = Node(*mid)
        else:
            p.children.remove(node)
            insort(p.data, mid)

        new_nodes = [Node(k, v, p) for k, v in node.data]
        p.children.extend(new_nodes)
        p.children.sort()
        for child in new_nodes:
            child.parent = p
        self.absorb(p)

    def successor(self, key):
        node = self.access(key)
        if node is None:
            return None
        else:
            return node.successor(key)

    @property
    def num_nodes(self):
        return self.root.size

    @property
    def height(self):
        return self.root.height

    @property
    def minimum(self):
        if self.root is None:
            return None
        return self.root.minimum()

    @property
    def maximum(self):
        if self.root is None:
            return None
        return self.root.maximum()

    def remove(self, key):
        if self.isempty:
            return False
        target = self.access(key)
        if target is None:
            raise KeyError(f"{target} not found.")

        # case 1: value is in an internal node
        if not target.isleaf:
            # replace by in-order successor
            pass

        self.__delete_fixup(target)

    def __delete_fixup(self, target):
        # case 1: terminal case absorb the middle node
        if len(target.data) == 2:
            x = target.data.children.pop(MIDDLE)
            self.merge(target, x)
        else:
            assert len(target.data) == 1
            p = target.parent
            if p is None:
                # base case
                self.merge(self.root, target)
            else:
                if len(p.children) == 2:
                    target_pos = p.data.find_in_children(target)
                    sibling = p.children[not target_pos]

                    # case 1: The hole has a 2-node as a parent and a 2-node as a sibling.
                    if len(sibling.children) == 2:
                        self.merge(p, target.children)
                        self.merge(p, sibling)
                    # case 2: The hole has a 2-node as a parent and a 3-node as a sibling.
                    elif len(sibling.children) == 3:
                        self.__rotate(p, target_pos)
                        # terminal case
                else:
                    assert len(p.children) == 3
                    # case 3 hole has a 3-node as a parent and a 2-node as a sibling. There are two subcases:
                    # TODO: Implement
                    pass

    @staticmethod
    def __rotate(y: Node, direction: int):
        if y is None:
            raise ValueError("can't rotate null value")

        # split sibling
        target = y.children[direction]
        sibling = y.children[not direction]
        yy = sibling.data.pop(direction)
        new_node = Node(*yy, y.parent)

        beta = sibling.children[LEFT] if direction == LEFT else sibling.children[-1]
        y.children = target.children
        y.children.extend(beta)
        y.children.sort()
        for child in y.children:
            child.parent = y
        new_node.children = [y, sibling]
        new_node.children.sort()
        for child in new_node.children:
            child.parent = new_node

    def absorb(self, p):
        # if the parent has three nodes. Check if the parent has three children
        while p is not None and p.isfull:
            mid = p.data.pop(MIDDLE)
            if p.parent is None:
                g = Node(*mid)
                self.root = g
            else:
                g = p.parent
                insort(g.data, mid)
                g.children.remove(p)

            left, right = (Node(k, v, g) for k, v in p.data)
            p.data.pop(MIDDLE)
            g.children.extend([right, left])
            g.children.sort()

            if len(p.children) == 4:
                x1, x2 = p.children[:2]
                y1, y2 = p.children[2:]

                x1.parent = x2.parent = left
                y1.parent = y2.parent = right

                left.children = [x1, x2]
                right.children = [y1, y2]

            p = g

    def merge(self, node, child):
        """Merge two nodes assuming that one is a parent and the other a child."""
        if child in node.children:
            node.children.remove(child)
        node.data.extend(child.data)
        node.children.extend(child.children)
        for c in child.children:
            c.parent = node
        node.data.sort()
        node.children.sort()
        self.absorb(node)

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
            if len(node.children) > 0:
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
    NUM = 10
    # values = [randint(0, LIM) for _ in range(NUM)]
    values = [7, 17, 18, 0, 10, 14, 17, 10, 12, 12, 7, 14, 8, 19, 9, 8, 4]
    print(values)
    print(len(values))
    st = Tree()
    st.insert(values)
    print(st.successor(12))
