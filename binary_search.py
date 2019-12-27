from random import randint
from datetime import datetime

def binary_search_recursive(low, high, key, A):
    """Slower"""
    if low > high:
        return False
    mid = (low + high) >> 1
    if A[mid] == key:
        return True
    elif A[mid] < key:
        return binary_search_recursive(mid + 1, high, key, A)
    else:
        return binary_search_recursive(low, mid - 1, key, A)

def binary_search_iterative(low, high, key, A):
    """Faster"""
    while low <= high:
        mid = (low + high) >> 1
        if A[mid] == key:
            return True
        elif A[mid] < key:
            low = mid + 1
        else:
            high = mid - 1
    return False

def brute_search(A, key):
    """Slowest"""
    for elem in A:
        if elem == key:
            return True
    else:
        return False


def test_binary_search(n):
    while True:
        A = [randint(0, n) for _ in range(n)]
        A.sort()
        searches = [randint(0, int(n * 1.5)) for _ in range(n << 1)]
        # tic = datetime.now()
        br = []
        for x in searches:
            br.append(binary_search_recursive(0, n - 1, x, A))
        # toc = datetime.now()
        # print("Recursive binary search ran in", (toc - tic).total_seconds(), "seconds for", int(n * 1.5), "searches")
        # tic1 = datetime.now()
        bi = []
        for x in searches:
            bi.append(binary_search_iterative(0, n-1, x, A))
        # toc1 = datetime.now()
        # print("Iterative binary search ran in", (toc1 - tic1).total_seconds(), "seconds for", int(n * 1.5), "searches")
        if bi != br:
            print(searches, bi, br, A)
            break


if __name__ == '__main__':
    test_binary_search(10000)
