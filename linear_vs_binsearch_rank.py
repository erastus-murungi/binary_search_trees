from numpy import random
from bisect import bisect_left
from time import perf_counter


def linear_rank(A, k):
    """return the number of elements less than or equal to k in O(n) time"""
    n = len(A)
    i = 0
    while i < n and k > A[i]:
        i += 1
    return i


if __name__ == "__main__":
    key = 2_500_000
    test = random.randint(0, 10000000, 3000000)
    test.sort()
    t1 = perf_counter()
    x = bisect_left(test, key)
    tdiff1 = perf_counter() - t1
    print(f"Bisect ran in {tdiff1} seconds.")

    t2 = perf_counter()
    y = linear_rank(test, key)

    tdiff2 = perf_counter() - t2
    print(f"Linear ran in {tdiff2} seconds. ")

    print(f"Bisect id faster than linear rank by a factor of x{tdiff2 / tdiff1 :.2f}")
    print(x, y)
