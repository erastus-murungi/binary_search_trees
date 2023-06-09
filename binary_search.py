def binary_search_recursive(low, high, key, A):
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
    for elem in A:
        if elem == key:
            return True
    else:
        return False
