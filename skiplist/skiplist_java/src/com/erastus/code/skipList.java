package com.erastus.code;

import org.jetbrains.annotations.NotNull;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;

/**
 * Implementation of a skip list
 *
 * @author erastusmurungi
 */


public class SkipList<T extends Comparable<T>> implements Iterable<T> {
    int count;   // the number of elements which have been added to the skip list so far
    int height;  // the height of the highest node
    static int MAXHEIGHT = 32;
    Node<T> header; // a pointer to the entry point. The value of header = Infinity
    T max = null;

    /**
     * constructor
     *
     * @param bound the largest possible value of the type the class is initialized with
     *                 if T = Integer => Integer.MAX_VALUE
     *                 = Double => Double.POSITIVE_INFINITY
     *                 = Float => Float.POSITIVE_INFINITY
     *                 = String => (char) 0xFFFF
     */

    public SkipList(T bound) {
        count = 0;
        header = new Node<T>(bound, MAXHEIGHT);
        height = 1;
    }

    /**
     * Search for a single value in the skip list
     *
     * @param value search for value in the skiplist.
     * @return the node containing value if it it found else return the header
     */
    public Node<T> findNode(T value) {

        Node<T> s = header;

        for (int level = height - 1; level >= 0; level--) {
            while (s.next[level] != null && s.next[level].getValue().compareTo(value) <= 0) {
                s = s.next[level];
            }
        }
        return s;
    }

    /**
     * @return the height of the list
     */

    public int getHeight() {
        return height;
    }

    /**
     * Deletes an element from the skip list if the element is in the skip list
     *
     * @param value The value to be deleted
     * @return true if the value was in the skip list and was deleted else false
     */

    public boolean remove(T value) {

        Node<T> s = header;
        Node<T> target = s;

        // find the leftmost instance of the value
        for (int level = height - 1; level >= 0; level--) {
            while (target.next[level] != null && target.next[level].getValue().compareTo(value) < 0) {
                target = target.next[level];
            }
        }

        // so far we have the predecessor since we were traversing the skip list using the < operator
        // we take one more leap to get to the node contain the element we were looking for
        target = target.next[0];

        // element is not in the list
        if (target == null || !target.value.equals(value))
            return false;

        // traverse the skip list again from the top. This time, whenever we find a pointer to the target node,
        // we replace it with next  s.next[level]-> target-> target.next[level] => s.next[level] = target.next[level]

        for (int level = height - 1; level >= 0; level--) {
            while (s.next[level] != null && s.next[level].getValue().compareTo(value) < 0) {
                s = s.next[level];
            }
            if (s.next[level] == target) {
                s.next[level] = target.next[level];
            }
        }
        // update the maximum value manually if the value we have deleted was the maximum
        if (value.equals(max)) {
            max = calcMax();
        }
        // decrement number of objects in the list
        --count;
        return true;
    }

    public boolean contains(T value) {
        Node<T> s = header;

        for (int level = height - 1; level >= 0; level--) {
            while (s.next[level] != null && s.next[level].getValue().compareTo(value) <= 0) {
                s = s.next[level];

                // if s.nxt > value, then s <= value. If s == value, no need to continue search
                if (s.value.equals(value))
                    return true;
            }
        }
        return false;
    }

    /**
     * @param value
     * @return largest value <= value
     */

    public T floor(T value) {
        Node<T> s = header;
        for (int level = height - 1; level >= 0; level--) {
            while (s.next[level] != null && s.next[level].value.compareTo(value) < 0) {
                s = s.next[level];
            }
        }
        if (s.next[0] == null || s.next[0].value.compareTo(value) > 0) {
            if (s.value.equals(header.value)) {
                return null;
            }
            return s.value;
        }
        else {
            return s.next[0].value;
        }
    }

    /**
     * @param value operand
     * @return the smallest value in the skiplist >= value
     */

    public T ceiling(T value) {
        Node<T> s = header;
        for (int level = height - 1; level >= 0; level--) {
            while (s.next[level] != null && s.next[level].value.compareTo(value) <= 0) {
                s = s.next[level];
            }
        }
        if (s == header || s.value.compareTo(value) < 0) {
            if (s.next[0] == null) {
                return null;
            } else return s.next[0].value;
        }
        else return s.value;
    }

    public T nextLarger(T value) {
        Node<T> p = findNextNode(value);
        return p == null ? header.value : p.value;
    }

    public T nextSmaller(T value) {
        Node<T> p = findPredNode(value);
        return p == header ? null : p.value;
    }

    public Node<T> findPredNode(T value) {
        Node<T> s = header;
        int level = height - 1;
        for (; level >= 0; level--) {
            while (s.next[level] != null && s.next[level].getValue().compareTo(value) < 0)
                s = s.next[level];
        }
        return s;
    }

    public T getBound() {
        return header.value;
    }

    public Node<T> findNextNode(T value) {
        Node<T> s = findPredNode(value);
        while (s.next[0] != null && s.next[0].getValue().compareTo(value) <= 0) {
            s = s.next[0];
        }
        return s.next[0];
    }

    public int pickHeight() {
        Random rand = new Random();
        int k = 1;
        while (((1 & rand.nextInt()) == 0) && (k < MAXHEIGHT)) {
            ++k;
        }
        return k;
    }

    public void insert(T value) {
        Node<T> s = header;
        int level;
        int h = pickHeight();
        height = Math.max(h, height);
        Node<T> elt = new Node<T>(value, h);

        // first traverse the upper levels to get to the elements designated height
        for (level = height - 1; level >= h; level--) {
            while (s.next[level] != null && s.next[level].getValue().compareTo(value) < 0)
                s = s.next[level];
        }
        // start insertion by splicing
        for (; level >= 0; level--) {
            while (s.next[level] != null && s.next[level].getValue().compareTo(value) < 0) {
                s = s.next[level];
            }
            elt.next[level] = s.next[level];
            s.next[level] = elt;
        }
        max = max == null || max.compareTo(value) < 0 ? value : max;
        ++count;
    }

    private T calcMax() {
        // a search for ∞ returns the node with the largest value
        return findNode(header.value).value;
    }

    private void updateMax() {
        max = calcMax();
    }

    private void updateCount() {
        count = 0;
        Node<T> s = header.next[0];
        while (s != null) {
            count++;
            s = s.next[0];
        }

    }

    public ArrayList<T> toVector() {
        ArrayList<T> v = new ArrayList<T>();
        Node<T> s = header.next[0];
        while (s != null) {
            v.add(s.value);
            s = s.next[0];
        }
        return v;
    }

    public int length() {
        return count;
    }

    public boolean isEmpty() {
        return count == 0;
    }

    public T getMax() {
        return max;
    }

    public T getMin() {
        return (isEmpty()) ? null : header.next[0].value;
    }

    /**
     * remove all the elements from the skip list
     */
    public void clear() {
        Node<T>[] s = header.next;
        for (int level = 0; level < height; level++) {
            s[level] = null;
        }
        count = 0;
        height = 1;
        max = null;
    }

    /**
     * removes all the keys in this list with key >= splitKey and return them in a new list
     * the two lists have rhe same height
     *
     * @param splitKey the value with which to split the skip list
     * @return a new skiplist
     */

    public SkipList<T> split(T splitKey) {
        SkipList<T> newList = new SkipList<>(header.value);
        newList.height = height;
        Node<T> s = header;

        // walk down the list
        for (int level = height - 1; level >= 0; level--) {
            while (s.next[level] != null && s.next[level].value.compareTo(splitKey) < 0) {
                s = s.next[level];
            }
            // s.next[level] >= split key
            newList.header.next[level] = s.next[level];
            // terminate the first list
            s.next[level] = null;
        }
        // trim the heights to the appropriate levels
        // go down the tower until header.next[level] has an element
        while (this.height > 1 && this.header.next[height - 1] == null) {
            height--;
        }
        while (newList.height > 1 && newList.header.next[height - 1] == null) {
            newList.height--;
        }
        // update max and count
        newList.updateCount();
        newList.updateMax();
        updateMax();
        count -= newList.count;
        return newList;
    }

    /**
     * Appends list2 to the end of list1, assumes last key in list1 ≤ first key in list2
     *
     * @param other skip list
     *              assumes the max heights of the two lists are thr same
     */
    public void concatenate(SkipList<T> other) {
        if (getMax().compareTo(other.getMin()) > 0) {
            return;
        }

        height = Math.max(height, other.height);

        int level;
        Node<T> s = header;
        for (level = height - 1; level >= 0; level--) {
            while (s.next[level] != null) {
                s = s.next[level];
            }
            // s.next[level] == null
            if (level <= other.height) {
                s.next[level] = other.header.next[level];
            }
        }
        count += other.count;
        max = other.getMax();
    }

    /**
     * working is similar to that of merge in mergesort
     *
     * @param other another skip list
     */

    @SuppressWarnings("unchecked")
    public SkipList<T> merge(@NotNull SkipList<T> other, boolean allowDuplicates) {
        Node<T>[] frontier = (Node<T>[]) Array.newInstance(Node.class, MAXHEIGHT);
        boolean unFlipped = true;
        int level, i;
        T key1, key2;
        SkipList<T> self = this;

        SkipList<T> list = new SkipList<>(header.value);
        list.height = Math.max(self.height, other.height);

        for (level = 0; level < list.height; level++) {
            frontier[level] = list.header;
        }

        while (self.header.next[0] != null && other.header.next[0] != null) {
            key1 = self.header.next[0].value;
            key2 = other.header.next[0].value;

            // let key1 <= key2 at all times, else swap the skip lists
            if (key1.compareTo(key2) > 0) {
                unFlipped = !unFlipped;
                key2 = key1;
                SkipList<T> listT = self;
                self = other;
                other = listT;
            }

            // remove from list1 all the elements with values <= key2 and put them into list2
            level = 0;
            do {
                frontier[level].next[level] = self.header.next[level];
                level++;
            } while (level <= list.height && self.header.next[level] != null && self.header.next[level].value.compareTo(key2) <= 0);

            Node<T> s = self.header;
            for (i = level - 1; i >= 0; i--) {
                while (s.next[i] != null && s.next[i].value.compareTo(key2) <= 0) {
                    s = s.next[i];
                }
                frontier[i] = s;
                self.header.next[i] = s.next[i];
            }
            // duplicate removal
            if (!allowDuplicates) {
                if (key2.compareTo(s.value) == 0) {
                    if (unFlipped) {
                        s.value = other.header.next[0].value;
                    }
                    Node<T> y = other.header.next[0];
                    for (i = 0; i < y.next.length; i++) {
                        other.header.next[i] = y.next[i];
                    }
                }
            }
        }

        SkipList<T> leftOver = (other.header.next[0] == null) ? self : other;

        int n = 0;
        Node<T> s = leftOver.header;

        while (s.next[n] != null) {
            ++n;
        }
        for (i = 0; i < n; i++) {
            frontier[i].next[i] = leftOver.header.next[i];
        }
        for (i = n; i < list.height; i++) {
            frontier[i].next[i] = null;
        }
        while (list.height > 0 && list.header.next[list.height - 1] == null) {
            list.height--;
        }

        list.updateMax();
        list.updateCount();
        return list;
    }

    @Override
    public @NotNull Iterator<T> iterator() {
        return new Iterator<>() {

            private int currentIndex = 0;
            private final int currentSize = length();
            Node<T> current = header;


            @Override
            public boolean hasNext() {
                return currentIndex < currentSize;
            }

            @Override
            public T next() {
                current = current.next[0];
                currentIndex++;
                return current.value;
            }

            @Override
            public void remove() {
                throw new UnsupportedOperationException();
            }
        };
    }

    @Override
    public String toString() {
        return "skipList{" +
                "elements=" + toVector().toString() +
                ", " +
                "size=" + count +
                '}';
    }
}
