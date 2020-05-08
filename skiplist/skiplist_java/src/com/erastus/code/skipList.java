package com.erastus.code;

import java.util.ArrayList;
import java.util.Random;

/**
 * Implementation of a skip list
 * @author erastusmurungi
 */


public class skipList<T extends Comparable<T>> {
    int count;   // the number of elements which have been added to the skip list so far
    int height;  // the height of the highest node
    static int MAXHEIGHT = 32;
    skipNode<T> header; // a pointer to the entry point. The value of header = Infinity
    T max = null;

    /**
     * constructor
     * @param infinity the largest possible value of the type the class is initialized with
     *                 if T = Integer => Integer.MAX_VALUE
     *                      = Double => Double.POSITIVE_INFINITY
     *                      = Float => Float.POSITIVE_INFINITY
     *                      = String => (char) 0xFFFF
     */

    public skipList(T infinity) {
        count = 0;
        header = new skipNode<T>(infinity, MAXHEIGHT);
        height = 1;
    }

    /**
     * Search for a single value in the skip list
     * @param value search for value in the skiplist.
     * @return the node containing value if it it found else return the header
     */
    public skipNode<T> findNode(T value) {

        skipNode<T> s = header;

        for (int level = height - 1; level >= 0; level--) {
            while (s.next[level] != null && s.next[level].getValue().compareTo(value) <= 0) {
                s = s.next[level];

                // if s.nxt > value, then s <= value. If s == value, no need to continue search
                if (s.value.equals(value))
                    return s;
            }
        }
        return s;
    }

    /**
     * @return the height of the list
     */

    public int getHeight(){
        return height;
    }

    /**
     * Deletes an element from the skip list if the element is in the skip list
     * @param value The value to be deleted
     * @return true if the value was in the skip list and was deleted else false
     */

    public boolean remove(T value) {

        skipNode<T> s = header;
        skipNode<T> target = s;

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
        if (value.equals(max)){
            max = calcMax();
        }
        // decrement number of objects in the list
        --count;
        return true;
    }

    public boolean contains(T x){
        skipNode<T> found = findNode(x);
        return found.value.equals(x);
    }

    public T nextLarger(T x){
        skipNode<T> p = findNextNode(x);
        return p == null ? header.value : p.value;
    }

    public T nextSmaller(T x){
        skipNode<T> p = findPredNode(x);
        return p == header ? null : p.value;
    }

    public skipNode<T> findPredNode(T value) {
        skipNode<T> s = header;
        int level = height - 1;
        for (; level >= 0; level--) {
            while (s.next[level] != null && s.next[level].getValue().compareTo(value) < 0)
                s = s.next[level];
        }
        return s;
    }

    public T dummyInfinity(){
        return header.value;
    }

    public skipNode<T> findNextNode(T value) {
        skipNode<T> s = findPredNode(value);
        while (s.next[0] != null && s.next[0].getValue().compareTo(value) <= 0){
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
        skipNode<T> s = header;
        int level;
        int h = pickHeight();
        height = Math.max(h, height);
        skipNode<T> elt = new skipNode<T>(value, h);

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

    private T calcMax(){
        skipNode<T> s = header;
        while (s.next[0] != null)
            s = s.next[0];
        return s.value;
    }

    public ArrayList<T> toVector(){
        ArrayList<T> v = new ArrayList<T>();
        skipNode<T> s = header.next[0];
        while (s != null){
            v.add(s.value);
            s = s.next[0];
        }
        return v;
    }

    public int length(){
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
    public void clear(){
        skipNode<T> []s = header.next;
        for (int i = 0; i < height; i++){
            s[i] = null;
        }
        count = 0;
        height = 1;
        max = null;
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
