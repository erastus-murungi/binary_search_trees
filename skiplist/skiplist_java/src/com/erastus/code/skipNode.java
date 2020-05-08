package com.erastus.code;

import org.jetbrains.annotations.NotNull;

import java.lang.reflect.Array;
import java.util.Arrays;

/**
 * definition of a node in a skip list
 * @param <T> a comparable type such as an Integer, Double, String, Float
 */
public class skipNode<T extends Comparable<T>> {
    T value;
    skipNode<T>[] next;

    skipNode(T x) { value = x; }

    @SuppressWarnings("unchecked")
    skipNode(T x, int height) {
        value = x;
        next = (skipNode<T>[]) Array.newInstance(skipNode.class, height);
    }

    public T getValue() {
        return value;
    }

    int compareTo(@NotNull skipNode<T> other){
        return this.value.compareTo(other.value);
    }

    @Override
    public String toString() {
        return "skipNode{" +
                "value=" + value +
                ", next=" + Arrays.toString(next) +
                '}';
    }
}
