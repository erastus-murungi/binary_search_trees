package com.erastus.code;

import org.jetbrains.annotations.NotNull;

import java.lang.reflect.Array;
import java.util.Arrays;

/**
 * definition of a node in a skip list
 * @param <T> a comparable type such as an Integer, Double, String, Float
 */
public class Node<T extends Comparable<T>> {
    T value;
    Node<T>[] next;

    Node(T x) { value = x; }

    @SuppressWarnings("unchecked")
    Node(T x, int height) {
        value = x;
        next = (Node<T>[]) Array.newInstance(Node.class, height);
    }

    public T getValue() {
        return value;
    }

    int compareTo(@NotNull Node<T> other){
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
