package com.erastus.testing;

import com.erastus.code.skipList;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;


class skipListTest {
    @org.junit.jupiter.api.Test

    public void createSkipListTest() throws Exception {
        skipList<Integer> stInteger = new skipList<Integer>(Integer.MAX_VALUE);
        skipList<String> stString = new skipList<String>(Character.toString(0xFFFF));
        skipList<Double> stDouble = new skipList<Double>(Double.POSITIVE_INFINITY);
        skipList<Float> stFloat = new skipList<Float>(Float.POSITIVE_INFINITY);

        stString.insert("Erastus");
        stString.insert("Murungi");
        stString.insert("Murithi");
        assertEquals("Murungi", stString.getMax());
        assertEquals("Erastus", stString.getMin());
    }

    @org.junit.jupiter.api.Test
    public void containsTest() throws Exception {
        skipList<Integer> stInt = new skipList<Integer>(Integer.MAX_VALUE);
        Integer[] expected1 = {9, 10, 20, 30, 42, 48, 61, 64, 74, 83, 98};

        Integer[] array = new Integer[11];
        Arrays.setAll(array, i -> i);
        List<Integer> shuffledIndices = Arrays.asList(array);
        Collections.shuffle(shuffledIndices);

        skipList<String> stString = new skipList<String>(null);
        for (Integer n : expected1)
            stInt.insert(n);

        for (Integer index : shuffledIndices) {
            assertTrue(stInt.contains(expected1[index]));
        }
        assertFalse(stInt.contains(100));
        stInt.clear();
        assertTrue(stInt.isEmpty());

        int streamSize = 10000;
        Set<Integer> randIntegers= new HashSet<>();

        int r;
        for (int i = 0; i < streamSize; i++) {
            r = (int) (5 + Math.random() * 20);
            randIntegers.add(r);
            stInt.insert(r);
        }
        for (int i = 0; i < streamSize; i++) {
            r = (int) (5 + Math.random() * 30);
            assertEquals(randIntegers.contains(r), stInt.contains(r));
        }
    }

    @org.junit.jupiter.api.Test
    void insertTest() throws Exception {
        /* test strategy
         * add random values, delete them in arbitrary order
         * check if the list is empty
         */
        int N = 100;
        Random rand = new Random();
        Set<Integer> values = new HashSet<>();
        skipList<Integer> stInt = new skipList<Integer>(Integer.MAX_VALUE);
        for (int i = 0; i < N; i++) {
            values.add(rand.nextInt(10000));
        }
        for (Integer value: values){
            stInt.insert(value);
        }
        for (Integer value: values){
            stInt.remove(value);
            assertFalse(stInt.contains(value));
        }
    }


    @org.junit.jupiter.api.Test
    void nextLargerTest() {
        skipList<Integer> stInt = new skipList<>(Integer.MAX_VALUE);
        Integer[] expected1 = {9, 10, 20, 30, 42, 48, 61, 64, 74, 83, 98};

        for (Integer n : expected1)
            stInt.insert(n);

        assertEquals(98, stInt.nextLarger(83));
        assertEquals(stInt.getBound(), stInt.nextLarger(98));
        assertEquals(83, stInt.nextLarger(74));
        assertEquals(61, stInt.nextLarger(48));
        assertEquals(48, stInt.nextLarger(42));
    }

    @org.junit.jupiter.api.Test
    void nextSmallerTest() {
        skipList<Integer> stInt = new skipList<>(Integer.MAX_VALUE);
        Integer[] expected1 = {9, 10, 20, 30, 42, 48, 61, 64, 74, 83, 98};

        for (Integer n : expected1)
            stInt.insert(n);

        assertEquals(83, stInt.nextSmaller(98));
        assertEquals(98, stInt.nextSmaller(stInt.getBound()));
        assertEquals(74, stInt.nextSmaller(83));
        assertEquals(48, stInt.nextSmaller(61));
        assertEquals(42, stInt.nextSmaller(48));
        assertNull(stInt.nextSmaller(9));

    }


    @org.junit.jupiter.api.Test
    void removeTest() throws Exception {
        /* similar values */
        Double[] expected = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        skipList<Double> stDouble = new skipList<Double>(Double.MAX_VALUE);
        for (Double d : expected) {
            stDouble.insert(d);
        }
        for (Double d : expected) {
            assertTrue(stDouble.contains(d));
        }
        int i;
        for (i = 0; i < 9; i++) {
            assertTrue(stDouble.contains(expected[i]));
            stDouble.remove(expected[i]);
        }
        assertEquals(1, stDouble.length());
        stDouble.remove(expected[i]);
        assertTrue(stDouble.isEmpty());

        Random rand = new Random();

        /*random values */

        int N = 10;
        double[] randDoubles = rand.doubles(N, 10.0, 200.0).toArray();
        for (Double d : randDoubles)
            stDouble.insert(d);

        for (i = 0; i < N - 1; i++) {
            assertTrue(stDouble.contains(randDoubles[i]));
            stDouble.remove(randDoubles[i]);
        }
        assertEquals(1, stDouble.length());
        stDouble.remove(randDoubles[i]);
        assertTrue(stDouble.isEmpty());
    }

    @org.junit.jupiter.api.Test
    void getMaxTest() {
        Double[] values = {9.7, 10.8, 0.0, 30.6, 42.1, 48.5, 61.66};
        skipList<Double> stDouble = new skipList<>(Double.POSITIVE_INFINITY);
        assertNull(stDouble.getMax());
        stDouble.insert(Math.random());
        assertEquals(stDouble.getMax(), stDouble.getMin());
        stDouble.insert(Math.random());
        stDouble.clear();

        for (Double val : values) {
            stDouble.insert(val);
        }
        assertEquals(61.66, stDouble.getMax());
        stDouble.remove(61.66);
        assertEquals(48.5, stDouble.getMax());
        stDouble.remove(48.5);
        assertEquals(42.1, stDouble.getMax());


    }

    @org.junit.jupiter.api.Test
    void getMinTest() {
        Double[] values = {9.7, 10.8, 5.0, 30.6, 42.1, 48.5, 61.66};
        skipList<Double> stDouble = new skipList<>(Double.POSITIVE_INFINITY);
        assertNull(stDouble.getMin());
        stDouble.insert(Math.random());
        assertEquals(stDouble.getMax(), stDouble.getMin());
        stDouble.clear();

        for (Double val : values) {
            stDouble.insert(val);
        }
        assertEquals(5.0, stDouble.getMin());
    }

    @org.junit.jupiter.api.Test
    void toVectorTest() throws Exception {
        Random rand = new Random();
        skipList<Integer> stInt = new skipList<>(Integer.MAX_VALUE);

        int streamSize = 1000;
        Integer[] randIntegers = new Integer[streamSize];

        for (int i = 0; i < streamSize; i++) {
            randIntegers[i] = rand.nextInt();
            stInt.insert(randIntegers[i]);
        }

        ArrayList<Integer> sortedInts = stInt.toVector();

        ArrayList<Integer> expected = new ArrayList<>(Arrays.asList(randIntegers));
        Collections.sort(expected);

        assertEquals(expected, sortedInts);

        // test with doubles

        skipList<Double> stDouble = new skipList<>(Double.POSITIVE_INFINITY);
        int N = 1000;
        Double[] randDoubles = new Double[N];
        for (int i = 0; i < streamSize; i++) {
            randDoubles[i] = Math.random();
            stDouble.insert(randDoubles[i]);
        }
        ArrayList<Double> sortedDoubles = stDouble.toVector();

        ArrayList<Double> expectedDoubles = new ArrayList<>(Arrays.asList(randDoubles));
        Collections.sort(expectedDoubles);

        assertEquals(expectedDoubles, sortedDoubles);
    }


    @org.junit.jupiter.api.Test
    void splitTest() throws Exception {
        Double[] values = {9.7, 10.8, 0.0, 30.6, 42.1, 48.5, 61.66};
        skipList<Double> stDouble = new skipList<>(Double.POSITIVE_INFINITY);

        for (Double val: values){
            stDouble.insert(val);
        }

        skipList<Double> stLarge = stDouble.split(40.0);
        Double[] smaller = {9.7, 10.8, 0.0, 30.6};
        Double[] larger = {42.1, 48.5, 61.66};

        for (Double small: smaller){
            assertTrue(stDouble.contains(small));
        }

        for (Double large: larger) {
            assertFalse(stDouble.contains(large));
        }

        for (Double small: smaller){
            assertFalse(stLarge.contains(small));
        }

        for (Double large: larger) {
            assertTrue(stLarge.contains(large));
        }

        assertEquals(30.6, stDouble.getMax());
        assertEquals(42.1, stLarge.getMin());
        assertEquals(3, stLarge.length());
        assertEquals(4, stDouble.length());


        stDouble.concatenate(stLarge);
        for (Double small: smaller){
            assertTrue(stDouble.contains(small));
        }

        for (Double large: larger) {
            assertTrue(stDouble.contains(large));
        }
    }

    @org.junit.jupiter.api.Test
    void mergeTest() throws Exception {
        Double[] X = {0.88, 0.83, 0.92, 0.71, 0.44, 0.87, 0.15, 0.34, 0.46, 0.99, 0.01, 0.33};
        Double[] Y = {0.81, 0.99, 0.99, 0.99, 0.22, 0.7 , 0.02, 0.24, 0.87, 0.48, 0.54, 0.19, 0.45, 0.92};

        skipList<Double> st1 = new skipList<>(Double.POSITIVE_INFINITY);
        skipList<Double> st2 = new skipList<>(Double.POSITIVE_INFINITY);

        for (Double s: X)
            st1.insert(s);

        for (Double s: Y)
            st2.insert(s);

        skipList<Double> st = st1.merge(st2, false);
        for (Double s: Y)
            assertTrue(st.contains(s));

        Random rand = new Random();
        double[] X1 = rand.doubles(1000).toArray();
        double[] Y1 = rand.doubles(1000).toArray();

        skipList<Double> st3 = new skipList<>(Double.POSITIVE_INFINITY);
        skipList<Double> st4 = new skipList<>(Double.POSITIVE_INFINITY);

        for (Double s: X1)
            st3.insert(s);

        for (Double s: Y1)
            st4.insert(s);

        skipList<Double> st5 = st3.merge(st4, false);

        for (Double s: Y1)
            assertTrue(st5.contains(s));


        int[] X2 = rand.ints(100000).toArray();
        int[] Y2 = rand.ints(100000).toArray();

        skipList<Integer> st6 = new skipList<>(Integer.MAX_VALUE);
        skipList<Integer> st7 = new skipList<>(Integer.MAX_VALUE);

        for (Integer s: X2)
            st6.insert(s);

        for (Integer s: Y2)
            st7.insert(s);

        skipList<Integer> st8 = st6.merge(st7, false);

        for (Integer s: Y2)
            assertTrue(st8.contains(s));

    }
}
