package com.erastus.code;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class Main {

    public static void main(String[] args) {
        // int N = Integer.parseInt(args[1]);
        int N = 1000000;
        int low = 10;
        int high = 50;
        List<Integer> randIntegers = Stream.iterate(0,
                n -> (int) (low + Math.random() * (high - low))).limit(N).collect(Collectors.toList());

        SkipList<Integer> st = new SkipList<>(Integer.MAX_VALUE);

        Long[] times = new Long[N];
        long startTime;
        int i = 0;
        for (Integer n : randIntegers) {
            startTime = System.nanoTime();
            st.insert(n);
            times[i++] = (System.nanoTime() - startTime);
        }
        double mean = (double) (Arrays.stream(times).reduce((long) 0, Long::sum)) / N;
        double sumProd = Arrays.stream(times).map((x) -> (x - mean) * (x - mean)).reduce(0.0, Double::sum);
        double stdDev = Math.sqrt(sumProd / N);
        System.out.printf("insert time: %.2f \u00B1 %.2f \u00B5s \n", mean / 1e3, stdDev / 1e3);

        i = 0;
        for (Integer n : randIntegers) {
            startTime = System.nanoTime();
            st.contains(n);
            times[i++] = (System.nanoTime() - startTime);
        }
        double mean1 = (double) (Arrays.stream(times).reduce((long) 0, Long::sum)) / N;
        sumProd = Arrays.stream(times).map((x) -> (x - mean1) * (x - mean1)).reduce(0.0, Double::sum);
        stdDev = Math.sqrt(sumProd / N);
        System.out.printf("search time: %.2f \u00B1 %.2f ns\n", mean1, stdDev);

        Double[] X = {0.88, 0.83, 0.92, 0.71, 0.81, 0.99, 0.22, 0.7 };
        SkipList<Double> stDouble = new SkipList<>(Double.POSITIVE_INFINITY);
        for (Double x: X){
            stDouble.insert(x);
        }
        System.out.println(Colors.GREEN_BRIGHT);
        stDouble.forEach(x -> System.out.println(x+x));
        System.out.println(Colors.RESET);
    }
}
