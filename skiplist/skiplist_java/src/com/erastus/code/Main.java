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

	    skipList<Integer> st = new skipList<>(Integer.MAX_VALUE);

        Long[] times = new Long[N];
        long startTime;
        int i = 0;
	    for (Integer n : randIntegers){
	        startTime = System.nanoTime();
	        st.insert(n);
	        times[i++] = (System.nanoTime() - startTime);
        }
	    double mean = (double) (Arrays.stream(times).reduce((long) 0, Long::sum)) / N;
        double sumProd = Arrays.stream(times).map((x) -> (x - mean) * (x - mean)).reduce( 0.0, Double::sum);
        double stdDev = Math.sqrt(sumProd / N);
        System.out.printf("%d skiplist.inserts took: %.2f \u00B1 %.2f \u00B5s", N, mean/1e3, stdDev/1e3);
    }
}
