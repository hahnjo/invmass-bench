# A performance study on a more CPU-friendly invariant mass calculation

## Compiling the benchmark

The only requirement is [google/benchmark](https://github.com/google/benchmark).
Assuming the compiler knows where to find its headers and libraries, this should work:

```
$ g++ -O2 -o bench bench.cpp -lbenchmark
```
