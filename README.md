# A performance study on a more CPU-friendly invariant mass calculation

## Compiling the benchmark

The only requirement is [google/benchmark](https://github.com/google/benchmark).
On Arch it can be installed with `pacman -Syu benchmark`.
If you don't want to install Google benchmark yourself, see [below](#with-cmake).

Assuming the compiler knows where to find its headers and libraries, this should work:

```
$ g++ -O3 -o bench bench.cpp -lbenchmark
$ ./bench
```

### With CMake

The CMake build download and build Google benchmark on the fly if it doesn't find it in the system.

```
$ mkdir build
$ cmake -S . -B build
$ cmake --build build
$ ./build/bench
```
