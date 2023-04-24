# A performance study on a more CPU-friendly invariant mass calculation

## Compiling the benchmark

The only requirement is [google/benchmark](https://github.com/google/benchmark).
Assuming the compiler knows where to find its headers and libraries, this should work:

```
$ g++ -O3 -o bench bench.cpp -lbenchmark
$ ./bench
```

On Arch Linux, Google benchmark can be installed with `pacman -Syu benchmark`.

If you don't want to install Google benchmark yourself, the [CMake build](#with-cmake) will do it for you.


### With CMake

The CMake build downloads and builds Google benchmark on the fly if it doesn't find it in the system.

```
$ mkdir build
$ cmake -S . -B build
$ cmake --build build
$ ./build/bench
```

## Context

We want to calculate the following as efficiently as possible:

```cpp
void InvariantMasses(std::size_t bulkSize, const std::vector<bool> &eventMask,
                     float *pts, float *etas, float *phis, float *masses,
                     std::size_t *sizes, std::vector<float> &results) {
  std::size_t elementIdx = 0u;
  for (std::size_t i = 0ul; i < bulkSize; ++i) {
    if (eventMask[i]) { // we don't have a value for this entry yet
      results[i] = InvMass(pts + elementIdx, etas + elementIdx,
                           phis + elementIdx, masses + elementIdx,
                           sizes[i]);
    }
    elementIdx += sizes[i];
  }
```

where `pts`, `etas`, `phis` and `masses` are flattened arrays of arrays of floats (one array per dataset row) and
`sizes` contains the sizes of the sub-arrays for every row. There are `bulkSize` rows in total.
We are only interested in `results[i]` if `eventMask[i]` is `true`.

`InvMass` in turn looks like this:

```cpp
template <typename float>
float InvariantMassBaseline(const float *pt, const float *eta, const float *phi, const float *mass, std::size_t size) {
    float x_sum = 0.f;
    float y_sum = 0.f;
    float z_sum = 0.f;
    float e_sum = 0.f;

    for (std::size_t i = 0u; i < size; ++i) {
      // Convert to (e, x, y, z) coordinate system and update sums
      const auto x = pt[i] * std::cos(phi[i]);
      x_sum += x;
      const auto y = pt[i] * std::sin(phi[i]);
      y_sum += y;
      const auto z = pt[i] * std::sinh(eta[i]);
      z_sum += z;
      const auto e = std::sqrt(x * x + y * y + z * z + mass[i] * mass[i]);
      e_sum += e;
    }

    // Return invariant mass with (+, -, -, -) metric
    return std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                     z_sum * z_sum);
}
```

## Latest results

Running on my laptop with:
- hyperthreading turned off from BIOS
- Intel turbo-boost turned off from BIOS
- Intel speedstep turned off from BIOS
- powersaving disabled via `cpupower frequency-set --governor performance`
- system otherwise idle

```
2023-04-23T18:53:52-06:00
Running ./bench
Run on (8 X 2300.16 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 256 KiB (x8)
  L3 Unified 16384 KiB (x1)
Load Average: 0.40, 0.49, 0.51
-------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
Baseline                      75501 ns        75456 ns         9265
BaselineSimpleSinh            37879 ns        37858 ns        18483
Bulk                          76428 ns        76385 ns         9161
BulkIgnoreMask                70023 ns        69982 ns         9972
BulkIgnoreMaskSimpleSinh      32921 ns        32902 ns        21246
```
