#include <benchmark/benchmark.h>

#include <cmath>
#include <numeric> // std::accumulate
#include <random>
#include <stdexcept>
#include <vector>

struct Input {
  std::size_t bulkSize;
  std::vector<bool> requestedMask;
  float *pts;
  float *etas;
  float *phis;
  float *masses;
  std::size_t *sizes;
};

Input MakeInput() {
  std::default_random_engine e(1234); // fixed seed
  std::uniform_real_distribution<double> rand(0., 1.);

  std::size_t bulkSize = 1000 + rand(e) * 2;
  std::size_t *sizes = new std::size_t[bulkSize];
  std::vector<bool> requestedMask(bulkSize);
  std::size_t nElements = 0u;
  for (std::size_t i = 0u; i < bulkSize; ++i) {
    double val = rand(e);
    sizes[i] = val * 5;
    requestedMask[i] = val > 0.6;
    nElements += sizes[i];
  }

  float *pts = new float[nElements];
  float *etas = new float[nElements];
  float *phis = new float[nElements];
  float *masses = new float[nElements];

  for (std::size_t i = 0u; i < nElements; ++i) {
    const float val = rand(e);
    pts[i] = val;
    etas[i] = val;
    phis[i] = val;
    masses[i] = val;
  }

  return Input{bulkSize, requestedMask, pts, etas, phis, masses, sizes};
}

static const auto input = MakeInput();

void SanityCheck(const std::vector<float> &results) {
  const auto sum = std::accumulate(results.begin(), results.end(), 0.);
  if (std::abs(sum - 736.624) > 1e-5)
    std::runtime_error("Sanity check failed: sum of results was " +
                       std::to_string(sum) + " instead of 736.624");
}

// original
template <typename T>
T InvariantMassBaseline(T *pt, T *eta, T *phi, T *mass, std::size_t size) {
  T x_sum = 0.;
  T y_sum = 0.;
  T z_sum = 0.;
  T e_sum = 0.;

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

void EvalDefineBaseline(std::size_t bulkSize,
                        const std::vector<bool> &requestedMask, float *pts,
                        float *etas, float *phis, float *masses,
                        std::size_t *sizes, std::vector<float> &results,
                        std::vector<bool> &currentMask) {
  std::size_t elementIdx = 0u;
  for (std::size_t i = 0ul; i < bulkSize; ++i) {
    if (requestedMask[i] &&
        !currentMask[i]) { // we don't have a value for this entry yet
      results[i] = InvariantMassBaseline(pts + elementIdx, etas + elementIdx,
                                         phis + elementIdx, masses + elementIdx,
                                         sizes[i]);
      currentMask[i] = true;
    }
    elementIdx += sizes[i];
  }
}

static void Baseline(benchmark::State &state) {
  std::vector<float> results(input.bulkSize);
  benchmark::DoNotOptimize(results);
  for (auto _ : state) {
    std::vector<bool> currentMask(input.bulkSize, false);
    benchmark::DoNotOptimize(currentMask);
    EvalDefineBaseline(input.bulkSize, input.requestedMask, input.pts,
                       input.etas, input.phis, input.masses, input.sizes,
                       results, currentMask);
    // to force writing to memory of results and currentMask
    benchmark::ClobberMemory();
  }

  SanityCheck(results);
}
BENCHMARK(Baseline);

template <typename T>
void InvariantMassBulk(const std::vector<bool> &requestedMask,
                       std::vector<bool> &currentMask, std::size_t bulkSize,
                       std::vector<T> &results, T *pt, T *eta, T *phi, T *mass,
                       std::size_t *sizes) {
  std::size_t elementIdx = 0u;
  for (std::size_t i = 0; i < bulkSize; ++i) {
    if (requestedMask[i] && !currentMask[i]) {

      T x_sum = 0.;
      T y_sum = 0.;
      T z_sum = 0.;
      T e_sum = 0.;

      // short loop: sizes[i] is typically small (~2.5 on average)
      for (std::size_t j = 0u; j < sizes[i]; ++j) {
        const auto x = pt[elementIdx + j] * std::cos(phi[elementIdx + j]);
        x_sum += x;
        const auto y = pt[elementIdx + j] * std::sin(phi[elementIdx + j]);
        y_sum += y;
        const auto z = pt[elementIdx + j] * std::sinh(eta[elementIdx + j]);
        z_sum += z;
        // this single-scalar sqrt is the hottest instruction
        const auto e = std::sqrt(x * x + y * y + z * z +
                                 mass[elementIdx + j] * mass[elementIdx + j]);
        e_sum += e;
      }

      results[i] = std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                             z_sum * z_sum);
    }
    elementIdx += sizes[i];
  }
}

static void Bulk(benchmark::State &state) {
  std::vector<float> results(input.bulkSize);
  benchmark::DoNotOptimize(results);
  for (auto _ : state) {
    std::vector<bool> currentMask(input.bulkSize, false);
    benchmark::DoNotOptimize(currentMask);
    InvariantMassBulk(input.requestedMask, currentMask, input.bulkSize, results,
                      input.pts, input.etas, input.phis, input.masses,
                      input.sizes);
    // to force writing to memory of results and currentMask
    benchmark::ClobberMemory();
  }

  SanityCheck(results);
}
BENCHMARK(Bulk);

template <typename T>
void InvariantMassBulkIgnoreMask(const std::vector<bool> &requestedMask,
                                 std::vector<bool> &currentMask,
                                 std::size_t bulkSize, std::vector<T> &results,
                                 T *pt, T *eta, T *phi, T *mass,
                                 std::size_t *sizes) {

  const auto nElements = std::accumulate(sizes, sizes + bulkSize, 0u);

  std::vector<T> xs(nElements);
  std::vector<T> ys(nElements);
  std::vector<T> zs(nElements);
  std::vector<T> es2(nElements);
  std::vector<T> es(nElements);

  // trigonometric functions are expensive and don't vectorize so we only call them when needed
  std::size_t elementIdx = 0u;
  for (std::size_t i = 0u; i < bulkSize; ++i) {
    const auto size = sizes[i];
    if (requestedMask[i] && !currentMask[i]) {
      for (std::size_t j = 0u; j < size; ++j) {
        const auto pt_ = pt[elementIdx + j];
        const auto phi_ = phi[elementIdx + j];
        xs[i] = pt_ * std::cos(phi_);
        ys[i] = pt_ * std::sin(phi_);
        zs[i] = pt_ * std::sinh(eta[elementIdx + j]);
      }
    }
    elementIdx += size;
  }

  // looks like the CPU is happier by calculating these for all elements, even if we'll discard many of the results...
  for (std::size_t i = 0; i < nElements; ++i) {
    es2[i] = pt[i] * pt[i] + zs[i] * zs[i] + mass[i] * mass[i];
  }

  for (std::size_t i = 0; i < nElements; ++i) {
    es[i] = std::sqrt(es2[i]);
  }

  elementIdx = 0u;
  for (std::size_t i = 0; i < bulkSize; ++i) {
    T x_sum = 0.;
    T y_sum = 0.;
    T z_sum = 0.;
    T e_sum = 0.;
    const auto size = sizes[i];
    if (requestedMask[i] && !currentMask[i]) {
      for (std::size_t j = 0u; j < sizes[i]; ++j) {
        const auto idx = elementIdx + j;
        x_sum += xs[idx];
        y_sum += ys[idx];
        z_sum += zs[idx];
        e_sum += es[idx];
      }
      results[i] = std::sqrt(e_sum * e_sum - x_sum * x_sum - y_sum * y_sum -
                             z_sum * z_sum);
    }
    elementIdx += size;
  }
}

static void BulkIgnoreMask(benchmark::State &state) {
  std::vector<float> results(input.bulkSize);
  benchmark::DoNotOptimize(results);
  for (auto _ : state) {
    std::vector<bool> currentMask(input.bulkSize, false);
    benchmark::DoNotOptimize(currentMask);
    InvariantMassBulkIgnoreMask(input.requestedMask, currentMask,
                                input.bulkSize, results, input.pts, input.etas,
                                input.phis, input.masses, input.sizes);
    // to force writing to memory of results and currentMask
    benchmark::ClobberMemory();
  }

  SanityCheck(results);
}
BENCHMARK(BulkIgnoreMask);

BENCHMARK_MAIN();
