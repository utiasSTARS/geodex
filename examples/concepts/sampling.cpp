// Examples of the Sampling concept page, the C++ version of sampling.py. It also
// defines a custom sampler. Custom samplers are C++ only.
//
// Usage: sampling [--json out.json]

#include <cmath>
#include <cstdint>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

// [docs-start:random-point]
#include <geodex/geodex.hpp>
// [docs-end:random-point]

#include "../common/json_output.hpp"

using geodex_examples::Json;

// [docs-start:custom-sampler]
/// Kronecker sequence sampler. Coordinate i of point k is frac(k * sqrt(p_i)) for
/// the i-th prime p_i. The sequence is low-discrepancy and does not need tables.
struct KroneckerSampler {
  std::uint64_t k = 0;

  void sample(int n, Eigen::Ref<Eigen::VectorXd> out) {
    static constexpr double primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    ++k;
    for (int i = 0; i < n; ++i) {
      const double x = static_cast<double>(k) * std::sqrt(primes[i % 12]);
      out[i] = x - std::floor(x);
    }
  }

  void seed(std::uint64_t s) { k = s; }  // makes this a SeedableSampler
};

// Any manifold takes it in its sampler slot.
using KroneckerPlane =
    geodex::Euclidean<2, geodex::EuclideanStandardMetric<2>, KroneckerSampler>;
// [docs-end:custom-sampler]

int main(int argc, char** argv) {
  std::vector<std::pair<std::string, Json>> out;

  {
    // [docs-start:random-point]
    geodex::Sphere<> sphere;
    geodex::Euclidean<3> r3;
    geodex::SO3<> so3;

    auto s = sphere.random_point();  // uniform on the sphere
    auto p = r3.random_point();      // uniform in the box [-1, 1]^3
    auto r = so3.random_point();     // Haar-uniform rotation, a unit quaternion
    // [docs-end:random-point]
    out.emplace_back("random_point", Json::array({static_cast<int>(s.size()),
                                                  static_cast<int>(p.size()),
                                                  static_cast<int>(r.size())}));
  }

  {
    // [docs-start:choose-sampler]
    // The sampler is the last policy of a manifold. This type samples plain Halton.
    geodex::Euclidean<3, geodex::EuclideanStandardMetric<3>, geodex::HaltonSampler>
        r3;
    // [docs-end:choose-sampler]
  }

  {
    // [docs-start:seed]
    geodex::Sphere<> a, b;
    a.seed(42);
    b.seed(42);
    std::cout << (a.random_point() == b.random_point())
              << "\n";  // 1, the same sequence

    geodex::set_default_seed(
        7);  // reseed the source of every default sampler constructed afterwards
    // [docs-end:seed]
    a.seed(42);
    out.emplace_back("seed", Json(Eigen::Vector3d(a.random_point())));
  }

  {
    // [docs-start:product]
    auto pm = geodex::make_product(geodex::SO3<>{}, geodex::Euclidean<3>{});
    pm.seed(0);
    auto x =
        pm.random_point();  // one joint scrambled Halton sample over both blocks
    // [docs-end:product]
    out.emplace_back("product", Json(Eigen::VectorXd(x)));
  }

  {
    // [docs-start:standalone]
    geodex::ScrambledHaltonSampler s{5};  // a seed makes the sequence reproducible
    Eigen::VectorXd u(3);
    s.sample(3, u);  // u holds the next point of the sequence in [0, 1)^3
    // [docs-end:standalone]
    Eigen::VectorXd next(3);
    s.sample(3, next);
    out.emplace_back("standalone", Json::array({Json(u), Json(next)}));
  }

  {
    KroneckerPlane plane;
    plane.seed(0);
    std::cout << plane.random_point().transpose() << "\n";
  }

  geodex_examples::write_json_arg(argc, argv, Json::object(out));
  return 0;
}
