// Reads one configuration per line from stdin. For each, prints "valid <0|1>" with the
// kernel's self-collision result in an empty environment, then "x y z r" for each sphere in
// the kernel's order. Build with -DKERNEL_HEADER="path/to/robot.hh" -DKERNEL_STRUCT=Name.

#include <cstdio>
#include <iostream>
#include <vector>

#include KERNEL_HEADER

int main() {
  using Robot = ::vamp::robots::KERNEL_STRUCT;
  constexpr std::size_t W = ::vamp::FloatVectorWidth;
  ::vamp::collision::Environment<::vamp::FloatVector<W>> env;
  std::vector<double> q(Robot::dimension);
  while (true) {
    for (auto& v : q) {
      if (!(std::cin >> v)) return 0;
    }
    typename Robot::template ConfigurationBlock<W> block;
    for (std::size_t j = 0; j < Robot::dimension; ++j) {
      block[j] = ::vamp::FloatVector<W>(static_cast<float>(q[j]));
    }
    typename Robot::template Spheres<W> out;
    Robot::template sphere_fk<W>(block, out);
    const auto x = out.x.to_array();
    const auto y = out.y.to_array();
    const auto z = out.z.to_array();
    const auto r = out.r.to_array();
    const std::size_t stride = x.size() / Robot::n_spheres;
    std::printf("valid %d\n", Robot::template fkcc<W>(env, block) ? 1 : 0);
    for (std::size_t i = 0; i < Robot::n_spheres; ++i) {
      std::printf("%.7g %.7g %.7g %.7g\n", x[i * stride], y[i * stride], z[i * stride],
                  r[i * stride]);
    }
  }
}
