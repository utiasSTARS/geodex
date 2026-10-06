// Example of the Robot Guides page, the C++ version of catalog.py.
//
// Lists the built-in robots and the number of configuration coordinates of each.
// The count of a mobile robot includes its base pose.
//
// Usage: catalog [--json out.json]

// [docs-start:catalog]
#include <cstdio>

#include <string>

#include <geodex/integration/vamp/registry.hpp>
#include <geodex/robots/mass_matrix.hpp>
// [docs-end:catalog]

#include <vector>

#include "../common/json_output.hpp"

// [docs-start:catalog]
int main(int argc, char** argv) {
  namespace gr = geodex::robots;
  namespace gv = geodex::integration::vamp;

  for (const auto robot : gr::registered_robots()) {
    const std::string name(gr::name(robot));
    std::printf("%16s: %d coordinates\n", name.c_str(), gv::robot_dimension(name));
  }
  // [docs-end:catalog]

  using geodex_examples::Json;
  std::vector<Json> robots;
  for (const auto robot : gr::registered_robots()) {
    const std::string name(gr::name(robot));
    robots.push_back(
        Json::object({{"name", name}, {"dim", gv::robot_dimension(name)}}));
  }
  geodex_examples::write_json_arg(argc, argv,
                                  Json::object({{"robots", Json::array(robots)}}));
  // [docs-start:catalog]
  return 0;
}
// [docs-end:catalog]
