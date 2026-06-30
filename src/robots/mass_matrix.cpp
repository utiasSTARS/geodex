#include "geodex/robots/mass_matrix.hpp"

namespace geodex::robots {

auto registered_robots() -> std::vector<Robot> {
  return {
  // GEODEX_ROBOT_REGISTERED_BEGIN
    Robot::Panda,
    Robot::Ur5,
    Robot::Fetch,
    Robot::Baxter,
    Robot::Pr2,
  // GEODEX_ROBOT_REGISTERED_END
  };
}

}  // namespace geodex::robots
