#include "geodex/robots/mass_matrix.hpp"

#include <array>

namespace geodex::robots {

namespace {

constexpr std::array kRegisteredRobots{
    // GEODEX_ROBOT_REGISTERED_BEGIN
    Robot::Baxter,
    Robot::Fr3Gripper,
    Robot::HuskyUr5e,
    Robot::Panda,
    Robot::Pr2,
    Robot::RidgebackUr5e,
    Robot::Stretch3,
    Robot::Stretch4,
    Robot::Ur5,
    // GEODEX_ROBOT_REGISTERED_END
};

}  // namespace

auto registered_robots() -> std::span<const Robot> { return kRegisteredRobots; }

auto robot_from_name(std::string_view s) -> std::optional<Robot> {
  if (s.empty()) return std::nullopt;
  for (const Robot r : kRegisteredRobots) {
    if (name(r) == s) return r;
  }
  return std::nullopt;
}

}  // namespace geodex::robots
