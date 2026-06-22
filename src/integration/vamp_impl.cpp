/// @file vamp_impl.cpp
/// @brief Single thin compilation unit hosting the VAMP integration's
///        SIMD-tainted code.
///
/// This translation unit is the only place inside the build that compiles
/// VAMP's intrinsics; the @c geodex_vamp static archive applies AVX2/FMA
/// (x86_64) or NEON-by-default (aarch64) compile options to it as PRIVATE
/// flags so consumer translation units stay free of SIMD options. The
/// header-only public API in @c registry.hpp resolves to the out-of-line
/// definitions below.

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "geodex/integration/vamp/detail/scene_loader.hpp"
#include "geodex/integration/vamp/registry.hpp"
#include "geodex/integration/vamp/robots/baxter.hpp"
#include "geodex/integration/vamp/robots/fetch.hpp"
#include "geodex/integration/vamp/robots/panda.hpp"
#include "geodex/integration/vamp/robots/pr2.hpp"
#include "geodex/integration/vamp/robots/ur5.hpp"

namespace geodex::integration::vamp {

auto load_scene(const std::string& yaml_path) -> EnvHandle {
  return detail::load_scene_impl(yaml_path);
}

auto make_vamp_checker(const std::string& robot_name, EnvHandle env)
    -> std::unique_ptr<CollisionChecker> {
  if (robot_name == "baxter") return detail::make_baxter_checker(std::move(env));
  if (robot_name == "fetch") return detail::make_fetch_checker(std::move(env));
  if (robot_name == "panda") return detail::make_panda_checker(std::move(env));
  if (robot_name == "pr2") return detail::make_pr2_checker(std::move(env));
  if (robot_name == "ur5") return detail::make_ur5_checker(std::move(env));
  throw std::runtime_error("geodex::integration::vamp: no robot registered as '" + robot_name +
                           "'");
}

auto make_vamp_motion_validator(const std::string& robot_name,
                                const ompl::base::SpaceInformationPtr& si, EnvHandle env)
    -> std::unique_ptr<ompl::base::MotionValidator> {
  if (robot_name == "baxter") return detail::make_baxter_motion_validator(si, std::move(env));
  if (robot_name == "fetch") return detail::make_fetch_motion_validator(si, std::move(env));
  if (robot_name == "panda") return detail::make_panda_motion_validator(si, std::move(env));
  if (robot_name == "pr2") return detail::make_pr2_motion_validator(si, std::move(env));
  if (robot_name == "ur5") return detail::make_ur5_motion_validator(si, std::move(env));
  throw std::runtime_error("geodex::integration::vamp: no robot registered as '" + robot_name +
                           "'");
}

auto registered_robots() -> std::vector<std::string> {
  return {"baxter", "fetch", "panda", "pr2", "ur5"};
}

}  // namespace geodex::integration::vamp
