// Exits non-zero unless the linked OMPL fork carries the geodex OMPL patch.
//
// The patch gives GreedyRRTstar one informed sampler per tree (checked at compile time
// through the goal-tree member) and registers the symmetric_motion_validity and
// max_neighbors parameters in the library's constructor (checked at run time against the
// compiled library, not only the headers).

#include <cstdio>
#include <memory>

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/planners/rrt/GreedyRRTstar.h>

namespace {

class PatchedGreedyRRTstar : public ompl::geometric::GreedyRRTstar {
 public:
  using GreedyRRTstar::GreedyRRTstar;

  bool has_goal_tree_sampler_member() const {
    return sizeof(infSamplerGoal_) == sizeof(ompl::base::InformedSamplerPtr);
  }
};

}  // namespace

int main() {
  auto space = std::make_shared<ompl::base::RealVectorStateSpace>(2);
  space->setBounds(0.0, 1.0);
  auto si = std::make_shared<ompl::base::SpaceInformation>(space);
  si->setStateValidityChecker([](const ompl::base::State*) { return true; });
  si->setup();

  PatchedGreedyRRTstar planner(si);
  int failures = 0;
  if (!planner.has_goal_tree_sampler_member()) {
    std::fprintf(stderr, "missing per-tree informed sampler\n");
    ++failures;
  }
  for (const char* name : {"symmetric_motion_validity", "max_neighbors"}) {
    if (!planner.params().hasParam(name)) {
      std::fprintf(stderr, "GreedyRRTstar does not register the '%s' parameter\n", name);
      ++failures;
    }
  }
  planner.setMaxNeighbors(8);
  planner.setSymmetricMotionValidity(false);
  if (planner.getMaxNeighbors() != 8 || planner.getSymmetricMotionValidity()) {
    std::fprintf(stderr, "patched GreedyRRTstar setters do not round-trip\n");
    ++failures;
  }

  if (failures != 0) {
    std::fprintf(stderr, "the installed OMPL fork lacks the geodex OMPL patch\n");
    return 1;
  }
  std::printf("OMPL fork carries the geodex OMPL patch\n");
  return 0;
}
