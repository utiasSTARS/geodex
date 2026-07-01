#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_sphere(nb::module_& m);
void bind_euclidean(nb::module_& m);
void bind_torus(nb::module_& m);
void bind_se2(nb::module_& m);
void bind_so2(nb::module_& m);
void bind_so3(nb::module_& m);
void bind_se3(nb::module_& m);
void bind_product(nb::module_& m);
void bind_metrics(nb::module_& m);
void bind_config_space(nb::module_& m);
void bind_algorithms(nb::module_& m);
void bind_heuristics(nb::module_& m);
void bind_collision(nb::module_& m);

#ifdef GEODEX_PYTHON_HAS_PINOCCHIO
void bind_pinocchio(nb::module_& m);
#endif

#ifdef GEODEX_PYTHON_HAS_VAMP
void bind_vamp(nb::module_& m);
#endif

NB_MODULE(_geodex_core, m) {
  m.doc() = "geodex: planning on Riemannian manifolds";

  bind_sphere(m);
  bind_euclidean(m);
  bind_torus(m);
  bind_se2(m);
  bind_so2(m);
  bind_so3(m);
  bind_se3(m);
  bind_metrics(m);
  bind_config_space(m);
  bind_product(m);
  bind_heuristics(m);
  bind_algorithms(m);
  bind_collision(m);

#ifdef GEODEX_PYTHON_HAS_PINOCCHIO
  bind_pinocchio(m);
#endif

#ifdef GEODEX_PYTHON_HAS_VAMP
  bind_vamp(m);
#endif
}
