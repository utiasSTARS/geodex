#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_sphere(nb::module_& m);
void bind_euclidean(nb::module_& m);
void bind_torus(nb::module_& m);
void bind_se2(nb::module_& m);
void bind_metrics(nb::module_& m);
void bind_config_space(nb::module_& m);
void bind_algorithms(nb::module_& m);

NB_MODULE(_geodex_core, m) {
  m.doc() = "geodex: planning on Riemannian manifolds";

  bind_sphere(m);
  bind_euclidean(m);
  bind_torus(m);
  bind_se2(m);
  bind_metrics(m);
  bind_config_space(m);
  bind_algorithms(m);
}
