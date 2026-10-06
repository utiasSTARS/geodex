#include <cstdint>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>

#include "geodex/core/sampler.hpp"

namespace nb = nanobind;

void bind_samplers(nb::module_& m) {
  nb::class_<geodex::ScrambledHaltonSampler>(
      m, "ScrambledHaltonSampler",
      "Scrambled Halton low-discrepancy sampler, the geodex default.\n\n"
      "Samples points in [0, 1)^n with even coverage and a random per-seed scramble.\n"
      "Pass a seed for a reproducible sequence, or none for a fresh scramble\n"
      "from the global source.")
      .def(nb::init<>())
      .def(nb::init<std::uint64_t>(), nb::arg("seed"))
      .def(
          "sample",
          [](geodex::ScrambledHaltonSampler& s, int n) {
            Eigen::VectorXd out(n);
            s.sample(n, out);
            return out;
          },
          nb::arg("n"), "Sample the next n-dimensional point in [0, 1)^n.")
      .def("seed", &geodex::ScrambledHaltonSampler::seed, nb::arg("seed"),
           "Reseed with a fresh scramble and start.");

  nb::class_<geodex::HaltonSampler>(
      m, "HaltonSampler",
      "Deterministic Halton low-discrepancy sampler.\n\n"
      "Samples the same sequence in [0, 1)^n every run, with no randomization.")
      .def(nb::init<>())
      .def(
          "sample",
          [](geodex::HaltonSampler& s, int n) {
            Eigen::VectorXd out(n);
            s.sample(n, out);
            return out;
          },
          nb::arg("n"), "Sample the next n-dimensional point in [0, 1)^n.")
      .def("seed", &geodex::HaltonSampler::seed, nb::arg("seed"),
           "Reset the sequence to start from the given index.");

  nb::class_<geodex::PseudoRandomSampler>(
      m, "PseudoRandomSampler",
      "Pseudo-random sampler wrapping mt19937, the i.i.d. baseline.\n\n"
      "Pass a seed for a reproducible stream, or none to share a thread-local\n"
      "generator across default instances.")
      .def(nb::init<>())
      .def(nb::init<std::uint64_t>(), nb::arg("seed"))
      .def(
          "sample",
          [](geodex::PseudoRandomSampler& s, int n) {
            Eigen::VectorXd out(n);
            s.sample(n, out);
            return out;
          },
          nb::arg("n"), "Sample the next n-dimensional point in [0, 1)^n.")
      .def("seed", &geodex::PseudoRandomSampler::seed, nb::arg("seed"),
           "Reseed and switch to an owned generator.");
}
