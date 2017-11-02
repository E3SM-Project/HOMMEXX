#include <Types.hpp>
#include "profiling.hpp"
#include "Context.hpp"

#include <iostream>

namespace Homme {

extern "C" {

void init_kokkos(const bool print_configuration = true) {
  /* Make certain profiling is only done for code we're working on */
  profiling_pause();

  /* Set OpenMP Environment variables to control how many
   * threads/processors Kokkos uses */
  Kokkos::initialize();

  ExecSpace::print_configuration(std::cout, print_configuration);
}

void finalize_kokkos() {
  Homme::Context::finalize_singleton();
  Kokkos::finalize();
}

} // extern "C"

} // namespace Homme
