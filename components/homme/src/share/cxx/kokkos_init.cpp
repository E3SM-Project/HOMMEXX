#include <Types.hpp>

#include <iostream>

namespace Homme {

extern "C" {

void init_kokkos(const bool print_configuration = true) {
  /* Set OpenMP Environment variables to control how many
   * threads/processors Kokkos uses */

  Kokkos::initialize();

  ExecSpace::print_configuration(std::cout, print_configuration);
}

void finalize_kokkos() { Kokkos::finalize(); }

} // extern "C"

} // namespace Homme
