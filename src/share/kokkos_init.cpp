
#include <Kokkos_Core.hpp>

#include <iostream>

namespace Homme {

extern "C" {
void init_kokkos() {
  /* Set OpenMP Environment variables to control how many
   * threads/processors Kokkos uses */
  Kokkos::initialize();
//  Kokkos::OpenMP::print_configuration(std::cout, true);
}

void finalize_kokkos() { Kokkos::finalize(); }
}
}  // namespace Homme

