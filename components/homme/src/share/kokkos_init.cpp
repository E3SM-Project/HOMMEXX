
#include <Kokkos_Core.hpp>

#include <iostream>

namespace Homme {

extern "C" {
void init_kokkos() {
  /* Set OpenMP Environment variables to control how many
   * threads/processors Kokkos uses */
  Kokkos::initialize();
  Kokkos::OpenMP::print_configuration(std::cout, true);

  std::cout << "Kokkos initialized!" << std::endl <<
    "omp_max_threads: " << omp_get_max_threads() <<
    std::endl << "omp_get_num_threads: " <<
    omp_get_num_threads() << std::endl;
}

void finalize_kokkos() { Kokkos::finalize(); }
}
}  // namespace Homme

