
#include <Kokkos_Core.hpp>

#include <iostream>

namespace Homme {

extern "C" {

void init_kokkos(const int& num_threads) {
  /* Set OpenMP Environment variables to control how many
   * threads/processors Kokkos uses */

  Kokkos::InitArguments args;
  args.num_threads = num_threads;
  Kokkos::initialize(args);
  //  Kokkos::OpenMP::print_configuration(std::cout, true);
}

void finalize_kokkos() { Kokkos::finalize(); }

} // extern "C"

} // namespace Homme
