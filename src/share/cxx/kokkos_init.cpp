#include <Types.hpp>

#include <iostream>

namespace Homme {

extern "C" {

void init_kokkos(const int& num_threads, bool print_configuration = true) {
  /* Set OpenMP Environment variables to control how many
   * threads/processors Kokkos uses */

  Kokkos::InitArguments args;
  args.num_threads = num_threads;
  Kokkos::initialize(args);

  Homme::ExecSpace::print_configuration(std::cout, print_configuration);
}

void finalize_kokkos() { Kokkos::finalize(); }

} // extern "C"

} // namespace Homme
