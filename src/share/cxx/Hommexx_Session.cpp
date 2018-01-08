#include "Hommexx_Session.hpp"

#include "ExecSpaceDefs.hpp"
#include "profiling.hpp"
#include "Context.hpp"

#include <iostream>

namespace Homme
{

void initialize_hommexx_session ()
{
  /* Make certain profiling is only done for code we're working on */
  profiling_pause();

  /* Set Environment variables to control how many
   * threads/processors Kokkos uses */
  initialize_kokkos();

  ExecSpace::print_configuration(std::cout, true);

  // Put here other initialization routines (e.g., MPI)
}

void finalize_hommexx_session ()
{
  Context::finalize_singleton();
  Kokkos::finalize();

  // Put here other finalization routines (e.g., MPI)
}

} // namespace Homme
