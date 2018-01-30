
#include <iostream>

#include <mpi.h>

#include "mpi/ErrorDefs.hpp"
#include "Hommexx_Session.hpp"

namespace Homme {
namespace Errors {

void runtime_check(bool cond, const std::string& message, int code) {
  if (!cond) {
    runtime_abort(message,code);
  }
}

void runtime_abort(const std::string& message, int code) {
  std::cerr << message << std::endl << "Exiting..." << std::endl;
  finalize_hommexx_session();
  MPI_Abort(MPI_COMM_WORLD, code);
}

} // namespace Homme
} // namespace Errors
