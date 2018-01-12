
#include <iostream>

#include <mpi.h>

#include "mpi/ErrorDefs.hpp"
#include "Hommexx_Session.hpp"

namespace Homme {
namespace Errors {

void runtime_abort(std::string message, int code) {
  std::cerr << message << std::endl << "Exiting..." << std::endl;
  finalize_hommexx_session();
  MPI_Abort(MPI_COMM_WORLD, code);
}
}
}
