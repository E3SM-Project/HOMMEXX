#ifndef HOMMEXX_COMM_HPP
#define HOMMEXX_COMM_HPP

#include <mpi.h>

namespace Homme
{

// A small wrapper around an MPI_Comm, together with its rank/size

// Note: I thought about using Teuchos::MpiComm from Trilinos,
//       but Teuchos does not support one-sided communication,
//       which we *may* use in the future in the cxx code

struct Comm
{
  MPI_Comm  m_mpi_comm;

  int       m_size;
  int       m_rank;

  Comm ();
  Comm (MPI_Comm mpi_comm);

  // So far this simply gets info from MPI_COMM_WORLD. But when
  // Hommexx will be plugged back into acme, it will have to find
  // out what processes are dedicated to Homme, and create a comm
  // involving only those processes.
  void init ();
};

} // namespace Homme

#endif // HOMMEXX_COMM_HPP
