#include "Comm.hpp"

#ifdef HAVE_CONFIG_H
#include "config.h.c"
#endif
#include "Hommexx_config.h"

#ifdef CAM
#error "Error! We have not yet implemented the interface with CAM.\n"
#endif

namespace Homme
{

Comm::Comm()
 : m_mpi_comm (MPI_COMM_NULL)
 , m_size     (0)
 , m_rank     (-1)
{
  // Nothing to be done here
}

Comm::Comm(MPI_Comm mpi_comm)
 : m_mpi_comm (mpi_comm)
{
  MPI_Comm_size(m_mpi_comm,&m_size);
  MPI_Comm_rank(m_mpi_comm,&m_rank);

#ifndef NDEBUG
  MPI_Comm_set_errhandler(m_mpi_comm,MPI_ERRORS_RETURN);
#endif
}

void Comm::init ()
{
  if (m_mpi_comm!=MPI_COMM_NULL) {
    std::cerr << "Error! You cannot call 'init' if the Comm was constructed from a given MPI_Comm.\n";
    MPI_Abort(m_mpi_comm,1);
  }
  int flag;
  MPI_Initialized (&flag);
  if (!flag)
  {
    // Note: if we move main to C, consider initializing passing &argc/&argv
    //       (the effect on the program would be the same, except that
    //        mpi-specific stuff is removed from argc/argv)
    MPI_Init (nullptr, nullptr);
  }

  m_mpi_comm = MPI_COMM_WORLD;

  MPI_Comm_size(m_mpi_comm,&m_size);
  MPI_Comm_rank(m_mpi_comm,&m_rank);

#ifndef NDEBUG
  MPI_Comm_set_errhandler(m_mpi_comm,MPI_ERRORS_RETURN);
#endif
}

} // namespace Homme
