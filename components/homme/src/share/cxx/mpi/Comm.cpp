#include "Comm.hpp"

#ifdef HAVE_CONFIG_H
#include "config.h.c"
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
}

void Comm::init ()
{
  int flag;
  MPI_Initialized (&flag);
  if (!flag)
  {
    // Note: if we move main to C, consider initializing passing &argc/&argv
    //       (the effect on the program would be the same, except that
    //        mpi-specific stuff is removed from argc/argv)
    MPI_Init (nullptr, nullptr);
  }

#ifdef CAM
  #error "Error! Add the code for cam!\n"
#else
  m_mpi_comm = MPI_COMM_WORLD;
#endif

  MPI_Comm_size(m_mpi_comm,&m_size);
  MPI_Comm_rank(m_mpi_comm,&m_rank);
}

} // namespace Homme
