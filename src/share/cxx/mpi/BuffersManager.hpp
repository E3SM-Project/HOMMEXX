#ifndef HOMMEXX_MPI_BUFFERS_MANAGER_HPP
#define HOMMEXX_MPI_BUFFERS_MANAGER_HPP

#include "Types.hpp"

#include <vector>
#include <memory>

namespace Homme
{

// Forward declaration
class Connectivity;

class BuffersManager
{
public:

  BuffersManager ();

  // Request enough storage to hold the required number of fields. Does not allocate views
  void request_num_fields (const int num_2d_fields, const int num_3d_fields);

  // Allocate views. Prohibits further calls to request_buffers_sizes
  void allocate_buffers (std::shared_ptr<Connectivity> connectivity);

  // Check that the allocated views have at least the required size
  bool check_views_capacity (const int num_2d_fields,  const int num_3d_fields) const;

  // Deep copy the send/recv buffer to/from the mpi_send/recv buffer
  // Note: these are no-ops if MPIMemSpace=ExecMemSpace
  void sync_send_buffer ();
  void sync_recv_buffer ();

  ExecViewManaged<Real*> get_send_buffer           () const { return m_send_buffer;           }
  ExecViewManaged<Real*> get_recv_buffer           () const { return m_recv_buffer;           }
  ExecViewManaged<Real*> get_local_buffer          () const { return m_local_buffer;          }
  MPIViewManaged<Real*>  get_mpi_send_buffer       () const { return m_mpi_send_buffer;       }
  MPIViewManaged<Real*>  get_mpi_recv_buffer       () const { return m_mpi_recv_buffer;       }
  ExecViewManaged<Real*> get_blackhole_send_buffer () const { return m_blackhole_send_buffer; }
  ExecViewManaged<Real*> get_blackhole_recv_buffer () const { return m_blackhole_recv_buffer; }

private:

  // The number of 2d and 3d fields that the buffers need to be able to handle
  int m_num_2d_fields;
  int m_num_3d_fields;

  // Used to check whether user can still request different sizes
  bool m_views_allocated;

  // The buffers
  ExecViewManaged<Real*>  m_send_buffer;
  ExecViewManaged<Real*>  m_recv_buffer;
  ExecViewManaged<Real*>  m_local_buffer;

  // The mpi buffers (same as the previous send/recv buffers if MPIMemSpace=ExecMemSpace)
  MPIViewManaged<Real*>   m_mpi_send_buffer;
  MPIViewManaged<Real*>   m_mpi_recv_buffer;

  // The blackhole send/recv buffers (used for missing connections)
  ExecViewManaged<Real*>  m_blackhole_send_buffer;
  ExecViewManaged<Real*>  m_blackhole_recv_buffer;
};

inline void BuffersManager::sync_send_buffer ()
{
  Kokkos::deep_copy(m_mpi_send_buffer, m_send_buffer);
}

inline void BuffersManager::sync_recv_buffer ()
{
  Kokkos::deep_copy(m_recv_buffer, m_mpi_recv_buffer);
}

} // namespace Homme

#endif // HOMMEXX_MPI_BUFFERS_MANAGER_HPP
