#include "BuffersManager.hpp"

#include "Connectivity.hpp"

namespace Homme
{

BuffersManager::BuffersManager ()
 : m_num_2d_fields   (0)
 , m_num_3d_fields   (0)
 , m_views_allocated (false)
{
  // Nothing to be done here
}

void BuffersManager::request_num_fields (const int num_2d_fields, const int num_3d_fields)
{
  if (m_views_allocated) {
    // You cannot call this method after `allocate_buffers` has been called
    // TODO: would like to use MPI_Abort, but to do that, you need a communicator
    std::cerr << "Error! Buffer views have already been allocated.\n";
    std::abort();
  }

  if (num_2d_fields>m_num_2d_fields) {
    m_num_2d_fields = num_2d_fields;
  }

  if (num_3d_fields>m_num_3d_fields) {
    m_num_3d_fields = num_3d_fields;
  }
}

void BuffersManager::allocate_buffers (std::shared_ptr<Connectivity> connectivity)
{
  // Make sure connectivity is valid
  assert (connectivity && connectivity->is_finalized());

  // The buffer size for each connection kind
  int elem_buf_size[2];
  elem_buf_size[etoi(ConnectionKind::CORNER)] = (m_num_2d_fields + m_num_3d_fields*NUM_LEV*VECTOR_SIZE) * 1;
  elem_buf_size[etoi(ConnectionKind::EDGE)]   = (m_num_2d_fields + m_num_3d_fields*NUM_LEV*VECTOR_SIZE) * NP;

  // Compute the buffers sizes and allocating
  size_t mpi_buffer_size = 0;
  size_t local_buffer_size = 0;
  size_t blackhole_buffer_size = 0;

  mpi_buffer_size += elem_buf_size[etoi(ConnectionKind::CORNER)] * connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::SHARED,ConnectionKind::CORNER);
  mpi_buffer_size += elem_buf_size[etoi(ConnectionKind::EDGE)]   * connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::SHARED,ConnectionKind::EDGE);

  local_buffer_size += elem_buf_size[etoi(ConnectionKind::CORNER)] * connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::LOCAL,ConnectionKind::CORNER);
  local_buffer_size += elem_buf_size[etoi(ConnectionKind::EDGE)]   * connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::LOCAL,ConnectionKind::EDGE);

  blackhole_buffer_size += NUM_LEV * VECTOR_SIZE;

  // The buffers used for packing/unpacking
  m_send_buffer  = ExecViewManaged<Real*>("send buffer",  mpi_buffer_size);
  m_recv_buffer  = ExecViewManaged<Real*>("recv buffer",  mpi_buffer_size);
  m_local_buffer = ExecViewManaged<Real*>("local buffer", local_buffer_size);

  // The buffers used in MPI calls
  m_mpi_send_buffer = Kokkos::create_mirror_view(decltype(m_mpi_send_buffer)::execution_space(),m_send_buffer);
  m_mpi_recv_buffer = Kokkos::create_mirror_view(decltype(m_mpi_recv_buffer)::execution_space(),m_recv_buffer);

  // The "fake" buffers used for MISSING connections
  m_blackhole_send_buffer = ExecViewManaged<Real*>("blackhole array",blackhole_buffer_size);
  m_blackhole_recv_buffer = ExecViewManaged<Real*>("blackhole array",blackhole_buffer_size);
  Kokkos::deep_copy(m_blackhole_send_buffer,0.0);
  Kokkos::deep_copy(m_blackhole_recv_buffer,0.0);

  m_views_allocated = true;
}

bool BuffersManager::check_views_capacity (const int num_2d_fields, const int num_3d_fields) const
{
  return (m_num_2d_fields>=num_2d_fields) && (m_num_3d_fields>=num_3d_fields);
}

} // namespace Homme
