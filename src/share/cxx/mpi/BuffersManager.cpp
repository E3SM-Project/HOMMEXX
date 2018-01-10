#include "BuffersManager.hpp"

#include "BoundaryExchange.hpp"
#include "Connectivity.hpp"

namespace Homme
{

BuffersManager::BuffersManager ()
 : m_mpi_buffer_size   (0)
 , m_local_buffer_size (0)
 , m_views_are_valid   (false)
{
  // The "fake" buffers used for MISSING connections. These do not depend on the requirements
  // from the custormers, so we can create them right away.
  constexpr size_t blackhole_buffer_size = NUM_LEV * VECTOR_SIZE;
  m_blackhole_send_buffer = ExecViewManaged<Real*>("blackhole array",blackhole_buffer_size);
  m_blackhole_recv_buffer = ExecViewManaged<Real*>("blackhole array",blackhole_buffer_size);
  Kokkos::deep_copy(m_blackhole_send_buffer,0.0);
  Kokkos::deep_copy(m_blackhole_recv_buffer,0.0);
}

BuffersManager::BuffersManager (std::shared_ptr<Connectivity> connectivity)
 : BuffersManager()
{
  set_connectivity(connectivity);
}

void BuffersManager::check_for_reallocation ()
{
  for (auto be_ptr : m_customers_be) {
    update_requested_sizes (be_ptr->get_num_2d_fields(),be_ptr->get_num_3d_fields());
  }
}

void BuffersManager::set_connectivity (std::shared_ptr<Connectivity> connectivity)
{
  // We don't allow a null connectivity, or a change of connectivity
  assert (connectivity && !m_connectivity);

  m_connectivity = connectivity;
}

bool BuffersManager::check_views_capacity (const int num_2d_fields, const int num_3d_fields) const
{
  size_t mpi_buffer_size, local_buffer_size;
  required_buffer_sizes (num_2d_fields, num_3d_fields, mpi_buffer_size, local_buffer_size);

  return (mpi_buffer_size<=m_mpi_buffer_size) && (local_buffer_size<=m_local_buffer_size);
}

void BuffersManager::allocate_buffers ()
{
  // If views are marked as valid, they are already allocated, and no other
  // customer has requested a larger size
  if (m_views_are_valid) {
    return;
  }

  // The buffers used for packing/unpacking
  m_send_buffer  = ExecViewManaged<Real*>("send buffer",  m_mpi_buffer_size);
  m_recv_buffer  = ExecViewManaged<Real*>("recv buffer",  m_mpi_buffer_size);
  m_local_buffer = ExecViewManaged<Real*>("local buffer", m_local_buffer_size);

  // The buffers used in MPI calls
  m_mpi_send_buffer = Kokkos::create_mirror_view(decltype(m_mpi_send_buffer)::execution_space(),m_send_buffer);
  m_mpi_recv_buffer = Kokkos::create_mirror_view(decltype(m_mpi_recv_buffer)::execution_space(),m_recv_buffer);

  m_views_are_valid = true;

  // Tell to all our customers that they need to redo the setup of the internal buffer views
  for (auto& be_ptr : m_customers_be) {
    // Invalidate buffer views and requests in the customer (if none built yet, it's a no-op)
    be_ptr->clear_buffer_views_and_requests ();

    // Build buffer views and requests in the customer.
    // NOTE: if the registration is not completed yet, they will be built when
    //       registration_completed is called
    be_ptr->build_buffer_views_and_requests ();
  }
}

void BuffersManager::add_customer (BoundaryExchange* add_me)
{
  // We don't allow null customers (although this should never happen)
  assert (add_me!=nullptr);

  // We also don't allow re-registration
  assert (std::find(m_customers_be.begin(),m_customers_be.end(),add_me)==m_customers_be.end());

  // Add to the list of customers
  m_customers_be.push_back(add_me);

  // If this customer has already started the registration, we can already update the buffers sizes
  update_requested_sizes(add_me->get_num_2d_fields(),add_me->get_num_3d_fields());
}

void BuffersManager::remove_customer (BoundaryExchange* remove_me)
{
  // We don't allow null customers (although this should never happen)
  assert (remove_me!=nullptr);

  // Find the customer
  auto it = std::find(m_customers_be.begin(),m_customers_be.end(),remove_me);

  // We don't allow removal of non-customers
  assert (it!=m_customers_be.end());

  // Remove the customer
  m_customers_be.erase(it);
}

void BuffersManager::update_requested_sizes (const int num_2d_fields, const int num_3d_fields)
{
  // Make sure connectivity is valid
  assert (m_connectivity && m_connectivity->is_finalized());

  // Compute the requested buffers sizes and compare with stored ones
  size_t mpi_buffer_size, local_buffer_size;
  required_buffer_sizes (num_2d_fields, num_3d_fields, mpi_buffer_size, local_buffer_size);

  if (mpi_buffer_size>m_mpi_buffer_size) {
    m_mpi_buffer_size = mpi_buffer_size;
    m_views_are_valid = false;
  }

  if(local_buffer_size>m_local_buffer_size) {
    m_local_buffer_size = local_buffer_size;
    m_views_are_valid = false;
  }
}

void BuffersManager::required_buffer_sizes (const int num_2d_fields, const int num_3d_fields,
                                            size_t& mpi_buffer_size, size_t& local_buffer_size) const
{
  mpi_buffer_size = local_buffer_size = 0;

  // The buffer size for each connection kind
  int elem_buf_size[2];
  elem_buf_size[etoi(ConnectionKind::CORNER)] = (num_2d_fields + num_3d_fields*NUM_LEV*VECTOR_SIZE) * 1;
  elem_buf_size[etoi(ConnectionKind::EDGE)]   = (num_2d_fields + num_3d_fields*NUM_LEV*VECTOR_SIZE) * NP;

  // Compute the requested buffers sizes and compare with stored ones
  mpi_buffer_size += elem_buf_size[etoi(ConnectionKind::CORNER)] * m_connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::SHARED,ConnectionKind::CORNER);
  mpi_buffer_size += elem_buf_size[etoi(ConnectionKind::EDGE)]   * m_connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::SHARED,ConnectionKind::EDGE);

  local_buffer_size += elem_buf_size[etoi(ConnectionKind::CORNER)] * m_connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::LOCAL,ConnectionKind::CORNER);
  local_buffer_size += elem_buf_size[etoi(ConnectionKind::EDGE)]   * m_connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::LOCAL,ConnectionKind::EDGE);
}

} // namespace Homme
