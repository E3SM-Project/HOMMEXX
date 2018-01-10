#ifndef HOMMEXX_MPI_BUFFERS_MANAGER_HPP
#define HOMMEXX_MPI_BUFFERS_MANAGER_HPP

#include "Types.hpp"

#include <vector>
#include <memory>

namespace Homme
{

// Forward declarations
class Connectivity;
class BoundaryExchange;

class BuffersManager
{
public:

  BuffersManager ();
  BuffersManager (std::shared_ptr<Connectivity> connectivity);

  // I'm not sure copying this class is a good idea.
  BuffersManager& operator= (const BuffersManager&) = delete;

  // Checks whether the connectivity is already set
  bool is_connectivity_set () const { return m_connectivity!=nullptr; }

  // Set the connectivity class
  void set_connectivity (std::shared_ptr<Connectivity> connectivity);

  // Ask the manager to re-check whether there is enough storage for all the BE's
  void check_for_reallocation ();

  // Check that the allocated views can handle the requested number of 2d/3d fields
  bool check_views_capacity (const int num_2d_fields,  const int num_3d_fields) const;

  // Allocate the buffers (overwriting possibly already allocated ones if needed)
  void allocate_buffers ();

  // Deep copy the send/recv buffer to/from the mpi_send/recv buffer
  // Note: these are no-ops if MPIMemSpace=ExecMemSpace
  void sync_send_buffer ();
  void sync_recv_buffer ();

  ExecViewManaged<Real*> get_send_buffer           () const;
  ExecViewManaged<Real*> get_recv_buffer           () const;
  ExecViewManaged<Real*> get_local_buffer          () const;
  MPIViewManaged<Real*>  get_mpi_send_buffer       () const;
  MPIViewManaged<Real*>  get_mpi_recv_buffer       () const;
  ExecViewManaged<Real*> get_blackhole_send_buffer () const;
  ExecViewManaged<Real*> get_blackhole_recv_buffer () const;

private:

  // Make BoundaryExchange a friend, so it can call the next two methods underneath
  friend class BoundaryExchange;

  // Adds/removes the given BoundaryExchange to/from the list of 'customers' of this class
  // Note: the only class that should call these methods is BoundaryExchange, so
  //       it can register/unregister itself as a customer
  void add_customer (BoundaryExchange* add_me);
  void remove_customer (BoundaryExchange* remove_me);

  // If necessary, updates buffers sizes so that there is enough storage to hold the required number of fields.
  // Note: this method does not (re)allocate views
  void update_requested_sizes (const int num_2d_fields, const int num_3d_fields);

  // Computes the required storages
  void required_buffer_sizes (const int num_2d_fields, const int num_3d_fields, size_t& mpi_buffer_size, size_t& local_buffer_size) const;

  // The size of the mpi and local buffers
  size_t m_mpi_buffer_size;
  size_t m_local_buffer_size;

  // Used to check whether user can still request different sizes
  bool m_views_are_valid;

  // Customers of this BuffersManager
  std::vector<BoundaryExchange*>  m_customers_be;

  // The connectivity (needed to allocate buffers)
  std::shared_ptr<Connectivity> m_connectivity;

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

inline ExecViewManaged<Real*>
BuffersManager::get_send_buffer () const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_send_buffer;
}

inline ExecViewManaged<Real*>
BuffersManager::get_recv_buffer () const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_recv_buffer;
}

inline ExecViewManaged<Real*>
BuffersManager::get_local_buffer () const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_local_buffer;
}

inline MPIViewManaged<Real*>
BuffersManager::get_mpi_send_buffer() const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_mpi_send_buffer;
}

inline MPIViewManaged<Real*>
BuffersManager::get_mpi_recv_buffer() const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_mpi_recv_buffer;
}

inline ExecViewManaged<Real*>
BuffersManager::get_blackhole_send_buffer () const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_blackhole_send_buffer;
}

inline ExecViewManaged<Real*>
BuffersManager::get_blackhole_recv_buffer () const
{
  // We ensure that the buffers are valid
  assert(m_views_are_valid);
  return m_blackhole_recv_buffer;
}

} // namespace Homme

#endif // HOMMEXX_MPI_BUFFERS_MANAGER_HPP
