#ifndef HOMMEXX_BOUNDARY_EXCHANGE_HPP
#define HOMMEXX_BOUNDARY_EXCHANGE_HPP

#include "Connectivity.hpp"
#include "ConnectivityHelpers.hpp"

#include "Utility.hpp"
#include "Types.hpp"
#include "Hommexx_Debug.hpp"

#include <memory>

#include <vector>
#include <map>

#include <assert.h>

namespace Homme
{

class BoundaryExchange
{
public:

  BoundaryExchange();
  BoundaryExchange(const Connectivity& connectivity);
  ~BoundaryExchange();

  // These number refers to *scalar* fields. A 2-vector field counts as 2 fields.
  void set_num_fields (int num_3d_fields, int num_2d_fields);

  // Clean up MPI stuff and registered fields
  void clean_up ();

  // Check whether fields have already been registered
  bool is_registration_completed () const { return m_registration_completed; }

  // Note: num_dims is the # of dimensions to exchange, while idim is the first to exchange
  template<int DIM, typename... Properties>
  void register_field (HostView<Real*[DIM][NP][NP],Properties...> field, int num_dims, int idim);
  template<int DIM, typename... Properties>
  void register_field (HostView<Scalar*[DIM][NP][NP][NUM_LEV],Properties...> field, int num_dims, int idim);

  // Note: the outer dimension MUST be sliced, while the inner dimension can be fully exchanged
  template<int OUTER_DIM, int DIM, typename... Properties>
  void register_field (HostView<Scalar*[OUTER_DIM][DIM][NP][NP][NUM_LEV],Properties...> field, int idim_out, int num_dims, int idim);

  template<typename... Properties>
  void register_field (HostView<Real*[NP][NP],Properties...> field);
  template<typename... Properties>
  void register_field (HostView<Scalar*[NP][NP][NUM_LEV],Properties...> field);

  // Initialize the window, the buffers, and the MPI data types
  void registration_completed();

  // Exchange all registered fields
  void exchange ();

private:

  void build_requests ();
  void pack_and_send ();
  void recv_and_unpack ();

  HostViewManaged<HostViewUnmanaged<Real[NP][NP]>**>                   m_2d_fields;
  HostViewManaged<HostViewUnmanaged<Scalar[NP][NP][NUM_LEV]>**>        m_3d_fields;

  // These views can look quite complicated. Basically, we want something like
  // edge_buffer(ielem,ifield,iedge) to point to the right area of one of the
  // three buffers above. In particular, if it is a local connection, it will
  // point to m_local_buffer (on both send and recv views), while for shared
  // connection, it will point to the corresponding mpi buffer, and for missing
  // connection, it will point to the send/recv blackhole.

  HostViewManaged<HostViewUnmanaged<Real*>**[NUM_CONNECTIONS]>               m_send_2d_buffers;
  HostViewManaged<HostViewUnmanaged<Real*>**[NUM_CONNECTIONS]>               m_recv_2d_buffers;

  HostViewManaged<HostViewUnmanaged<Scalar*[NUM_LEV]>**[NUM_CONNECTIONS]>    m_send_3d_buffers;
  HostViewManaged<HostViewUnmanaged<Scalar*[NUM_LEV]>**[NUM_CONNECTIONS]>    m_recv_3d_buffers;

  // The number of registered fields
  int         m_num_2d_fields;
  int         m_num_3d_fields;

  // The following flags are used to ensure that a bad user does not call setup/cleanup/registration
  // methods of this class in an order that generate errors. And if he/she does, we try to avoid errors.
  bool        m_registration_started;
  bool        m_registration_completed;
  bool        m_cleaned_up;

  const Comm&         m_comm;

  const Connectivity& m_connectivity;

  int                 m_num_elements;

  int                       m_elem_buf_size[2];

  MPI_Datatype              m_mpi_data_type[2];

  std::vector<MPI_Request>  m_send_requests;
  std::vector<MPI_Request>  m_recv_requests;

  struct mpi_deleter_wrapper {
    void operator() (Real* ptr) {
      HOMMEXX_MPI_CHECK_ERROR(MPI_Free_mem(ptr));
    }
  };

  using mpi_handled_ptr = std::unique_ptr<Real[],mpi_deleter_wrapper>;

  mpi_handled_ptr               m_send_buffer;
  mpi_handled_ptr               m_recv_buffer;
  std::unique_ptr<Real[]>       m_local_buffer;

  Real         m_blackhole_send[NUM_LEV*VECTOR_SIZE];
  Real         m_blackhole_recv[NUM_LEV*VECTOR_SIZE];

};

BoundaryExchange& get_boundary_exchange(const std::string& be_name);
std::map<std::string,BoundaryExchange>& get_all_boundary_exchange();

// ============================ REGISTER METHODS ========================= //

template<int DIM, typename... Properties>
void BoundaryExchange::register_field (HostView<Real*[DIM][NP][NP],Properties...> field, int num_dims, int start_dim)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_2d_fields+num_dims<=m_2d_fields.extent_int(1));

  for (int ie=0; ie<m_num_elements; ++ie) {
    for (int idim=0; idim<num_dims; ++idim) {
      m_2d_fields(ie,m_num_2d_fields+idim) = Homme::subview(field,ie,start_dim+idim);
    }
  }

  m_num_2d_fields += num_dims;
}

template<int DIM, typename... Properties>
void BoundaryExchange::register_field (HostView<Scalar*[DIM][NP][NP][NUM_LEV],Properties...> field, int num_dims, int start_dim)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_3d_fields+num_dims<=m_3d_fields.extent_int(1));

  for (int ie=0; ie<m_num_elements; ++ie) {
    for (int idim=0; idim<num_dims; ++idim) {
      m_3d_fields(ie,m_num_3d_fields+idim) = Homme::subview(field,ie,start_dim+idim);
    }
  }

  m_num_3d_fields += num_dims;
}

template<int OUTER_DIM, int DIM, typename... Properties>
void BoundaryExchange::register_field (HostView<Scalar*[OUTER_DIM][DIM][NP][NP][NUM_LEV],Properties...> field, int outer_dim, int num_dims, int start_dim)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && outer_dim>=0 && DIM>0 && OUTER_DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_3d_fields+num_dims<=m_3d_fields.extent_int(1));

  for (int ie=0; ie<m_num_elements; ++ie) {
    for (int idim=0; idim<num_dims; ++idim) {
      m_3d_fields(ie,m_num_3d_fields+idim) = Homme::subview(field,ie,outer_dim,start_dim+idim);
    }
  }

  m_num_3d_fields += num_dims;
}

template<typename... Properties>
void BoundaryExchange::register_field (HostView<Real*[NP][NP],Properties...> field)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (m_num_2d_fields+1<=m_2d_fields.extent_int(1));

  for (int ie=0; ie<m_num_elements; ++ie) {
    m_2d_fields(ie,m_num_2d_fields) = Homme::subview(field,ie);
  }

  ++m_num_2d_fields;
}

template<typename... Properties>
void BoundaryExchange::register_field (HostView<Scalar*[NP][NP][NUM_LEV],Properties...> field)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (m_num_3d_fields+1<=m_3d_fields.extent_int(1));

  for (int ie=0; ie<m_num_elements; ++ie) {
    m_3d_fields(ie,m_num_3d_fields) = Homme::subview(field,ie);
  }

  ++m_num_3d_fields;
}

} // namespace Homme

#endif // HOMMEXX_BOUNDARY_EXCHANGE_HPP
