#ifndef HOMMEXX_BOUNDARY_EXCHANGE_HPP
#define HOMMEXX_BOUNDARY_EXCHANGE_HPP

#include "Connectivity.hpp"
#include "ConnectivityHelpers.hpp"

#include "Types.hpp"
#include "Hommexx_Debug.hpp"

#include <memory>

#include <vector>
#include <map>

#include <assert.h>

namespace Homme
{

// Forward declarations
class BoundaryExchange;

// Helper functor for packing/unpacking (pimpl idiom, defined in cpp file)
struct PackUnpackFunctor;

// The main class, handling the pack/exchange/unpack process
class BoundaryExchange
{
public:
  struct TagPack   {};
  struct TagUnpack {};

  BoundaryExchange();
  BoundaryExchange(const Connectivity& connectivity);

  // Thou shall not copy this class
  BoundaryExchange(const BoundaryExchange& src) = delete;

  ~BoundaryExchange();

  // These number refers to *scalar* fields. A 2-vector field counts as 2 fields.
  void set_num_fields (int num_3d_fields, int num_2d_fields);

  // Clean up MPI stuff and registered fields
  void clean_up ();

  // Check whether fields have already been registered
  bool is_registration_completed () const { return m_registration_completed; }

  // Note: num_dims is the # of dimensions to exchange, while idim is the first to exchange
  template<int DIM, typename... Properties>
  void register_field (ExecView<Real*[DIM][NP][NP],Properties...> field, int num_dims, int idim);
  template<int DIM, typename... Properties>
  void register_field (ExecView<Scalar*[DIM][NP][NP][NUM_LEV],Properties...> field, int num_dims, int idim);

  // Note: the outer dimension MUST be sliced, while the inner dimension can be fully exchanged
  template<int OUTER_DIM, int DIM, typename... Properties>
  void register_field (ExecView<Scalar*[OUTER_DIM][DIM][NP][NP][NUM_LEV],Properties...> field, int idim_out, int num_dims, int idim);

  template<typename... Properties>
  void register_field (ExecView<Real*[NP][NP],Properties...> field);
  template<typename... Properties>
  void register_field (ExecView<Scalar*[NP][NP][NUM_LEV],Properties...> field);

  // Initialize the window, the buffers, and the MPI data types
  void registration_completed();

  // Exchange all registered fields
  void exchange ();

  template<typename ptr_type,typename raw_type>
  struct Pointer {

    Pointer& operator= (const ptr_type& p) { ptr = p; return *this; }

    KOKKOS_FORCEINLINE_FUNCTION
    raw_type& operator[] (int i) { return ptr[i]; }

    ptr_type ptr;
  };

  friend struct PackUnpackFunctor;

private:

  void build_requests ();

  const Comm&               m_comm;
  const Connectivity        m_connectivity;

  int                       m_num_elements;

  int                       m_elem_buf_size[2];
  MPI_Datatype              m_mpi_data_type[2];

  std::vector<MPI_Request>  m_send_requests;
  std::vector<MPI_Request>  m_recv_requests;

  ExecViewManaged<ExecViewManaged<Real[NP][NP]>**>                   m_2d_fields;
  ExecViewManaged<ExecViewManaged<Scalar[NP][NP][NUM_LEV]>**>        m_3d_fields;

  // These are the raw buffers to be stuffed in the buffers views, and used in pack/unpack
  ExecViewManaged<Real*>     m_send_buffer;
  ExecViewManaged<Real*>     m_recv_buffer;
  ExecViewManaged<Real*>     m_local_buffer;

  // These are the raw buffers to be used in MPI calls
  MPIViewManaged<Real*>     m_mpi_send_buffer;
  MPIViewManaged<Real*>     m_mpi_recv_buffer;

  // These are the dummy send/recv buffers used for missing connections
  ExecViewManaged<Real[NUM_LEV*VECTOR_SIZE]>    m_blackhole_send;
  ExecViewManaged<Real[NUM_LEV*VECTOR_SIZE]>    m_blackhole_recv;

  // These views can look quite complicated. Basically, we want something like
  // send_buffer(ielem,ifield,iedge) to point to the right area of one of the
  // three buffers above. In particular, if it is a local connection, it will
  // point to m_local_buffer (on both send and recv views), while for shared
  // connection, it will point to the corresponding mpi buffer, and for missing
  // connection, it will point to the send/recv blackhole.

  ExecViewManaged<ExecViewUnmanaged<Real*>**[NUM_CONNECTIONS]>               m_send_2d_buffers;
  ExecViewManaged<ExecViewUnmanaged<Real*>**[NUM_CONNECTIONS]>               m_recv_2d_buffers;

  ExecViewManaged<ExecViewUnmanaged<Scalar*[NUM_LEV]>**[NUM_CONNECTIONS]>    m_send_3d_buffers;
  ExecViewManaged<ExecViewUnmanaged<Scalar*[NUM_LEV]>**[NUM_CONNECTIONS]>    m_recv_3d_buffers;

  // Policies and functor used for pack/unpack phases
  Kokkos::TeamPolicy<ExecSpace,TagPack>       m_pack_policy;
  Kokkos::TeamPolicy<ExecSpace,TagUnpack>     m_unpack_policy;
  std::shared_ptr<PackUnpackFunctor>          m_pack_unpack_functor;

  // The number of registered fields
  int         m_num_2d_fields;
  int         m_num_3d_fields;

  // The following flags are used to ensure that a bad user does not call setup/cleanup/registration
  // methods of this class in an order that generate errors. And if he/she does, we try to avoid errors.
  bool        m_registration_started;
  bool        m_registration_completed;
  bool        m_cleaned_up;
};

// ============================ REGISTER METHODS ========================= //

template<int DIM, typename... Properties>
void BoundaryExchange::register_field (ExecView<Real*[DIM][NP][NP],Properties...> field, int num_dims, int start_dim)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_2d_fields+num_dims<=m_2d_fields.extent_int(1));

  {
    auto l_num_2d_fields = m_num_2d_fields;
    auto l_2d_fields = m_2d_fields;
    Kokkos::parallel_for(MDRangePolicy<ExecSpace,2>({0,0},{m_num_elements,num_dims},{1,1}),
                         KOKKOS_LAMBDA(const int ie, const int idim){
        l_2d_fields(ie,l_num_2d_fields+idim) = Kokkos::subview(field,ie,start_dim+idim,ALL,ALL);
    });
  }

  m_num_2d_fields += num_dims;
}

template<int DIM, typename... Properties>
void BoundaryExchange::register_field (ExecView<Scalar*[DIM][NP][NP][NUM_LEV],Properties...> field, int num_dims, int start_dim)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_3d_fields+num_dims<=m_3d_fields.extent_int(1));

  {
    auto l_num_3d_fields = m_num_3d_fields;
    auto l_3d_fields = m_3d_fields;
    Kokkos::parallel_for(MDRangePolicy<ExecSpace,2>({0,0},{m_num_elements,num_dims},{1,1}),
                         KOKKOS_LAMBDA(const int ie, const int idim){
        l_3d_fields(ie,l_num_3d_fields+idim) = Kokkos::subview(field,ie,start_dim+idim,ALL,ALL,ALL);
    });
  }

  m_num_3d_fields += num_dims;
}

template<int OUTER_DIM, int DIM, typename... Properties>
void BoundaryExchange::register_field (ExecView<Scalar*[OUTER_DIM][DIM][NP][NP][NUM_LEV],Properties...> field, int outer_dim, int num_dims, int start_dim)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && outer_dim>=0 && DIM>0 && OUTER_DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_3d_fields+num_dims<=m_3d_fields.extent_int(1));

  {
    auto l_num_3d_fields = m_num_3d_fields;
    auto l_3d_fields = m_3d_fields;
    Kokkos::parallel_for(MDRangePolicy<ExecSpace,2>({0,0},{m_num_elements,num_dims},{1,1}),
                         KOKKOS_LAMBDA(const int ie, const int idim){
        l_3d_fields(ie,l_num_3d_fields+idim) = Kokkos::subview(field,ie,outer_dim,start_dim+idim,ALL,ALL,ALL);
    });
  }

  m_num_3d_fields += num_dims;
}

template<typename... Properties>
void BoundaryExchange::register_field (ExecView<Real*[NP][NP],Properties...> field)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (m_num_2d_fields+1<=m_2d_fields.extent_int(1));

  {
    auto l_num_2d_fields = m_num_2d_fields;
    auto l_2d_fields = m_2d_fields;
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,m_num_elements),
                         KOKKOS_LAMBDA(const int ie){
      l_2d_fields(ie,l_num_2d_fields) = Kokkos::subview(field,ie,ALL,ALL);
    });
  }

  ++m_num_2d_fields;
}

template<typename... Properties>
void BoundaryExchange::register_field (ExecView<Scalar*[NP][NP][NUM_LEV],Properties...> field)
{
  using Kokkos::ALL;

  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (m_num_3d_fields+1<=m_3d_fields.extent_int(1));

  {
    auto l_num_3d_fields = m_num_3d_fields;
    auto l_3d_fields = m_3d_fields;
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,m_num_elements),
                         KOKKOS_LAMBDA(const int ie){
      l_3d_fields(ie,l_num_3d_fields) = Kokkos::subview(field,ie,ALL,ALL,ALL);
    });
  }

  ++m_num_3d_fields;
}

} // namespace Homme

#endif // HOMMEXX_BOUNDARY_EXCHANGE_HPP
