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

// Forward declaration
class BuffersManager;

// The main class, handling the pack/exchange/unpack process
class BoundaryExchange
{
public:

  BoundaryExchange();
  BoundaryExchange(std::shared_ptr<Connectivity> connectivity, std::shared_ptr<BuffersManager> buffers_manager);

  // Thou shall not copy this class
  BoundaryExchange(const BoundaryExchange& src) = delete;

  ~BoundaryExchange();

  // Set the connectivity if default constructor was used
  void set_connectivity (std::shared_ptr<Connectivity> connectivity);

  // Set the buffers manager (registration must not be completed)
  void set_buffers_manager (std::shared_ptr<BuffersManager> buffers_manager);

  // These number refers to *scalar* fields. A 2-vector field counts as 2 fields.
  void set_num_fields (int num_3d_fields, int num_2d_fields);

  // Clean up MPI stuff and registered fields (but leaves connectivity and buffers manager)
  void clean_up ();

  // Check whether fields registration has already started/finished
  bool is_registration_started   () const { return m_registration_started;   }
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

  // Initialize the buffers, and the MPI data types
  void registration_completed();

  // Exchange all registered fields
  void exchange ();

  // Get the number of 2d/3d fields that this object handles
  int get_num_2d_fields () const { return m_num_2d_fields; }
  int get_num_3d_fields () const { return m_num_3d_fields; }

  template<typename ptr_type,typename raw_type>
  struct Pointer {

    Pointer& operator= (const ptr_type& p) { ptr = p; return *this; }

    KOKKOS_FORCEINLINE_FUNCTION
    raw_type& operator[] (int i) { return ptr[i]; }

    KOKKOS_FORCEINLINE_FUNCTION
    ptr_type get() { return ptr; }

    ptr_type ptr;
  };

  void pack_and_send ();
  void recv_and_unpack ();

private:

  // Make BuffersManager a friend, so it can call the methods underneath
  friend class BuffersManager;

  void clear_buffer_views_and_requests ();
  void build_buffer_views_and_requests ();

  std::shared_ptr<Connectivity>   m_connectivity;

  int                       m_elem_buf_size[2];
  MPI_Datatype              m_mpi_data_type[2];

  std::vector<MPI_Request>  m_send_requests;
  std::vector<MPI_Request>  m_recv_requests;

  ExecViewManaged<ExecViewManaged<Real[NP][NP]>**>                   m_2d_fields;
  ExecViewManaged<ExecViewManaged<Scalar[NP][NP][NUM_LEV]>**>        m_3d_fields;

  // This class contains all the buffers to be stuffed in the buffers views, and used in pack/unpack,
  // as well as the mpi buffers used in MPI calls (which are the same as the former if MPIMemSpace=ExecMemSpace),
  // and the blackhole buffers (used for missing connections)
  std::shared_ptr<BuffersManager> m_buffers_manager;

  // These are the dummy send/recv buffers used for missing connections
  ExecViewManaged<Real[NUM_LEV*VECTOR_SIZE]>    m_blackhole_send;
  ExecViewManaged<Real[NUM_LEV*VECTOR_SIZE]>    m_blackhole_recv;

  // These views can look quite complicated. Basically, we want something like
  // send_buffer(ielem,ifield,iedge) to point to the right area of one of the
  // three buffers in the buffers manager. In particular, if it is a local connection, it will
  // point to local_buffer (on both send and recv views), while for shared
  // connection, it will point to the corresponding mpi buffer, and for missing
  // connection, it will point to the send/recv blackhole.

  ExecViewManaged<ExecViewUnmanaged<Real*>**[NUM_CONNECTIONS]>               m_send_2d_buffers;
  ExecViewManaged<ExecViewUnmanaged<Real*>**[NUM_CONNECTIONS]>               m_recv_2d_buffers;

  ExecViewManaged<ExecViewUnmanaged<Scalar*[NUM_LEV]>**[NUM_CONNECTIONS]>    m_send_3d_buffers;
  ExecViewManaged<ExecViewUnmanaged<Scalar*[NUM_LEV]>**[NUM_CONNECTIONS]>    m_recv_3d_buffers;

  // The number of registered fields
  int         m_num_2d_fields;
  int         m_num_3d_fields;

  // The following flags are used to ensure that a bad user does not call setup/cleanup/registration
  // methods of this class in an order that generate errors. And if he/she does, we try to avoid errors.
  bool        m_registration_started;
  bool        m_registration_completed;
  bool        m_buffer_views_and_requests_built;
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
    Kokkos::parallel_for(MDRangePolicy<ExecSpace,2>({0,0},{m_connectivity->get_num_elements(),num_dims},{1,1}),
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
    Kokkos::parallel_for(MDRangePolicy<ExecSpace,2>({0,0},{m_connectivity->get_num_elements(),num_dims},{1,1}),
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
    Kokkos::parallel_for(MDRangePolicy<ExecSpace,2>({0,0},{m_connectivity->get_num_elements(),num_dims},{1,1}),
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
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,m_connectivity->get_num_elements()),
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
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,m_connectivity->get_num_elements()),
                         KOKKOS_LAMBDA(const int ie){
      l_3d_fields(ie,l_num_3d_fields) = Kokkos::subview(field,ie,ALL,ALL,ALL);
    });
  }

  ++m_num_3d_fields;
}

} // namespace Homme

#endif // HOMMEXX_BOUNDARY_EXCHANGE_HPP
