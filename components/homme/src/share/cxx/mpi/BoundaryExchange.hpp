#ifndef HOMMEXX_BOUNDARY_EXCHANGE_HPP
#define HOMMEXX_BOUNDARY_EXCHANGE_HPP

#include "Connectivity.hpp"
#include "ConnectivityHelpers.hpp"

#include "Types.hpp"
#include "MpiHelpers.hpp"
#include "Hommexx_Debug.hpp"

#include "utilities/SubviewUtils.hpp"

#include <memory>

#include <vector>

#include <assert.h>

namespace Homme
{

// Forward declaration
class BuffersManager;

/*
 * BoundaryExchange: a class to handle the pack/exchange/unpack process
 *
 * This class (BE) takes care of exchanging the values of one or more fields in the
 * GP on the boundary of the elements with the neighboring elements. In particular,
 * it takes care of packing the values into some buffers, performing the exchange
 * (usually, this includes some MPI calls, unless running in serial mode),
 * and then unpacking the values, accumulating them into the receiving elements.
 * This process can be done for an arbitrary number of 2d fields (that is,
 * no vertical levels) and 3d fields (with vertical levels). If you have a
 * vector field of dimension DIM, you need to register DIM separate scalar
 * fields. Internally, for each input field the BE object stores a bunch of
 * separate Views, that view the input field at each element and component
 * (if vector field). When the exchange method is called, ALL the stored
 * fields are packed/exchanged/unpacked. Therefore, if you have two sets
 * of fields that need to be exchanged at different times, you need to
 * register them into two separate BE objects.
 *
 * The registration happens in three steps:
 *
 *  - a call to set_num_fields, which sets the number of 1d, 2d and 3d fields
 *    that will be exchanged. Once this method is called, it cannot be
 *    called again, unless the method clean_up is called first.
 *  - a number of calls to one of more of the register_field(...), methods,
 *    which set the fields into the BE class. You cannot register more fields
 *    than declared in the set_num_fields call. However you can, if you want,
 *    register less fields, although this scenario is not tested, and may
 *    be buggy, so you are probably better off calling set_num_fields with
 *    the actual number of fields you are going to register. Note that you
 *    are not allowed to call register_field(...) before set_num_fields.
 *    NOTE: you are NOT allowed to register 1d fields together with
 *          2d/3d fields. The idea is that 1d fields are exchanged only as
 *          min/max quantities (so they are not accumulated). Please, use
 *          two different BE objects for accumulation and for min/max.
 *  - a call to registration_completed, which ends the registration phase,
 *    and sets up all the internal structure to prepare for calls to
 *    exchange(). This method MUST be called BEFORE any call to exchange.
 *
 * This class relies on the BuffersManager (BM) class for the handling of the buffers.
 * See BuffersManager header for more info on that. As explained above,
 * you may have different BE objects, which are however never used at the
 * same time. Therefore, it makes sense to reuse the same BM for all of them.
 * For this reason, the BM can serve multiple 'customers'. The BE and BM
 * classes are linked by a provider-customer relationship. It is up to the BE
 * to register itself as a customer in the stored BM, and to unregister
 * itself before going out of scope, to make sure the BM does not keep
 * a dangling pointer to a non-existent customer. This is why BE's destructor
 * automatically removes the 'this' object from the stored BM's customers list
 * (assuming there is a stored BM, otherwise nothing happens).
 *
 * In order to work correctly, BE needs a valid Connectivity and a valid
 * BM (both stored as shared_ptr). They can be set at construction time
 * or later, via a setter method. There are only a few rules:
 *
 *  - once one is set, you cannot reset it
 *  - the Connectivity must be set BEFORE any call to set_num_fields
 *  - the BM must be set BEFORE any call to registration_completed
 *
 */

class BoundaryExchange
{
public:

  BoundaryExchange();
  BoundaryExchange(std::shared_ptr<Connectivity> connectivity, std::shared_ptr<BuffersManager> buffers_manager);

  // Thou shall not copy this class
  BoundaryExchange(const BoundaryExchange&) = delete;
  BoundaryExchange& operator= (const BoundaryExchange&) = delete;

  ~BoundaryExchange();

  // Set the connectivity if default constructor was used
  void set_connectivity (std::shared_ptr<Connectivity> connectivity);

  // Set the buffers manager (registration must not be completed)
  void set_buffers_manager (std::shared_ptr<BuffersManager> buffers_manager);

  // These number refers to *scalar* fields. A 2-vector field counts as 2 fields.
  void set_num_fields (const int num_1d_fields, const int num_2d_fields, const int num_3d_fields);

  // Clean up MPI stuff and registered fields (but leaves connectivity and buffers manager)
  void clean_up ();

  // Check whether fields registration has already started/finished
  bool is_registration_started   () const { return m_registration_started;   }
  bool is_registration_completed () const { return m_registration_completed; }

  // Note: num_dims is the # of dimensions to exchange, while start_dim is the first to exchange
  template<int DIM, typename... Properties>
  void register_field (ExecView<Real[DIM][NP][NP], Properties...> field, int ie, int num_dims, int start_dim);
  template<int DIM, typename... Properties>
  void register_field (ExecView<Real*[DIM][NP][NP], Properties...> field, int num_dims, int start_dim);
  template<int DIM, typename... Properties>
  void register_field (ExecView<Scalar[DIM][NP][NP][NUM_LEV], Properties...> field, int ie, int num_dims, int start_dim);
  template<int DIM, typename... Properties>
  void register_field (ExecView<Scalar*[DIM][NP][NP][NUM_LEV], Properties...> field, int num_dims, int start_dim);

  // Note: the outer dimension MUST be sliced, while the inner dimension can be fully exchanged
  template<int OUTER_DIM, int DIM, typename... Properties>
  void register_field (ExecView<Scalar[OUTER_DIM][DIM][NP][NP][NUM_LEV], Properties...> field, int ie, int idim_out, int num_dims, int start_dim);
  template<int OUTER_DIM, int DIM, typename... Properties>
  void register_field (ExecView<Scalar*[OUTER_DIM][DIM][NP][NP][NUM_LEV], Properties...> field, int idim_out, int num_dims, int start_dim);

  template<typename... Properties>
  void register_field (ExecView<Real[NP][NP], Properties...> field, int ie);
  template<typename... Properties>
  void register_field (ExecView<Real*[NP][NP], Properties...> field);
  template<typename... Properties>
  void register_field (ExecView<Scalar[NP][NP][NUM_LEV], Properties...> field, int ie);
  template<typename... Properties>
  void register_field (ExecView<Scalar*[NP][NP][NUM_LEV], Properties...> field);

  // This registration method should be used for the exchange of min/max fields
  template<typename... Properties>
  void register_min_max_fields (ExecView<Scalar[2][NUM_LEV], Properties...> field_min_max, int ie);

  // Size the buffers, and initialize the MPI types
  void registration_completed();

  // Exchange all registered 2d and 3d fields
  void exchange ();
  void exchange (ExecViewUnmanaged<const Real * [NP][NP]> rspheremp);

  // Exchange all registered 1d fields, performing min/max operations with neighbors
  void exchange_min_max ();

  // Get the number of 2d/3d fields that this object handles
  int get_num_1d_fields () const { return m_num_1d_fields; }
  int get_num_2d_fields () const { return m_num_2d_fields; }
  int get_num_3d_fields () const { return m_num_3d_fields; }

  template<typename ptr_type, typename raw_type>
  struct Pointer {

    Pointer& operator= (const ptr_type& p) { ptr = p; return *this; }

    KOKKOS_FORCEINLINE_FUNCTION
    raw_type& operator[] (int i) { return ptr[i]; }

    KOKKOS_FORCEINLINE_FUNCTION
    ptr_type get() { return ptr; }

    ptr_type ptr;
  };

  // Perform the pack_and_send and recv_and_unpack for boundary exchange of 2d/3d fields
  void pack_and_send ();
  void recv_and_unpack ();

  // Perform the pack_and_send and recv_and_unpack for min/max boundary exchange of 1d fields
  void pack_and_send_min_max ();
  void recv_and_unpack_min_max ();

  // If you are really not sure whether we are still transmitting, you can make sure we're done by calling this
  void waitall ();

private:

  short int m_exchange_type;

  // Make BuffersManager a friend, so it can call the method underneath
  friend class BuffersManager;
  void clear_buffer_views_and_requests ();

  void build_buffer_views_and_requests ();

  int max_num_registered_fields (HostViewManaged<int*> num_fields_per_elem);
  int min_num_registered_fields (HostViewManaged<int*> num_fields_per_elem);

  std::shared_ptr<Connectivity>   m_connectivity;

  int                       m_elem_buf_size[2];

  std::vector<MPI_Request>  m_send_requests;
  std::vector<MPI_Request>  m_recv_requests;

  ExecViewManaged<ExecViewUnmanaged<Scalar[NUM_LEV]>**[2]>            m_1d_fields;
  ExecViewManaged<ExecViewUnmanaged<Real[NP][NP]>**>                  m_2d_fields;
  ExecViewManaged<ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>**>       m_3d_fields;

  // This class contains all the buffers to be stuffed in the buffers views, and used in pack/unpack,
  // as well as the mpi buffers used in MPI calls (which are the same as the former if MPIMemSpace=ExecMemSpace),
  // and the blackhole buffers (used for missing connections)
  std::shared_ptr<BuffersManager> m_buffers_manager;

  // These views can look quite complicated. Basically, we want something like
  // send_buffer(ielem, ifield, iedge) to point to the right area of one of the
  // three buffers in the buffers manager. In particular, if it is a local connection, it will
  // point to local_buffer (on both send and recv views), while for shared
  // connection, it will point to the corresponding mpi buffer, and for missing
  // connection, it will point to the send/recv blackhole.

  ExecViewManaged<ExecViewUnmanaged<Scalar[NUM_LEV][2]>**[NUM_CONNECTIONS]>  m_send_1d_buffers;
  ExecViewManaged<ExecViewUnmanaged<Scalar[NUM_LEV][2]>**[NUM_CONNECTIONS]>  m_recv_1d_buffers;

  ExecViewManaged<ExecViewUnmanaged<Real*>**[NUM_CONNECTIONS]>               m_send_2d_buffers;
  ExecViewManaged<ExecViewUnmanaged<Real*>**[NUM_CONNECTIONS]>               m_recv_2d_buffers;

  ExecViewManaged<ExecViewUnmanaged<Scalar*[NUM_LEV]>**[NUM_CONNECTIONS]>    m_send_3d_buffers;
  ExecViewManaged<ExecViewUnmanaged<Scalar*[NUM_LEV]>**[NUM_CONNECTIONS]>    m_recv_3d_buffers;

  // The number of registered fields
  int         m_num_1d_fields;    // Without counting the 2x factor due to min/max fields
  int         m_num_2d_fields;
  int         m_num_3d_fields;

  HostViewManaged<int*> m_num_1d_fields_per_elem;
  HostViewManaged<int*> m_num_2d_fields_per_elem;
  HostViewManaged<int*> m_num_3d_fields_per_elem;

  // The following flags are used to ensure that a bad user does not call setup/cleanup/registration
  // methods of this class in an order that generate errors. And if he/she does, we try to avoid errors.
  bool        m_registration_started;
  bool        m_registration_completed;
  bool        m_buffer_views_and_requests_built;
  bool        m_cleaned_up;
  bool        m_send_pending;
  bool        m_recv_pending;

  int         m_num_elems;

  void init_slot_idx_to_elem_conn_pair(
    std::vector<int>& h_slot_idx_to_elem_conn_pair,
    std::vector<int>& pids, std::vector<int>& pids_os);
  void free_requests();
  // Only the impl knows about the raw pointer.
  void exchange(const ExecViewUnmanaged<const Real * [NP][NP]>* rspheremp);
public: // This is semantically private but must be public for nvcc.
  void recv_and_unpack(const ExecViewUnmanaged<const Real * [NP][NP]>* rspheremp);
};

// ============================ REGISTER METHODS ========================= //

template<int DIM, typename... Properties>
void BoundaryExchange::register_field (ExecView<Real[DIM][NP][NP], Properties...> field, int ie, int num_dims, int start_dim)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (ie>=0 && ie < m_2d_fields.extent_int(0));
  assert (num_dims>0 && start_dim>=0 && DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_2d_fields_per_elem(ie)+num_dims<=m_2d_fields.extent_int(1));
  assert (max_num_registered_fields(m_num_1d_fields_per_elem)==0);

  auto h_2d_fields = Kokkos::create_mirror_view(m_2d_fields);
  Kokkos::deep_copy(h_2d_fields,m_2d_fields);
  for(int idim=0; idim<num_dims; ++idim) {
    h_2d_fields(ie, m_num_2d_fields_per_elem(ie)+idim) = Homme::subview(field, start_dim+idim);
  }
  Kokkos::deep_copy(m_2d_fields,h_2d_fields);

  m_num_2d_fields_per_elem(ie) += num_dims;
}

template<int DIM, typename... Properties>
void BoundaryExchange::register_field (ExecView<Real*[DIM][NP][NP], Properties...> field, int num_dims, int start_dim)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (max_num_registered_fields(m_num_2d_fields_per_elem)+num_dims<=m_2d_fields.extent_int(1));
  assert (max_num_registered_fields(m_num_1d_fields_per_elem)==0);
  // Don't register one field on a few elements, then pause, register another field on all elements, then go
  // back to finish registering the first field on the remaining elements. When you register a field one
  // element at a time, please, complete all elements before moving to the next field
  assert (max_num_registered_fields(m_num_2d_fields_per_elem)==min_num_registered_fields(m_num_2d_fields_per_elem));

  auto h_2d_fields = Kokkos::create_mirror_view(m_2d_fields);
  Kokkos::deep_copy(h_2d_fields,m_2d_fields);
  for (int ie=0; ie<m_connectivity->get_num_local_elements(); ++ie) {
    for (int idim=0; idim<num_dims; ++idim) {
      h_2d_fields(ie,m_num_2d_fields_per_elem(ie)+idim) = Homme::subview(field, ie, start_dim+idim);
    }
    m_num_2d_fields_per_elem(ie) += num_dims;
  }
  Kokkos::deep_copy(m_2d_fields,h_2d_fields);
}

template<int DIM, typename... Properties>
void BoundaryExchange::register_field (ExecView<Scalar[DIM][NP][NP][NUM_LEV], Properties...> field, int ie, int num_dims, int start_dim)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (ie>=0 && ie < m_3d_fields.extent_int(0));
  assert (num_dims>0 && start_dim>=0 && DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_3d_fields_per_elem(ie)+num_dims<=m_3d_fields.extent_int(1));
  assert (max_num_registered_fields(m_num_1d_fields_per_elem)==0);

  auto h_3d_fields = Kokkos::create_mirror_view(m_3d_fields);
  Kokkos::deep_copy(h_3d_fields,m_3d_fields);
  for(int idim=0; idim<num_dims; ++idim) {
    h_3d_fields(ie, m_num_3d_fields_per_elem(ie)+idim) = Homme::subview(field, start_dim+idim);
  }
  Kokkos::deep_copy(m_3d_fields,h_3d_fields);

  m_num_3d_fields_per_elem(ie) += num_dims;
}

template<int DIM, typename... Properties>
void BoundaryExchange::register_field (ExecView<Scalar*[DIM][NP][NP][NUM_LEV], Properties...> field, int num_dims, int start_dim)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (max_num_registered_fields(m_num_3d_fields_per_elem)+num_dims<=m_3d_fields.extent_int(1));
  assert (max_num_registered_fields(m_num_1d_fields_per_elem)==0);
  // Don't register one field on a few elements, then pause, register another field on all elements, then go
  // back to finish registering the first field on the remaining elements. When you register a field one
  // element at a time, please, complete all elements before moving to the next field
  assert (max_num_registered_fields(m_num_3d_fields_per_elem)==min_num_registered_fields(m_num_3d_fields_per_elem));

  auto h_3d_fields = Kokkos::create_mirror_view(m_3d_fields);
  Kokkos::deep_copy(h_3d_fields,m_3d_fields);
  for (int ie=0; ie<m_connectivity->get_num_local_elements(); ++ie) {
    for (int idim=0; idim<num_dims; ++idim) {
      h_3d_fields(ie,m_num_3d_fields_per_elem(ie)+idim) = Homme::subview(field, ie, start_dim+idim);
    }
    m_num_3d_fields_per_elem(ie) += num_dims;
  }
  Kokkos::deep_copy(m_3d_fields,h_3d_fields);
}

template<int OUTER_DIM, int DIM, typename... Properties>
void BoundaryExchange::register_field (ExecView<Scalar[OUTER_DIM][DIM][NP][NP][NUM_LEV], Properties...> field, int ie, int outer_dim, int num_dims, int start_dim)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (ie>=0 && ie < m_3d_fields.extent_int(0));
  assert (num_dims>0 && start_dim>=0 && outer_dim>=0 && DIM>0 && OUTER_DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_3d_fields_per_elem(ie)+num_dims<=m_3d_fields.extent_int(1));
  assert (max_num_registered_fields(m_num_1d_fields_per_elem)==0);

  auto h_3d_fields = Kokkos::create_mirror_view(m_3d_fields);
  Kokkos::deep_copy(h_3d_fields,m_3d_fields);
  for(int idim=0; idim<num_dims; ++idim) {
    h_3d_fields(ie, m_num_3d_fields_per_elem(ie)+idim) = Homme::subview(field, outer_dim, start_dim+idim);
  }
  Kokkos::deep_copy(m_3d_fields,h_3d_fields);

  m_num_3d_fields_per_elem(ie) += num_dims;
}

template<int OUTER_DIM, int DIM, typename... Properties>
void BoundaryExchange::register_field (ExecView<Scalar*[OUTER_DIM][DIM][NP][NP][NUM_LEV], Properties...> field, int outer_dim, int num_dims, int start_dim)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && outer_dim>=0 && DIM>0 && OUTER_DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (max_num_registered_fields(m_num_3d_fields_per_elem)+num_dims<=m_3d_fields.extent_int(1));
  assert (max_num_registered_fields(m_num_1d_fields_per_elem)==0);

  // Don't register one field on a few elements, then pause, register another field on all elements, then go
  // back to finish registering the first field on the remaining elements. When you register a field one
  // element at a time, please, complete all elements before moving to the next field
  assert (max_num_registered_fields(m_num_3d_fields_per_elem)==min_num_registered_fields(m_num_3d_fields_per_elem));

  auto h_3d_fields = Kokkos::create_mirror_view(m_3d_fields);
  Kokkos::deep_copy(h_3d_fields,m_3d_fields);
  for (int ie=0; ie<m_connectivity->get_num_local_elements(); ++ie) {
    for (int idim=0; idim<num_dims; ++idim) {
      h_3d_fields(ie,m_num_3d_fields_per_elem(ie)+idim) = Homme::subview(field, ie, outer_dim, start_dim+idim);
    }
    m_num_3d_fields_per_elem(ie) += num_dims;
  }
  Kokkos::deep_copy(m_3d_fields,h_3d_fields);
}

template<typename... Properties>
void BoundaryExchange::register_field (ExecView<Real[NP][NP], Properties...> field, int ie)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (ie>=0 && ie < m_2d_fields.extent_int(0));
  assert (m_num_2d_fields_per_elem(ie)+1<=m_2d_fields.extent_int(1));
  assert (max_num_registered_fields(m_num_1d_fields_per_elem)==0);

  auto h_2d_fields = Kokkos::create_mirror_view(m_2d_fields);
  Kokkos::deep_copy(h_2d_fields,m_2d_fields);
  h_2d_fields(ie, m_num_2d_fields_per_elem(ie)) = field;
  Kokkos::deep_copy(m_2d_fields,h_2d_fields);

  ++m_num_2d_fields_per_elem(ie);
}

template<typename... Properties>
void BoundaryExchange::register_field (ExecView<Real*[NP][NP], Properties...> field)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (max_num_registered_fields(m_num_2d_fields_per_elem)+1<=m_2d_fields.extent_int(1));
  assert (max_num_registered_fields(m_num_1d_fields_per_elem)==0);
  // Don't register one field on a few elements, then pause, register another field on all elements, then go
  // back to finish registering the first field on the remaining elements. When you register a field one
  // element at a time, please, complete all elements before moving to the next field
  assert (max_num_registered_fields(m_num_2d_fields_per_elem)==min_num_registered_fields(m_num_2d_fields_per_elem));

  auto h_2d_fields = Kokkos::create_mirror_view(m_2d_fields);
  Kokkos::deep_copy(h_2d_fields,m_2d_fields);
  for (int ie=0; ie<m_connectivity->get_num_local_elements(); ++ie) {
    h_2d_fields(ie,m_num_2d_fields_per_elem(ie)) = Homme::subview(field, ie);
    ++m_num_2d_fields_per_elem(ie);
  }
  Kokkos::deep_copy(m_2d_fields,h_2d_fields);
}

template<typename... Properties>
void BoundaryExchange::register_field (ExecView<Scalar[NP][NP][NUM_LEV], Properties...> field, int ie)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (ie>=0 && ie < m_3d_fields.extent_int(0));
  assert (m_num_3d_fields_per_elem(ie)+1<=m_3d_fields.extent_int(1));
  assert (max_num_registered_fields(m_num_1d_fields_per_elem)==0);

  auto h_3d_fields = Kokkos::create_mirror_view(m_3d_fields);
  Kokkos::deep_copy(h_3d_fields,m_3d_fields);
  h_3d_fields(ie, m_num_3d_fields_per_elem(ie)) = field;
  Kokkos::deep_copy(m_3d_fields,h_3d_fields);

  ++m_num_3d_fields_per_elem(ie);
}

template<typename... Properties>
void BoundaryExchange::register_field (ExecView<Scalar*[NP][NP][NUM_LEV], Properties...> field)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (max_num_registered_fields(m_num_3d_fields_per_elem)+1<=m_3d_fields.extent_int(1));
  assert (max_num_registered_fields(m_num_1d_fields_per_elem)==0);
  // Don't register one field on a few elements, then pause, register another field on all elements, then go
  // back to finish registering the first field on the remaining elements. When you register a field one
  // element at a time, please, complete all elements before moving to the next field
  assert (max_num_registered_fields(m_num_3d_fields_per_elem)==min_num_registered_fields(m_num_3d_fields_per_elem));

  auto h_3d_fields = Kokkos::create_mirror_view(m_3d_fields);
  Kokkos::deep_copy(h_3d_fields,m_3d_fields);
  for (int ie=0; ie<m_connectivity->get_num_local_elements(); ++ie) {
    h_3d_fields(ie,m_num_3d_fields_per_elem(ie)) = Homme::subview(field, ie);
    ++m_num_3d_fields_per_elem(ie);
  }
  Kokkos::deep_copy(m_3d_fields,h_3d_fields);
}

template <typename... Properties>
void BoundaryExchange::register_min_max_fields(
    ExecView<Scalar[2][NUM_LEV], Properties...> field_min_max, int ie) {
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (ie>=0 && ie < m_1d_fields.extent_int(0));
  assert (m_num_1d_fields_per_elem(ie)+1<=m_1d_fields.extent_int(1));
  assert (max_num_registered_fields(m_num_2d_fields_per_elem) == 0 &&
          max_num_registered_fields(m_num_3d_fields_per_elem) == 0);

  auto h_1d_fields = Kokkos::create_mirror_view(m_1d_fields);
  Kokkos::deep_copy(h_1d_fields,m_1d_fields);
  h_1d_fields(ie,m_num_1d_fields_per_elem(ie),etoi(MAX_ID)) = Homme::subview(field_min_max, etoi(MAX_ID));
  h_1d_fields(ie,m_num_1d_fields_per_elem(ie),etoi(MIN_ID)) = Homme::subview(field_min_max, etoi(MIN_ID));
  Kokkos::deep_copy(m_1d_fields,h_1d_fields);
  ++m_num_1d_fields_per_elem(ie);
}

} // namespace Homme

#endif // HOMMEXX_BOUNDARY_EXCHANGE_HPP
