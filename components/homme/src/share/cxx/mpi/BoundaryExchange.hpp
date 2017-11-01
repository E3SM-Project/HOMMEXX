#ifndef HOMMEXX_BOUNDARY_EXCHANGE_HPP
#define HOMMEXX_BOUNDARY_EXCHANGE_HPP

#include "Connectivity.hpp"
#include "ConnectivityHelpers.hpp"

#include "Utility.hpp"
#include "Types.hpp"
#include "Hommexx_Debug.hpp"

#include <memory>

#include <vector>

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
  template<int DIM, typename MemoryManagement>
  void register_field (HostView<Real*[DIM][NP][NP],MemoryManagement> field, int num_dims, int idim);
  template<int DIM, typename MemoryManagement>
  void register_field (HostView<Scalar*[DIM][NP][NP][NUM_LEV],MemoryManagement> field, int num_dims, int idim);

  // Note: the outer dimension MUST be sliced, while the inner dimension can be fully exchanged
  template<int OUTER_DIM, int DIM, typename MemoryManagement>
  void register_field (HostView<Scalar*[OUTER_DIM][DIM][NP][NP][NUM_LEV],MemoryManagement> field, int idim_out, int num_dims, int idim);

  template<typename MemoryManagement>
  void register_field (HostView<Real*[NP][NP],MemoryManagement> field);
  template<typename MemoryManagement>
  void register_field (HostView<Scalar*[NP][NP][NUM_LEV],MemoryManagement> field);

  // Initialize the window, the buffers, and the MPI data types
  void registration_completed();

  // Exchange all registered fields
  void exchange ();

private:

  template<typename T>
  struct Pointer {
    static constexpr size_t rank = std::rank<T>::value;
    static_assert (rank==1 || rank==2 || rank==3, "Error! Unsupported rank.\n");

    static constexpr int first_extent  = static_cast<int>(std::extent<T,0>::value);
    static constexpr int second_extent = static_cast<int>(std::extent<T,1>::value);
    static constexpr int third_extent  = static_cast<int>(std::extent<T,2>::value);

    using raw_type = typename std::remove_all_extents<T>::type;

    Pointer () : ptr(nullptr) {}
    Pointer (raw_type* src) : ptr(src) {}

    const raw_type* const data() const { return ptr; }

    const raw_type& operator[] (const int i) const
    {
      assert (rank==1);
      assert (i>=0 && i<first_extent);
      return ptr[i];
    }
    raw_type& operator[] (const int i)
    {
      assert (rank==1);
      assert (i>=0 && i<first_extent);
      return ptr[i];
    }

    const raw_type& operator() (const int i) const
    {
      assert (rank==1);
      assert (i>=0 && i<first_extent);
      return ptr[i];
    }
    raw_type& operator() (const int i)
    {
      assert (rank==1);
      assert (i>=0 && i<first_extent);
      return ptr[i];
    }

    const raw_type& operator() (const int i, const int j) const
    {
      assert (rank==2);
      assert (i>=0 && i<first_extent && j>=0 && j<second_extent);
      return ptr[i*second_extent+j];
    }
    raw_type& operator() (const int i, const int j)
    {
      assert (rank==2);
      assert (i>=0 && i<first_extent && j>=0 && j<second_extent);
      return ptr[i*second_extent+j];
    }

    const raw_type& operator() (const int i, const int j, const int k) const
    {
      assert (rank==3);
      assert (i>=0 && i<first_extent && j>=0 && j<second_extent && k>=0 && k<third_extent);
      return ptr[(i*second_extent + j)*third_extent + k];
    }
    raw_type& operator() (const int i, const int j, const int k)
    {
      assert (rank==3);
      assert (i>=0 && i<first_extent && j>=0 && j<second_extent && k>=0 && k<third_extent);
      return ptr[(i*second_extent + j)*third_extent + k];
    }

  private:
    raw_type* ptr;
  };

  void build_requests ();
  void pack_and_send ();
  void recv_and_unpack ();

  HostViewManaged<Pointer<Real[NP][NP]>**>                   m_2d_fields;
  HostViewManaged<Pointer<Scalar[NP][NP][NUM_LEV]>**>        m_3d_fields;

  // These views can look quite complicated. Basically, we want something like
  // edge_buffer(ielem,ifield,iedge) to point to the right area of one of the
  // three buffers above. In particular, if it is a local connection, it will
  // point to m_local_buffer (on both send and recv views), while for shared
  // connection, it will point to the corresponding mpi buffer.

  HostViewManaged<Pointer<Real[1]>**[NUM_CORNERS]>              m_send_2d_corners_buffers;
  HostViewManaged<Pointer<Real[NP]>**[NUM_EDGES]>               m_send_2d_edges_buffers;
  HostViewManaged<Pointer<Scalar[NUM_LEV]>**[NUM_CORNERS]>      m_send_3d_corners_buffers;
  HostViewManaged<Pointer<Scalar[NP][NUM_LEV]>**[NUM_EDGES]>    m_send_3d_edges_buffers;

  HostViewManaged<Pointer<Real[1]>**[NUM_CORNERS]>              m_recv_2d_corners_buffers;
  HostViewManaged<Pointer<Real[NP]>**[NUM_EDGES]>               m_recv_2d_edges_buffers;
  HostViewManaged<Pointer<Scalar[NUM_LEV]>**[NUM_CORNERS]>      m_recv_3d_corners_buffers;
  HostViewManaged<Pointer<Scalar[NP][NUM_LEV]>**[NUM_EDGES]>    m_recv_3d_edges_buffers;

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

  int                 m_corner_size;
  int                 m_edge_size;

  MPI_Datatype m_mpi_corner_data_type;
  MPI_Datatype m_mpi_edge_data_type;

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

  Real         m_zero[NUM_LEV*VECTOR_SIZE];

};

BoundaryExchange& get_boundary_exchange(const std::string& be_name);
std::map<std::string,BoundaryExchange>& get_all_boundary_exchange();

// ============================ REGISTER METHODS ========================= //

template<int DIM, typename MemoryManagement>
void BoundaryExchange::register_field (HostView<Real*[DIM][NP][NP],MemoryManagement> field, int num_dims, int start_dim)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_2d_fields+num_dims<=m_2d_fields.extent_int(1));

  for (int ie=0; ie<m_num_elements; ++ie) {
    for (int idim=0; idim<num_dims; ++idim) {
      m_2d_fields(ie,m_num_2d_fields+idim) = Pointer<Real[NP][NP]>(&field(ie,start_dim+idim,0,0));
    }
  }

  m_num_2d_fields += num_dims;
}

template<int DIM, typename MemoryManagement>
void BoundaryExchange::register_field (HostView<Scalar*[DIM][NP][NP][NUM_LEV],MemoryManagement> field, int num_dims, int start_dim)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_3d_fields+num_dims<=m_3d_fields.extent_int(1));

  for (int ie=0; ie<m_num_elements; ++ie) {
    for (int idim=0; idim<num_dims; ++idim) {
      m_3d_fields(ie,m_num_3d_fields+idim) = Pointer<Scalar[NP][NP][NUM_LEV]>(&field(ie,start_dim+idim,0,0,0));
    }
  }

  m_num_3d_fields += num_dims;
}

template<int OUTER_DIM, int DIM, typename MemoryManagement>
void BoundaryExchange::register_field (HostView<Scalar*[OUTER_DIM][DIM][NP][NP][NUM_LEV],MemoryManagement> field, int outer_dim, int num_dims, int start_dim)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (num_dims>0 && start_dim>=0 && outer_dim>=0 && DIM>0 && OUTER_DIM>0);
  assert (start_dim+num_dims<=DIM);
  assert (m_num_3d_fields+num_dims<=m_3d_fields.extent_int(1));

  for (int ie=0; ie<m_num_elements; ++ie) {
    for (int idim=0; idim<num_dims; ++idim) {
      m_3d_fields(ie,m_num_3d_fields+idim) = Pointer<Scalar[NP][NP][NUM_LEV]>(&field(ie,outer_dim,start_dim+idim,0,0,0));
    }
  }

  m_num_3d_fields += num_dims;
}

template<typename MemoryManagement>
void BoundaryExchange::register_field (HostView<Real*[NP][NP],MemoryManagement> field)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (m_num_2d_fields+1<=m_2d_fields.extent_int(1));

  for (int ie=0; ie<m_num_elements; ++ie) {
    m_2d_fields(ie,m_num_2d_fields) = Pointer<Real*[NP][NP]>(&field(ie,0,0));
  }

  ++m_num_2d_fields;
}

template<typename MemoryManagement>
void BoundaryExchange::register_field (HostView<Scalar*[NP][NP][NUM_LEV],MemoryManagement> field)
{
  // Sanity checks
  assert (m_registration_started && !m_registration_completed);
  assert (m_num_3d_fields+1<=m_3d_fields.extent_int(1));

  for (int ie=0; ie<m_num_elements; ++ie) {
    m_3d_fields(ie,m_num_3d_fields) = Pointer<Scalar*[NP][NP][NUM_LEV]>(&field(ie,0,0,0));
  }

  ++m_num_3d_fields;
}

} // namespace Homme

#endif // HOMMEXX_BOUNDARY_EXCHANGE_HPP
