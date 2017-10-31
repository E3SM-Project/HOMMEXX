#include "BoundaryExchange.hpp"

#include "Control.hpp"

namespace Homme
{

BoundaryExchange& get_boundary_exchange(const std::string& name)
{
  std::map<std::string,BoundaryExchange>& be = get_all_boundary_exchange();

  Connectivity& connectivity = get_connectivity();
  if (be.find(name)==be.end())
  {
    be.emplace(name,connectivity);
  }

  return be[name];
}

std::map<std::string,BoundaryExchange>& get_all_boundary_exchange ()
{
  static std::map<std::string,BoundaryExchange> be;
  return be;
}

// ============================ IMPLEMENTATION ========================== //

BoundaryExchange::BoundaryExchange()
 : m_comm         (get_connectivity().get_comm())
 , m_connectivity (get_connectivity())
 , m_num_elements (m_connectivity.get_num_my_elems())
{
  m_send_buffer  = nullptr;
  m_recv_buffer  = nullptr;
  m_local_buffer = nullptr;

  m_num_2d_fields = 0;
  m_num_3d_fields = 0;

  // Prohibit registration until the number of fields has been set
  m_registration_started   = false;
  m_registration_completed = false;

  // Create requests and statuses
  m_send_requests.resize(m_connectivity.get_num_shared_connections(),MPI_REQUEST_NULL);
  m_recv_requests.resize(m_connectivity.get_num_shared_connections(),MPI_REQUEST_NULL);
  //m_statuses.resize(m_connectivity.get_num_shared_connections());

  // The zero array used for the dummy recv buffers at the 24 non-existent corner connections.
  std::fill_n(m_zero,NUM_LEV*VECTOR_SIZE,0.0);

  // We start with a clean class
  m_cleaned_up = true;
}

BoundaryExchange::BoundaryExchange(const Connectivity& connectivity)
 : m_comm         (connectivity.get_comm())
 , m_connectivity (connectivity)
 , m_num_elements (m_connectivity.get_num_my_elems())
{
  m_send_buffer  = nullptr;
  m_recv_buffer  = nullptr;
  m_local_buffer = nullptr;

  m_num_2d_fields = 0;
  m_num_3d_fields = 0;

  // Prohibit registration until the number of fields has been set
  m_registration_started   = false;
  m_registration_completed = false;

  // Create requests and statuses
  m_send_requests.resize(m_connectivity.get_num_shared_connections(),MPI_REQUEST_NULL);
  m_recv_requests.resize(m_connectivity.get_num_shared_connections(),MPI_REQUEST_NULL);
  //m_statuses.resize(m_connectivity.get_num_shared_connections());

  // The zero array used for the dummy recv buffers at the 24 non-existent corner connections.
  std::fill_n(m_zero,NUM_LEV*VECTOR_SIZE,0.0);

  // We start with a clean class
  m_cleaned_up = true;
}

BoundaryExchange::~BoundaryExchange()
{
  clean_up ();
}

void BoundaryExchange::set_num_fields(int num_2d_fields, int num_3d_fields)
{
  if (!m_cleaned_up) {
    // If for some reason we change the number of fields, we need to clean up everything.
    clean_up();
  }

  m_2d_fields = HostViewManaged<Pointer<Real[NP][NP]>**>("2d fields",m_num_elements,num_2d_fields);
  m_3d_fields = HostViewManaged<Pointer<Scalar[NP][NP][NUM_LEV]>**>("3d fields",m_num_elements,num_3d_fields);

  // Create buffers
  m_send_2d_corners_buffers = HostViewManaged<Pointer<Real[1]>**[NUM_CORNERS]>          ("2d send buffer",m_num_elements,num_2d_fields);
  m_send_2d_edges_buffers   = HostViewManaged<Pointer<Real[NP]>**[NUM_EDGES]>           ("2d send buffer",m_num_elements,num_2d_fields);
  m_send_3d_corners_buffers = HostViewManaged<Pointer<Scalar[NUM_LEV]>**[NUM_CORNERS]>  ("3d send buffer",m_num_elements,num_3d_fields);
  m_send_3d_edges_buffers   = HostViewManaged<Pointer<Scalar[NP][NUM_LEV]>**[NUM_EDGES]>("3d send buffer",m_num_elements,num_3d_fields);

  m_recv_2d_corners_buffers = HostViewManaged<Pointer<Real[1]>**[NUM_CORNERS]>          ("2d recv buffer",m_num_elements,num_2d_fields);
  m_recv_2d_edges_buffers   = HostViewManaged<Pointer<Real[NP]>**[NUM_EDGES]>           ("2d recv buffer",m_num_elements,num_2d_fields);
  m_recv_3d_corners_buffers = HostViewManaged<Pointer<Scalar[NUM_LEV]>**[NUM_CORNERS]>  ("3d recv buffer",m_num_elements,num_3d_fields);
  m_recv_3d_edges_buffers   = HostViewManaged<Pointer<Scalar[NP][NUM_LEV]>**[NUM_EDGES]>("3d recv buffer",m_num_elements,num_3d_fields);

  // Note: We need to initialize the recv corner buffers with null views, since we later check them
  //       to establish which buffers refer to one of the 24 non-existent corner connections
  Kokkos::deep_copy(m_recv_2d_corners_buffers,Pointer<Real[1]>(nullptr));
  Kokkos::deep_copy(m_recv_3d_corners_buffers,Pointer<Scalar[NUM_LEV]>(nullptr));

  // Now we can start register fields
  m_registration_started   = true;
  m_registration_completed = false;

  // We're not all clean
  m_cleaned_up = false;
}

void BoundaryExchange::clean_up()
{
  if (m_cleaned_up == true) {
    // Perhaps not possible, but just in case
    return;
  }

  // Make sure the data has been sent before we cleanup this class
  //MPI_Waitall(m_send_requests.size(),m_send_requests.data(),m_statuses.data());
  MPI_Waitall(m_send_requests.size(),m_send_requests.data(),MPI_STATUSES_IGNORE);
#ifdef HOMMEXX_DEBUG
  // TODO: check statuses
#endif

  // Free buffers
  MPI_Free_mem (m_send_buffer);
  MPI_Free_mem (m_recv_buffer);
  delete[] m_local_buffer;

  // Free MPI data types
  MPI_Type_free(&m_mpi_corner_data_type);
  MPI_Type_free(&m_mpi_edge_data_type);

  // Clear stored fields
  m_2d_fields = HostViewManaged<Pointer<Real[NP][NP]>**>(0,0);
  m_3d_fields = HostViewManaged<Pointer<Scalar[NP][NP][NUM_LEV]>**>(0,0);

  m_num_2d_fields = 0;
  m_num_3d_fields = 0;

  // Clear buffer views
  m_send_2d_corners_buffers = HostViewManaged<Pointer<Real[1]>**[NUM_CORNERS]>          (0,0);
  m_send_2d_edges_buffers   = HostViewManaged<Pointer<Real[NP]>**[NUM_EDGES]>           (0,0);
  m_send_3d_corners_buffers = HostViewManaged<Pointer<Scalar[NUM_LEV]>**[NUM_CORNERS]>  (0,0);
  m_send_3d_edges_buffers   = HostViewManaged<Pointer<Scalar[NP][NUM_LEV]>**[NUM_EDGES]>(0,0);

  m_recv_2d_corners_buffers = HostViewManaged<Pointer<Real[1]>**[NUM_CORNERS]>          (0,0);
  m_recv_2d_edges_buffers   = HostViewManaged<Pointer<Real[NP]>**[NUM_EDGES]>           (0,0);
  m_recv_3d_corners_buffers = HostViewManaged<Pointer<Scalar[NUM_LEV]>**[NUM_CORNERS]>  (0,0);
  m_recv_3d_edges_buffers   = HostViewManaged<Pointer<Scalar[NP][NUM_LEV]>**[NUM_EDGES]>(0,0);

  // If we clean up, we need to reset the number of fields
  m_registration_started   = false;
  m_registration_completed = false;

  // Clean requests and statuses
  m_send_requests.clear();
  m_recv_requests.clear();
  //m_statuses.clear();

  // Now we're all cleaned
  m_cleaned_up = true;
}

void BoundaryExchange::registration_completed()
{
  // There may be issues if we setup MPI twice, so let's not do it
  if (m_registration_completed) {
    return;
  }

  // Create the MPI data types, for corners and edges
  // Note: this is the size per element, per connection. It is the number of Real's to send/receive to/from the neighbor
  m_corner_size = (m_num_2d_fields + m_num_3d_fields*NUM_LEV*VECTOR_SIZE) * 1;
  m_edge_size   = (m_num_2d_fields + m_num_3d_fields*NUM_LEV*VECTOR_SIZE) * NP;
  MPI_Type_contiguous(m_corner_size, MPI_DOUBLE, &m_mpi_corner_data_type);
  MPI_Type_contiguous(m_edge_size, MPI_DOUBLE, &m_mpi_edge_data_type);
  MPI_Type_commit(&m_mpi_corner_data_type);
  MPI_Type_commit(&m_mpi_edge_data_type);

  // Compute the buffers sizes and allocating
  size_t mpi_buffers_size = 0;
  size_t local_buffers_size = 0;

  mpi_buffers_size += m_corner_size * m_connectivity.get_num_shared_corner_connections();
  mpi_buffers_size += m_edge_size   * m_connectivity.get_num_shared_edge_connections();

  local_buffers_size += m_corner_size * m_connectivity.get_num_local_corner_connections();
  local_buffers_size += m_edge_size   * m_connectivity.get_num_local_edge_connections();

  MPI_Alloc_mem (mpi_buffers_size*sizeof(Real),MPI_INFO_NULL,&m_send_buffer);
  MPI_Alloc_mem (mpi_buffers_size*sizeof(Real),MPI_INFO_NULL,&m_recv_buffer);
  m_local_buffer = new Real[local_buffers_size]; // This one can be allocated 'normally'

  // Setting the individual field/elements buffers to point to the right piece of the buffers
  int num_local_corner_connections  = m_connectivity.get_num_local_corner_connections();
  int num_local_edge_connections    = m_connectivity.get_num_local_edge_connections();
  int num_shared_corner_connections = m_connectivity.get_num_shared_corner_connections();
  int num_shared_edge_connections   = m_connectivity.get_num_shared_edge_connections();

  size_t offset = 0;
  // Set up the shared connections first (corners, then edges)
  for (int iconn=0; iconn<num_shared_corner_connections; ++iconn) {
    const ConnectionInfo& info = m_connectivity.get_shared_corner_connections()[iconn];
    const LidPosType& local = info.local;
    for (int ifield=0; ifield<m_num_2d_fields; ++ifield) {
      m_send_2d_corners_buffers(local.lid,ifield,local.pos) = Pointer<Real[1]>(m_send_buffer+offset);
      m_recv_2d_corners_buffers(local.lid,ifield,local.pos) = Pointer<Real[1]>(m_recv_buffer+offset);
      offset += 1;
    }
    for (int ifield=0; ifield<m_num_3d_fields; ++ifield) {
      m_send_3d_corners_buffers(local.lid,ifield,local.pos) = Pointer<Scalar[NUM_LEV]>(reinterpret_cast<Scalar*>(m_send_buffer+offset));
      m_recv_3d_corners_buffers(local.lid,ifield,local.pos) = Pointer<Scalar[NUM_LEV]>(reinterpret_cast<Scalar*>(m_recv_buffer+offset));
      offset += NUM_LEV*VECTOR_SIZE;
    }
  }
  for (int iconn=0; iconn<num_shared_edge_connections; ++iconn) {
    const ConnectionInfo& info = m_connectivity.get_shared_edge_connections()[iconn];
    const LidPosType& local = info.local;
    for (int ifield=0; ifield<m_num_2d_fields; ++ifield) {
      m_send_2d_edges_buffers(local.lid,ifield,local.pos) = Pointer<Real[NP]>(m_send_buffer+offset);
      m_recv_2d_edges_buffers(local.lid,ifield,local.pos) = Pointer<Real[NP]>(m_recv_buffer+offset);
      offset += NP;
    }
    for (int ifield=0; ifield<m_num_3d_fields; ++ifield) {
      m_send_3d_edges_buffers(local.lid,ifield,local.pos) = Pointer<Scalar[NP][NUM_LEV]>(reinterpret_cast<Scalar*>(m_send_buffer+offset));
      m_recv_3d_edges_buffers(local.lid,ifield,local.pos) = Pointer<Scalar[NP][NUM_LEV]>(reinterpret_cast<Scalar*>(m_recv_buffer+offset));
      offset += NP*NUM_LEV*VECTOR_SIZE;
    }
  }
  // Sanity check
  assert (offset==mpi_buffers_size);

  // Then set up the local connections (corners, then edges)
  offset = 0;
  for (int iconn=0; iconn<num_local_corner_connections; ++iconn) {
    const ConnectionInfo& info = m_connectivity.get_local_corner_connections()[iconn];
    const LidPosType& local = info.local;
    for (int ifield=0; ifield<m_num_2d_fields; ++ifield) {
      m_send_2d_corners_buffers(local.lid,ifield,local.pos) = Pointer<Real[1]>(m_local_buffer+offset);
      m_recv_2d_corners_buffers(local.lid,ifield,local.pos) = Pointer<Real[1]>(m_local_buffer+offset);
      offset += 1;
    }
    for (int ifield=0; ifield<m_num_3d_fields; ++ifield) {
      m_send_3d_corners_buffers(local.lid,ifield,local.pos) = Pointer<Scalar[NUM_LEV]>(reinterpret_cast<Scalar*>(m_local_buffer+offset));
      m_recv_3d_corners_buffers(local.lid,ifield,local.pos) = Pointer<Scalar[NUM_LEV]>(reinterpret_cast<Scalar*>(m_local_buffer+offset));
      offset += NUM_LEV*VECTOR_SIZE;
    }
  }
  for (int iconn=0; iconn<num_local_edge_connections; ++iconn) {
    const ConnectionInfo& info = m_connectivity.get_local_edge_connections()[iconn];
    const LidPosType& local = info.local;
    for (int ifield=0; ifield<m_num_2d_fields; ++ifield) {
      m_send_2d_edges_buffers(local.lid,ifield,local.pos) = Pointer<Real[NP]>(m_local_buffer+offset);
      m_recv_2d_edges_buffers(local.lid,ifield,local.pos) = Pointer<Real[NP]>(m_local_buffer+offset);
      offset += NP;
    }
    for (int ifield=0; ifield<m_num_3d_fields; ++ifield) {
      m_send_3d_edges_buffers(local.lid,ifield,local.pos) = Pointer<Scalar[NP][NUM_LEV]>(reinterpret_cast<Scalar*>(m_local_buffer+offset));
      m_recv_3d_edges_buffers(local.lid,ifield,local.pos) = Pointer<Scalar[NP][NUM_LEV]>(reinterpret_cast<Scalar*>(m_local_buffer+offset));
      offset += NP*NUM_LEV*VECTOR_SIZE;
    }
  }
  // Sanity check
  assert (offset==local_buffers_size);

  // Note: there are 24 squares on the cubed sphere that miss one corner connection.
  //       The unpack method loops over elements and ALL their 8 connections, which would
  //       cause a crash, when we hit the first of those elements' missing connections.
  //       To avoid this, for those 24 connections, we create an unmanaged view that
  //       points to the 0-filled array m_zero. Notice that we only have to create a
  //       view for the recv buffers, since they are the ones used in the unpack method.
  for (int ie=0; ie<m_num_elements; ++ie) {
    for (int icorner=0; icorner<NUM_CORNERS; ++icorner) {
      if (m_num_2d_fields>0 && m_recv_2d_corners_buffers(ie,0,icorner).data()!=nullptr) {
        continue;
      } else if (m_num_3d_fields>0 && m_recv_3d_corners_buffers(ie,0,icorner).data()!=nullptr) {
        continue;
      }

      // If we get here, then this is a non-existent corner connection. Let's set up the dummy views
      for (int ifield=0; ifield<m_num_2d_fields; ++ifield) {
        m_recv_2d_corners_buffers(ie,ifield,icorner) = Pointer<Real[1]>(m_zero);
      }
      for (int ifield=0; ifield<m_num_3d_fields; ++ifield) {
        m_recv_3d_corners_buffers(ie,ifield,icorner) = Pointer<Scalar[NUM_LEV]>(reinterpret_cast<Scalar*>(m_zero));
      }

      // Note: Homme does not support the case ne=1, so each element can have AT MOST
      //       one non-existent connection. Therefore, we can break the loop on corners.
      break;
    }
  }

  // Prohibit further registration of fields
  m_registration_started   = false;
  m_registration_completed = true;
}

void BoundaryExchange::exchange ()
{
  // Check that the registration has completed first
  assert (m_registration_completed);

  // Pack into send buffer
  pack();

  // Send and receive
  send_and_receive();

  // Unpack
  unpack();
}

void BoundaryExchange::pack()
{
  // Make sure the data has been sent (from previous calls) before reusing buffers
  //MPI_Waitall(m_send_requests.size(),m_send_requests.data(),m_statuses.data());
  MPI_Waitall(m_send_requests.size(),m_send_requests.data(),MPI_STATUSES_IGNORE);
#ifdef HOMMEXX_DEBUG
  // TODO: check statuses
#endif

  // When we pack, we copy data into the buffer views. We do not care whether the connection
  // is local or shared, since the process is the same. The difference is just in whether the
  // view's underlying pointer is pointing to an area of m_send_buffer or m_local_buffer.

  int num_connections = m_connectivity.get_num_connections();
  HostViewUnmanaged<const ConnectionInfo*> connections(m_connectivity.get_connections().data(),num_connections);

  // TODO: parallel for's
  for (int iconn=0; iconn<num_connections; ++iconn) {
    const ConnectionInfo& info = connections[iconn];
    const LidPosType& field_lpt  = info.local;
    // For the buffer, in case of local connection, use remote info. In fact, while with shared connections the
    // mpi call will take care of "copying" data to the remote recv buffer in the correct remote element lid,
    // for local connections we need to manually copy on the remote element lid. We can do it here
    const LidPosType& buffer_lpt = info.remote_pid==-1 ? info.remote : info.local;

    if (info.kind==CORNER_KIND) {
      const GaussPoint& pt = CORNER_PTS[field_lpt.pos];
      // First, pack 2d fields
      for (int ifield=0; ifield<m_num_2d_fields; ++ifield) {
        m_send_2d_corners_buffers(buffer_lpt.lid,ifield,buffer_lpt.pos)[0] = m_2d_fields(field_lpt.lid,ifield)(pt.ip,pt.jp);
      }
      // Then, pack 3d fields
      for (int ifield=0; ifield<m_num_3d_fields; ++ifield) {
        for (int ilev=0; ilev<NUM_LEV; ++ilev) {
          m_send_3d_corners_buffers(buffer_lpt.lid,ifield,buffer_lpt.pos)(ilev) = m_3d_fields(field_lpt.lid,ifield)(pt.ip,pt.jp,ilev);
        }
      }
    } else {
      // Note: if the remote edge is in the reverse order, we read the field_lpt points backwards
      const auto& pts  = EDGE_PTS[info.direction][field_lpt.pos];
      // First, pack 2d fields
      for (int ifield=0; ifield<m_num_2d_fields; ++ifield) {
        for (int k=0; k<NP; ++k) {
          m_send_2d_edges_buffers(buffer_lpt.lid,ifield,buffer_lpt.pos)(k) = m_2d_fields(field_lpt.lid,ifield)(pts[k].ip,pts[k].jp);
        }
      }
      // Then, pack 3d fields
      for (int ifield=0; ifield<m_num_3d_fields; ++ifield) {
        for (int k=0; k<NP; ++k) {
          for (int ilev=0; ilev<NUM_LEV; ++ilev) {
            m_send_3d_edges_buffers(buffer_lpt.lid,ifield,buffer_lpt.pos)(k,ilev) = m_3d_fields(field_lpt.lid,ifield)(pts[k].ip,pts[k].jp,ilev);
          }
        }
      }
    }
  }
}

void BoundaryExchange::send_and_receive()
{
  auto shared_corner_connections = m_connectivity.get_shared_corner_connections();
  auto shared_edge_connections   = m_connectivity.get_shared_edge_connections();

  const int num_shared_corner_connections = m_connectivity.get_num_shared_corner_connections();
  const int num_shared_edge_connections   = m_connectivity.get_num_shared_edge_connections();

  // Process all sends first (corners, then edges)
  for (int iconn=0; iconn<num_shared_corner_connections; ++iconn)
  {
    const ConnectionInfo& info = shared_corner_connections[iconn];

    // We build a tag that has info about the sender's element lid and connection position.
    // Since there are 8 neighbors, the easy way is to set tag=lid*8+pos
    int tag = info.local.lid*NUM_NEIGHBORS + info.local.pos;

    Real* ptr = m_send_buffer + iconn*m_corner_size;
    MPI_Isend(ptr,1,m_mpi_corner_data_type,info.remote_pid,tag,m_comm.m_mpi_comm,&m_send_requests[iconn]);
  }
  for (int iconn=0; iconn<num_shared_edge_connections; ++iconn)
  {
    const ConnectionInfo& info = shared_edge_connections[iconn];

    // We build a tag that has info about the sender's element lid and connection position.
    // Since there are 8 neighbors, the easy way is to set tag=lid*8+pos
    int tag = info.local.lid*NUM_NEIGHBORS + info.local.pos;

    Real* ptr = m_send_buffer + iconn*m_edge_size + num_shared_corner_connections*m_corner_size;
    MPI_Isend(ptr,1,m_mpi_edge_data_type,info.remote_pid,tag,m_comm.m_mpi_comm,&m_send_requests[iconn+num_shared_corner_connections]);
  }

  // Process all receives (corners, then edges)
  for (int iconn=0; iconn<num_shared_corner_connections; ++iconn)
  {
    const ConnectionInfo& info = shared_corner_connections[iconn];
    // We build a tag that has info about the sender's element lid and connection position.
    // Since there are 8 neighbors, the easy way is to set tag=lid*8+pos
    int tag = info.remote.lid*NUM_NEIGHBORS + info.remote.pos;

    Real* ptr = m_recv_buffer + iconn*m_corner_size;
    MPI_Irecv(ptr,1,m_mpi_corner_data_type,info.remote_pid,tag,m_comm.m_mpi_comm,&m_recv_requests[iconn]);
  }
  for (int iconn=0; iconn<num_shared_edge_connections; ++iconn)
  {
    const ConnectionInfo& info = shared_edge_connections[iconn];

    // We build a tag that has info about the sender's element lid and connection position.
    // Since there are 8 neighbors, the easy way is to set tag=lid*8+pos
    int tag = info.remote.lid*NUM_NEIGHBORS + info.remote.pos;

    Real* ptr = m_recv_buffer + iconn*m_edge_size + num_shared_corner_connections*m_corner_size;
    MPI_Irecv(ptr,1,m_mpi_edge_data_type,info.remote_pid,tag,m_comm.m_mpi_comm,&m_recv_requests[iconn+num_shared_corner_connections]);
  }
}

void BoundaryExchange::unpack()
{
  // Make sure all the data has been received before unpacking
  //MPI_Waitall(m_recv_requests.size(),m_recv_requests.data(),m_statuses.data());
  MPI_Waitall(m_recv_requests.size(),m_recv_requests.data(),MPI_STATUSES_IGNORE);
#ifdef HOMMEXX_DEBUG
  // TODO: check statuses
#endif
  for (int ie=0; ie<m_num_elements; ++ie)
  {
    for (int ifield=0; ifield<m_num_2d_fields; ++ifield) {
      for (int k=0; k<NP; ++k) {
        m_2d_fields(ie,ifield)(EDGE_PTS_FWD[SOUTH][k].ip,EDGE_PTS_FWD[SOUTH][k].jp) += m_recv_2d_edges_buffers(ie,ifield,SOUTH)[k];
        m_2d_fields(ie,ifield)(EDGE_PTS_FWD[NORTH][k].ip,EDGE_PTS_FWD[NORTH][k].jp) += m_recv_2d_edges_buffers(ie,ifield,NORTH)[k];
        m_2d_fields(ie,ifield)(EDGE_PTS_FWD[WEST][k].ip, EDGE_PTS_FWD[WEST][k].jp)  += m_recv_2d_edges_buffers(ie,ifield,WEST )[k];
        m_2d_fields(ie,ifield)(EDGE_PTS_FWD[EAST][k].ip, EDGE_PTS_FWD[EAST][k].jp)  += m_recv_2d_edges_buffers(ie,ifield,EAST )[k];
      }
      for (int icorner=0; icorner<NUM_CORNERS; ++icorner) {
        m_2d_fields(ie,ifield)(CORNER_PTS[icorner].ip,CORNER_PTS[icorner].jp) += m_recv_2d_corners_buffers(ie,ifield,icorner)[0];
      }
    }
    for (int ifield=0; ifield<m_num_3d_fields; ++ifield) {
      for (int k=0; k<NP; ++k) {
        for (int ilev=0; ilev<NUM_LEV; ++ilev) {
          m_3d_fields(ie,ifield)(EDGE_PTS_FWD[SOUTH][k].ip,EDGE_PTS_FWD[SOUTH][k].jp,ilev) += m_recv_3d_edges_buffers(ie,ifield,SOUTH)(k,ilev);
          m_3d_fields(ie,ifield)(EDGE_PTS_FWD[NORTH][k].ip,EDGE_PTS_FWD[NORTH][k].jp,ilev) += m_recv_3d_edges_buffers(ie,ifield,NORTH)(k,ilev);
          m_3d_fields(ie,ifield)(EDGE_PTS_FWD[WEST][k].ip, EDGE_PTS_FWD[WEST][k].jp ,ilev) += m_recv_3d_edges_buffers(ie,ifield,WEST )(k,ilev);
          m_3d_fields(ie,ifield)(EDGE_PTS_FWD[EAST][k].ip, EDGE_PTS_FWD[EAST][k].jp ,ilev) += m_recv_3d_edges_buffers(ie,ifield,EAST )(k,ilev);
        }
      }
      for (int icorner=0; icorner<NUM_CORNERS; ++icorner) {
        for (int ilev=0; ilev<NUM_LEV; ++ilev) {
          m_3d_fields(ie,ifield)(CORNER_PTS[icorner].ip,CORNER_PTS[icorner].jp,ilev) += m_recv_3d_corners_buffers(ie,ifield,icorner)(ilev);
        }
      }
    }
  }
/*
  // Get control structure
  Control& data  = get_control();

  // Retrieve the team size
  const int vectors_per_thread = DefaultThreadsDistribution<ExecSpace>::vectors_per_thread();
  const int threads_per_team   = data.team_size;

  // Setup the policy
  Kokkos::TeamPolicy<ExecSpace> policy(data.num_elems, threads_per_team, vectors_per_thread);
  policy.set_chunk_size(1);

  // Finally, sum local buffers into fields' views
  Kokkos::parallel_for(policy,[&](TeamMember team) {
    const int ie = team.league_rank();

    // First copy the edges (NOTE: we do not thread over edges nor edges points, in order to make the sum reproducible, and in the same order as it is done in Fortran)
    // Then copy corners (NOTE: this part *could* be threaded (provided edges have ALL been updated already), since corners do not overlap)
    // TODO: if we agree on a *different* reproducible sum order, we can thread also on edges' points, to get NPx more parallel work.
    //       in particular, if we decide that we sum a whole edge before moving to the next one, we can process the edges' points in parallel
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,m_num_2d_fields),
                         [&](const int ifield) {
      Kokkos::single(Kokkos::PerThread(team),
                     [&](){
        for (int k=0; k<NP; ++k) {
          m_2d_fields(ie,ifield)(EDGE_PTS_FWD[SOUTH][k].ip,EDGE_PTS_FWD[SOUTH][k].jp) += m_recv_2d_edges_buffers(ie,ifield,SOUTH)(k);
          m_2d_fields(ie,ifield)(EDGE_PTS_FWD[NORTH][k].ip,EDGE_PTS_FWD[NORTH][k].jp) += m_recv_2d_edges_buffers(ie,ifield,NORTH)(k);
          m_2d_fields(ie,ifield)(EDGE_PTS_FWD[WEST][k].ip, EDGE_PTS_FWD[WEST][k].jp)  += m_recv_2d_edges_buffers(ie,ifield,WEST )(k);
          m_2d_fields(ie,ifield)(EDGE_PTS_FWD[EAST][k].ip, EDGE_PTS_FWD[EAST][k].jp)  += m_recv_2d_edges_buffers(ie,ifield,EAST )(k);
        }
      });
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,NUM_CORNERS),
                           [&](const int icorner){
        m_2d_fields(ie,ifield)(CORNER_PTS[icorner].ip,CORNER_PTS[icorner].jp) += m_recv_2d_corners_buffers(ie,ifield,icorner)[0];
      });
    });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,m_num_3d_fields),
                         [&](const int ifield) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,NUM_LEV),
                         [&](const int ilev) {
        for (int k=0; k<NP; ++k) {
          m_3d_fields(ie,ifield)(EDGE_PTS_FWD[SOUTH][k].ip,EDGE_PTS_FWD[SOUTH][k].jp,ilev) += m_recv_3d_edges_buffers(ie,ifield,SOUTH)(k,ilev);
          m_3d_fields(ie,ifield)(EDGE_PTS_FWD[NORTH][k].ip,EDGE_PTS_FWD[NORTH][k].jp,ilev) += m_recv_3d_edges_buffers(ie,ifield,NORTH)(k,ilev);
          m_3d_fields(ie,ifield)(EDGE_PTS_FWD[WEST][k].ip, EDGE_PTS_FWD[WEST][k].jp, ilev) += m_recv_3d_edges_buffers(ie,ifield,WEST )(k,ilev);
          m_3d_fields(ie,ifield)(EDGE_PTS_FWD[EAST][k].ip, EDGE_PTS_FWD[EAST][k].jp, ilev) += m_recv_3d_edges_buffers(ie,ifield,EAST )(k,ilev);
        }
        for (int icorner=0; icorner<NUM_CORNERS; ++icorner) {
          m_3d_fields(ie,ifield)(CORNER_PTS[icorner].ip,CORNER_PTS[icorner].jp,ilev) += m_recv_3d_corners_buffers(ie,ifield,icorner)(ilev);
        }
      });
    });
  });
  ExecSpace::fence();
*/
}

} // namespace Homme
