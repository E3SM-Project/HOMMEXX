#include "BoundaryExchange.hpp"

#include "Context.hpp"
#include "Control.hpp"

namespace Homme
{

// ======================== IMPLEMENTATION ======================== //

BoundaryExchange::BoundaryExchange()
 : m_comm          (Context::singleton().get_connectivity().get_comm())
 , m_connectivity  (Context::singleton().get_connectivity())
 , m_num_elements  (Context::singleton().get_connectivity().get_num_elements())
{
  m_num_2d_fields = 0;
  m_num_3d_fields = 0;

  // Prohibit registration until the number of fields has been set
  m_registration_started   = false;
  m_registration_completed = false;

  // Create requests
  // Note: we put an extra null request at the end, so that if we have no share connections
  //       (1 rank), m_*_requests.data() is not NULL. NULL would cause MPI_Startall to abort
  m_send_requests.resize(m_connectivity.get_num_shared_connections<HostMemSpace>()+1,MPI_REQUEST_NULL);
  m_recv_requests.resize(m_connectivity.get_num_shared_connections<HostMemSpace>()+1,MPI_REQUEST_NULL);

  // These zero arrays are used for the dummy send/recv buffers at non-existent corner connections.
  m_blackhole_send = ExecViewManaged<Real[NUM_LEV*VECTOR_SIZE]>("blackhole array");
  m_blackhole_recv = ExecViewManaged<Real[NUM_LEV*VECTOR_SIZE]>("blackhole array");
  Kokkos::deep_copy(m_blackhole_send,0.0);
  Kokkos::deep_copy(m_blackhole_recv,0.0);

  // We start with a clean class
  m_cleaned_up = true;
}

BoundaryExchange::BoundaryExchange(const Connectivity& connectivity)
 : m_comm          (connectivity.get_comm())
 , m_connectivity  (connectivity)
 , m_num_elements  (connectivity.get_num_elements())
{
  m_num_2d_fields = 0;
  m_num_3d_fields = 0;

  // Prohibit registration until the number of fields has been set
  m_registration_started   = false;
  m_registration_completed = false;

  // Create requests
  // Note: we put an extra null request at the end, so that if we have no share connections
  //       (1 rank), m_*_requests.data() is not NULL. NULL would cause MPI_Startall to abort
  m_send_requests.resize(m_connectivity.get_num_shared_connections<HostMemSpace>()+1,MPI_REQUEST_NULL);
  m_recv_requests.resize(m_connectivity.get_num_shared_connections<HostMemSpace>()+1,MPI_REQUEST_NULL);

  // These zero arrays are used for the dummy send/recv buffers at non-existent corner connections.
  m_blackhole_send = ExecViewManaged<Real[NUM_LEV*VECTOR_SIZE]>("blackhole array");
  m_blackhole_recv = ExecViewManaged<Real[NUM_LEV*VECTOR_SIZE]>("blackhole array");
  Kokkos::deep_copy(m_blackhole_send,0.0);
  Kokkos::deep_copy(m_blackhole_recv,0.0);

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

  // Note: we do not set m_num_2d_fields and m_num_3d_fields, since we will use them as
  //       progressive indices while adding fields. Then, during registration_completed,
  //       we will check that they match the 2nd dimension of the m_2d_fields and m_3d_fields.

  // Create the fields views
  m_2d_fields = ExecViewManaged<ExecViewManaged<Real[NP][NP]>**>("2d fields",m_num_elements,num_2d_fields);
  m_3d_fields = ExecViewManaged<ExecViewManaged<Scalar[NP][NP][NUM_LEV]>**>("3d fields",m_num_elements,num_3d_fields);

  // Create buffers
  m_send_2d_buffers = ExecViewManaged<ExecViewUnmanaged<Real*>**[NUM_CONNECTIONS]>("2d send buffer",m_num_elements,num_2d_fields);
  m_recv_2d_buffers = ExecViewManaged<ExecViewUnmanaged<Real*>**[NUM_CONNECTIONS]>("2d send buffer",m_num_elements,num_2d_fields);
  m_send_3d_buffers = ExecViewManaged<ExecViewUnmanaged<Scalar*[NUM_LEV]>**[NUM_CONNECTIONS]>("3d send buffer",m_num_elements,num_3d_fields);
  m_recv_3d_buffers = ExecViewManaged<ExecViewUnmanaged<Scalar*[NUM_LEV]>**[NUM_CONNECTIONS]>("3d send buffer",m_num_elements,num_3d_fields);

  // Now we can start register fields
  m_registration_started   = true;
  m_registration_completed = false;

  // We're not all clean
  m_cleaned_up = false;
}

void BoundaryExchange::clean_up()
{
  if (m_cleaned_up) {
    // Perhaps not possible, but just in case
    return;
  }

  // Make sure the data has been sent before we cleanup this class
  HOMMEXX_MPI_CHECK_ERROR(MPI_Waitall(m_connectivity.get_num_shared_connections<HostMemSpace>(),m_send_requests.data(),MPI_STATUSES_IGNORE));

  // Free buffers
  m_send_buffer  = ExecViewManaged<Real*>("send buffer", 0);
  m_recv_buffer  = ExecViewManaged<Real*>("recv buffer", 0);
  m_local_buffer = ExecViewManaged<Real*>("local buffer",0);

  // Free MPI data types
  HOMMEXX_MPI_CHECK_ERROR(MPI_Type_free(&m_mpi_data_type[etoi(ConnectionKind::CORNER)]));
  HOMMEXX_MPI_CHECK_ERROR(MPI_Type_free(&m_mpi_data_type[etoi(ConnectionKind::EDGE)]));

  // Clear stored fields
  m_2d_fields = ExecViewManaged<ExecViewManaged<Real[NP][NP]>**>(0,0);
  m_3d_fields = ExecViewManaged<ExecViewManaged<Scalar[NP][NP][NUM_LEV]>**>(0,0);

  m_num_2d_fields = 0;
  m_num_3d_fields = 0;

  // Clear buffer views
  m_send_2d_buffers = ExecViewManaged<ExecViewUnmanaged<Real*>**[NUM_CONNECTIONS]>(0,0);
  m_recv_2d_buffers = ExecViewManaged<ExecViewUnmanaged<Real*>**[NUM_CONNECTIONS]>(0,0);
  m_send_3d_buffers = ExecViewManaged<ExecViewUnmanaged<Scalar*[NUM_LEV]>**[NUM_CONNECTIONS]>(0,0);
  m_recv_3d_buffers = ExecViewManaged<ExecViewUnmanaged<Scalar*[NUM_LEV]>**[NUM_CONNECTIONS]>(0,0);

  // If we clean up, we need to reset the number of fields
  m_registration_started   = false;
  m_registration_completed = false;

  // Clean requests
  for (int i=0; i<m_connectivity.get_num_shared_connections<HostMemSpace>(); ++i) {
    HOMMEXX_MPI_CHECK_ERROR(MPI_Request_free(&m_send_requests[i]));
    HOMMEXX_MPI_CHECK_ERROR(MPI_Request_free(&m_recv_requests[i]));
  }
  m_send_requests.clear();
  m_recv_requests.clear();

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
  m_elem_buf_size[etoi(ConnectionKind::CORNER)] = (m_num_2d_fields + m_num_3d_fields*NUM_LEV*VECTOR_SIZE) * 1;
  m_elem_buf_size[etoi(ConnectionKind::EDGE)]   = (m_num_2d_fields + m_num_3d_fields*NUM_LEV*VECTOR_SIZE) * NP;
  HOMMEXX_MPI_CHECK_ERROR(MPI_Type_contiguous(m_elem_buf_size[etoi(ConnectionKind::CORNER)], MPI_DOUBLE, &m_mpi_data_type[etoi(ConnectionKind::CORNER)]));
  HOMMEXX_MPI_CHECK_ERROR(MPI_Type_contiguous(m_elem_buf_size[etoi(ConnectionKind::EDGE)],   MPI_DOUBLE, &m_mpi_data_type[etoi(ConnectionKind::EDGE)]));
  HOMMEXX_MPI_CHECK_ERROR(MPI_Type_commit(&m_mpi_data_type[etoi(ConnectionKind::CORNER)]));
  HOMMEXX_MPI_CHECK_ERROR(MPI_Type_commit(&m_mpi_data_type[etoi(ConnectionKind::EDGE)]));

  // Compute the buffers sizes and allocating
  size_t mpi_buffer_size = 0;
  size_t local_buffer_size = 0;

  mpi_buffer_size   += m_elem_buf_size[etoi(ConnectionKind::CORNER)] * m_connectivity.get_num_connections<HostMemSpace>(ConnectionSharing::SHARED,ConnectionKind::CORNER);
  mpi_buffer_size   += m_elem_buf_size[etoi(ConnectionKind::EDGE)]   * m_connectivity.get_num_connections<HostMemSpace>(ConnectionSharing::SHARED,ConnectionKind::EDGE);

  local_buffer_size += m_elem_buf_size[etoi(ConnectionKind::CORNER)] * m_connectivity.get_num_connections<HostMemSpace>(ConnectionSharing::LOCAL,ConnectionKind::CORNER);
  local_buffer_size += m_elem_buf_size[etoi(ConnectionKind::EDGE)]   * m_connectivity.get_num_connections<HostMemSpace>(ConnectionSharing::LOCAL,ConnectionKind::EDGE);

  // Create the buffers
  m_send_buffer  = ExecViewManaged<Real*>("send buffer",  mpi_buffer_size);
  m_recv_buffer  = ExecViewManaged<Real*>("recv buffer",  mpi_buffer_size);
  m_local_buffer = ExecViewManaged<Real*>("local buffer", local_buffer_size);

  // Create the mpi buffers (same as send/recv buffers if ExecMemSpace=MPIMemSpace)
  m_mpi_send_buffer = Kokkos::create_mirror_view(decltype(m_mpi_send_buffer)::execution_space(),m_send_buffer);
  m_mpi_recv_buffer = Kokkos::create_mirror_view(decltype(m_mpi_recv_buffer)::execution_space(),m_recv_buffer);

  // Note: this may look cryptic, so I'll try to explain what's about to happen.
  //       We want to set the send/recv buffers to point to:
  //         - a portion of m_send/recv_buffer if info.sharing=SHARED
  //         - a portion of m_local_buffer if info.sharing=LOCAL
  //         - the blackhole_send/recv if info.sharing=MISSING
  //       After reserving the buffer portion, update the offset by a given increment, depending on info.kind:
  //         - increment[CORNER]  = m_elem_buf_size[CORNER)] = 1  * (m_num_2d_fields + NUM_LEV*VECTOR_SIZE m_num_3d_fields)
  //         - increment[EDGE]    = m_elem_buf_size[EDGE)]   = NP * (m_num_2d_fields + NUM_LEV*VECTOR_SIZE m_num_3d_fields)
  //         - increment[MISSING] = 0 (point to the same blackhole)
  // Note: m_blackhole_send will be written many times, but will never be read from.
  //       Kind of like streaming to /dev/null. m_blackhole_recv will be read from sometimes
  //       (24 times, to be precise, one for each of the 3 corner connections on each of the
  //       cube's vertices), but it's never written into, so will always contain zeros (set by the constructor).

  HostViewManaged<size_t[3]> h_buf_offset("");
  Kokkos::deep_copy(h_buf_offset,0);

  HostViewManaged<int[3]> h_increment("increment");
  h_increment[etoi(ConnectionKind::EDGE)]    = NP;
  h_increment[etoi(ConnectionKind::CORNER)]  =  1;
  h_increment[etoi(ConnectionKind::MISSING)] =  0;

  using local_buf_ptr_type = decltype(m_local_buffer.data());
  using local_buf_val_type = decltype(*m_local_buffer.data());
  HostViewManaged<Pointer<local_buf_ptr_type,local_buf_val_type>[3]> h_all_send_buffers("");
  HostViewManaged<Pointer<local_buf_ptr_type,local_buf_val_type>[3]> h_all_recv_buffers("");

  h_all_send_buffers[etoi(ConnectionSharing::LOCAL)]   = m_local_buffer.data();
  h_all_send_buffers[etoi(ConnectionSharing::SHARED)]  = m_send_buffer.data();
  h_all_send_buffers[etoi(ConnectionSharing::MISSING)] = m_blackhole_send.data();
  h_all_recv_buffers[etoi(ConnectionSharing::LOCAL)]   = m_local_buffer.data();
  h_all_recv_buffers[etoi(ConnectionSharing::SHARED)]  = m_recv_buffer.data();
  h_all_recv_buffers[etoi(ConnectionSharing::MISSING)] = m_blackhole_recv.data();

  // NOTE: I wanted to do this setup in parallel, on the execution space, but there
  //       is a reduction hidden. In particular, we need to access buf_offset atomically,
  //       so that it is not update while we are still using it. One solution would be to
  //       use Kokkos::atomic_fetch_add, but it may kill the concurrency. And given that
  //       we do not have a ton of concurrency in this setup phase, and given that it is
  //       precisely only a setup phase, we may as well do things serially on the host,
  //       then deep_copy back to device
  ConnectionHelpers helpers;
  auto h_send_2d_buffers = Kokkos::create_mirror_view(m_send_2d_buffers);
  auto h_recv_2d_buffers = Kokkos::create_mirror_view(m_recv_2d_buffers);
  auto h_send_3d_buffers = Kokkos::create_mirror_view(m_send_3d_buffers);
  auto h_recv_3d_buffers = Kokkos::create_mirror_view(m_recv_3d_buffers);
  auto h_connections = m_connectivity.get_connections<HostMemSpace>();
  for (int ie=0; ie<m_num_elements; ++ie) {
    for (int iconn=0; iconn<NUM_CONNECTIONS; ++iconn) {
      const ConnectionInfo& info = h_connections(ie,iconn);

      const LidPos local  = info.local;

      auto send_buffer = h_all_send_buffers[info.sharing];
      auto recv_buffer = h_all_recv_buffers[info.sharing];

      for (int ifield=0; ifield<m_num_2d_fields; ++ifield) {
        h_send_2d_buffers(local.lid,ifield,local.pos) = ExecViewUnmanaged<Real*>(send_buffer.get() + h_buf_offset[info.sharing],helpers.CONNECTION_SIZE[info.kind]);
        h_recv_2d_buffers(local.lid,ifield,local.pos) = ExecViewUnmanaged<Real*>(recv_buffer.get() + h_buf_offset[info.sharing],helpers.CONNECTION_SIZE[info.kind]);
        h_buf_offset[info.sharing] += h_increment[info.kind];
      }
      for (int ifield=0; ifield<m_num_3d_fields; ++ifield) {
        h_send_3d_buffers(local.lid,ifield,local.pos) = ExecViewUnmanaged<Scalar*[NUM_LEV]>(reinterpret_cast<Scalar*>(send_buffer.get() + h_buf_offset[info.sharing]),helpers.CONNECTION_SIZE[info.kind]);
        h_recv_3d_buffers(local.lid,ifield,local.pos) = ExecViewUnmanaged<Scalar*[NUM_LEV]>(reinterpret_cast<Scalar*>(recv_buffer.get() + h_buf_offset[info.sharing]),helpers.CONNECTION_SIZE[info.kind]);
        h_buf_offset[info.sharing] += h_increment[info.kind]*NUM_LEV*VECTOR_SIZE;
      }
    }
  }
  Kokkos::deep_copy(m_send_2d_buffers,h_send_2d_buffers);
  Kokkos::deep_copy(m_send_3d_buffers,h_send_3d_buffers);
  Kokkos::deep_copy(m_recv_2d_buffers,h_recv_2d_buffers);
  Kokkos::deep_copy(m_recv_3d_buffers,h_recv_3d_buffers);

#ifndef NDEBUG
  // Sanity check
  assert (h_buf_offset[etoi(ConnectionSharing::LOCAL)]==local_buffer_size);
  assert (h_buf_offset[etoi(ConnectionSharing::SHARED)]==mpi_buffer_size);
#endif // NDEBUG

  // Create persistend send/recv requests, to reuse over and over
  build_requests ();

  // Prohibit further registration of fields, and allow exchange
  m_registration_started   = false;
  m_registration_completed = true;
}

void BoundaryExchange::exchange ()
{
  // Check that the registration has completed first
  assert (m_registration_completed);

  // Hey, if some process can already send me stuff while I'm still packing, that's ok
  HOMMEXX_MPI_CHECK_ERROR(MPI_Startall(m_connectivity.get_num_shared_connections<HostMemSpace>(),m_recv_requests.data()));

  // Make sure the send requests are inactive (can't reuse buffers otherwise)
  // TODO: figure out why MPI_Waitall does not work. If the requests are all inactive, MPI_Waitall
  //       should return immediately. Instead, it appears to hang.
  int all_done = 0;
  while (all_done==0) {
    HOMMEXX_MPI_CHECK_ERROR(MPI_Testall(m_connectivity.get_num_shared_connections<HostMemSpace>(),m_send_requests.data(),&all_done,MPI_STATUSES_IGNORE));
  }
  //HOMMEXX_MPI_CHECK_ERROR(MPI_Waitall(m_connectivity.get_num_shared_connections<HostMemSpace>(),m_send_requests.data(),MPI_STATUSES_IGNORE));

  // ---- Pack and send ---- //
  pack_and_send ();

  // --- Recv and unpack --- //
  recv_and_unpack ();
}

void BoundaryExchange::build_requests()
{
  // TODO: we could make this for into parallel_for if we want. But it is just a setup cost.
  auto connections = m_connectivity.get_connections<HostMemSpace>();
  int buf_offset = 0;
  int irequest   = 0;
  for (int ie=0; ie<m_num_elements; ++ie) {
    for (int iconn=0; iconn<NUM_CONNECTIONS; ++iconn) {
      const ConnectionInfo& info = connections(ie,iconn);
      if (info.sharing!=etoi(ConnectionSharing::SHARED)) {
        continue;
      }

      // We build a tag that has info about the sender's element lid and connection position.
      // Since there are 8 neighbors, an easy way is to set tag=lid*8+pos
      int send_tag = info.local.lid*NUM_CONNECTIONS + info.local.pos;
      int recv_tag = info.remote.lid*NUM_CONNECTIONS + info.remote.pos;

      // Reserve the area in the buffers and update the offset
      MPIViewManaged<Real*>::pointer_type send_ptr = m_mpi_send_buffer.data() + buf_offset;
      MPIViewManaged<Real*>::pointer_type recv_ptr = m_mpi_recv_buffer.data() + buf_offset;
      buf_offset += m_elem_buf_size[info.kind];

      // Create the persistent requests
      HOMMEXX_MPI_CHECK_ERROR(MPI_Send_init(send_ptr,1,m_mpi_data_type[info.kind],info.remote_pid,send_tag,m_comm.m_mpi_comm,&m_send_requests[irequest]));
      HOMMEXX_MPI_CHECK_ERROR(MPI_Recv_init(recv_ptr,1,m_mpi_data_type[info.kind],info.remote_pid,recv_tag,m_comm.m_mpi_comm,&m_recv_requests[irequest]));

      // Increment the request counter;
      ++irequest;
    }
  }
}

void BoundaryExchange::pack_and_send ()
{
  // NOTE: all of these temporary copies are necessary because of the issue of lambda function not
  //       capturing the this pointer correctly on the device.
  auto connections = m_connectivity.get_connections<ExecMemSpace>();
  auto fields_2d = m_2d_fields;
  auto fields_3d = m_3d_fields;
  auto send_2d_buffers = m_send_2d_buffers;
  auto send_3d_buffers = m_send_3d_buffers;

  // ---- Pack ---- //
  // First, pack 2d fields...
  Kokkos::parallel_for(MDRangePolicy<ExecSpace,3>({0,0,0},{m_num_elements,NUM_CONNECTIONS,m_num_2d_fields},{1,1,1}),
                       KOKKOS_LAMBDA(const int ie, const int iconn, const int ifield) {
    ConnectionHelpers helpers;

    const ConnectionInfo info = connections(ie,iconn);
    const LidPos field_lidpos  = info.local;
    // For the buffer, in case of local connection, use remote info. In fact, while with shared connections the
    // mpi call will take care of "copying" data to the remote recv buffer in the correct remote element lid,
    // for local connections we need to manually copy on the remote element lid. We can do it here
    const LidPos buffer_lidpos = info.sharing==etoi(ConnectionSharing::LOCAL) ? info.remote : info.local;

    // Note: if it is an edge and the remote edge is in the reverse order, we read the field_lidpos points backwards
    const auto& pts = helpers.CONNECTION_PTS[info.direction][field_lidpos.pos];
    for (int k=0; k<helpers.CONNECTION_SIZE[info.kind]; ++k) {
      send_2d_buffers(buffer_lidpos.lid,ifield,buffer_lidpos.pos)(k) = fields_2d(field_lidpos.lid,ifield)(pts[k].ip,pts[k].jp);
    }
  });
  // ...then pack 3d fields.
  Kokkos::parallel_for(MDRangePolicy<ExecSpace,4>({0,0,0,0},{m_num_elements,NUM_CONNECTIONS,m_num_3d_fields,NUM_LEV},{1,1,1,1}),
                       KOKKOS_LAMBDA(const int ie, const int iconn, const int ifield, const int ilev) {
    ConnectionHelpers helpers;

    const ConnectionInfo info = connections(ie,iconn);
    const LidPos field_lidpos  = info.local;
    // For the buffer, in case of local connection, use remote info. In fact, while with shared connections the
    // mpi call will take care of "copying" data to the remote recv buffer in the correct remote element lid,
    // for local connections we need to manually copy on the remote element lid. We can do it here
    const LidPos buffer_lidpos = info.sharing==etoi(ConnectionSharing::LOCAL) ? info.remote : info.local;

    // Note: if it is an edge and the remote edge is in the reverse order, we read the field_lidpos points backwards
    const auto& pts = helpers.CONNECTION_PTS[info.direction][field_lidpos.pos];
    for (int k=0; k<helpers.CONNECTION_SIZE[info.kind]; ++k) {
      send_3d_buffers(buffer_lidpos.lid,ifield,buffer_lidpos.pos)(k,ilev) = fields_3d(field_lidpos.lid,ifield)(pts[k].ip,pts[k].jp,ilev);
    }
  });
  ExecSpace::fence();

  // ---- Send ---- //
  Kokkos::deep_copy(m_mpi_send_buffer, m_send_buffer); // Deep copy m_send_buffer into m_mpi_send_buffer (no op if MPI is on device)
  HOMMEXX_MPI_CHECK_ERROR(MPI_Startall(m_connectivity.get_num_shared_connections<HostMemSpace>(),m_send_requests.data())); // Fire off the sends
}

void BoundaryExchange::recv_and_unpack ()
{
  // ---- Recv ---- //
  HOMMEXX_MPI_CHECK_ERROR(MPI_Waitall(m_connectivity.get_num_shared_connections<HostMemSpace>(),m_recv_requests.data(),MPI_STATUSES_IGNORE)); // Wait for all data to arrive
  Kokkos::deep_copy(m_recv_buffer, m_mpi_recv_buffer); // Deep copy m_mpi_recv_buffer into m_recv_buffer (no op if MPI is on device)

  // NOTE: all of these temporary copies are necessary because of the issue of lambda function not
  //       capturing the this pointer correctly on the device.
  auto connections = m_connectivity.get_connections<ExecMemSpace>();
  auto fields_2d = m_2d_fields;
  auto fields_3d = m_3d_fields;
  auto recv_2d_buffers = m_recv_2d_buffers;
  auto recv_3d_buffers = m_recv_3d_buffers;

  // --- Unpack --- //
  // First, unpack 2d fields...
  Kokkos::parallel_for(MDRangePolicy<ExecSpace,2>({0,0},{m_num_elements,m_num_2d_fields},{1,1}),
                       KOKKOS_LAMBDA(const int ie, const int ifield) {
    ConnectionHelpers helpers;

    for (int k=0; k<NP; ++k) {
      for (int iedge : helpers.UNPACK_EDGES_ORDER) {
        fields_2d(ie,ifield)(helpers.CONNECTION_PTS_FWD[iedge][k].ip,helpers.CONNECTION_PTS_FWD[iedge][k].jp) += recv_2d_buffers(ie,ifield,iedge)[k];
      }
    }
    for (int icorner : helpers.UNPACK_CORNERS_ORDER) {
      if (recv_2d_buffers(ie,ifield,icorner).size() > 0) {
        fields_2d(ie,ifield)(helpers.CONNECTION_PTS_FWD[icorner][0].ip,helpers.CONNECTION_PTS_FWD[icorner][0].jp) += recv_2d_buffers(ie,ifield,icorner)[0];
      }
    }
  });
  // ...then unpack 3d fields.
  Kokkos::parallel_for(MDRangePolicy<ExecSpace,3>({0,0,0},{m_num_elements,m_num_3d_fields,NUM_LEV},{1,1,1}),
                       KOKKOS_LAMBDA(const int ie, const int ifield, const int ilev) {
    ConnectionHelpers helpers;

    for (int k=0; k<NP; ++k) {
      for (int iedge : helpers.UNPACK_EDGES_ORDER) {
        fields_3d(ie,ifield)(helpers.CONNECTION_PTS_FWD[iedge][k].ip,helpers.CONNECTION_PTS_FWD[iedge][k].jp,ilev) += recv_3d_buffers(ie,ifield,iedge)(k,ilev);
      }
    }
    for (int icorner : helpers.UNPACK_CORNERS_ORDER) {
      if (recv_3d_buffers(ie,ifield,icorner).size() > 0)
        fields_3d(ie,ifield)(helpers.CONNECTION_PTS_FWD[icorner][0].ip,helpers.CONNECTION_PTS_FWD[icorner][0].jp,ilev) += recv_3d_buffers(ie,ifield,icorner)(0,ilev);
    }
  });
  ExecSpace::fence();
}

} // namespace Homme
