#include "Connectivity.hpp"

#include <algorithm>

namespace Homme
{

Connectivity::Connectivity ()
 : m_finalized (false)
 , m_num_elements (-1)
{
  // TODO: change this to allow integration of Hommexx in acme, and
  //       pass the number of processes dedicated to homme
  m_comm.init();
}

void Connectivity::set_num_elements (const int num_elements)
{
  m_num_elements = num_elements;

  m_connections = HostViewManaged<ConnectionInfo*[NUM_CONNECTIONS]>("Connections", m_num_elements);
  m_num_connections = HostViewManaged<int[NUM_CONNECTION_SHARINGS+1][NUM_CONNECTION_KINDS+1]>("Connections counts");

  Kokkos::deep_copy (m_num_connections,0);

  // Initialize all connections to MISSING
  // Note: we still include local element/position, since we need that even for
  //       missinc connections!
  for (int ie=0; ie<m_num_elements; ++ie) {
    for (int iconn=0; iconn<NUM_CONNECTIONS; ++iconn) {
      ConnectionInfo& info = m_connections(ie,iconn);

      info.sharing = etoi(ConnectionSharing::MISSING);
      info.kind    = etoi(ConnectionKind::MISSING);

      info.local.lid = ie;
      info.local.pos = iconn;
    }
  }
}

void Connectivity::add_connection (const int first_elem_lid,  const int first_elem_pos,  const int first_elem_pid,
                                   const int second_elem_lid, const int second_elem_pos, const int second_elem_pid)
{
  if (m_finalized)
  {
    std::cerr << "Error! The connectivity has already been finalized.\n";
    std::abort();
  }

  // I believe edges appear twice in fortran, once per each ordering.
  // Here, we only need to store them once
  if (first_elem_pid==m_comm.m_rank)
  {
#ifdef HOMMEXX_DEBUG
    // There is no edge-to-corner connection. Either the elements share a corner or an edge.
    assert (CONNECTION_KIND[first_elem_pos]==CONNECTION_KIND[second_elem_pos]);
#endif

    ConnectionInfo& info = m_connections(first_elem_lid,first_elem_pos);
    LidPos& local  = info.local;
    LidPos& remote = info.remote;

    // Local and remote elements lid and connection position
    local.lid  = first_elem_lid;
    local.pos  = first_elem_pos;
    remote.lid = second_elem_lid;
    remote.pos = second_elem_pos;

    // Kind
    info.kind = etoi(CONNECTION_KIND[first_elem_pos]);


    // Direction
    info.direction  = etoi(CONNECTION_DIRECTION[local.pos][remote.pos]);

    if (second_elem_pid!=m_comm.m_rank)
    {
      info.sharing = etoi(ConnectionSharing::SHARED);
      info.remote_pid = second_elem_pid;

    } else {
      info.remote_pid = -1;
      info.sharing = etoi(ConnectionSharing::LOCAL);
    }

    ++m_num_connections(info.sharing,info.kind);
  }
}

void Connectivity::finalize()
{
  // Sanity check: Homme does not allow less than 2*2 elements per face, so each element
  // should have at most ONE missing connection
  for (int ie=0; ie<m_num_elements; ++ie) {
    bool missing[NUM_CORNERS] = {false, false, false, false};

    for (int ic=0; ic<NUM_CORNERS; ++ic) {
      if (m_connections(ie,ic+CORNERS_OFFSET).kind == etoi(ConnectionKind::MISSING)) {
        missing[ic] = true;

        // Just for tracking purposes
        ++m_num_connections(etoi(ConnectionSharing::MISSING),etoi(ConnectionKind::MISSING));
      }
    }
    assert (std::count(missing,missing+NUM_CORNERS,true)<=1);
  }

  // Updating counters for groups with same sharing/kind
  for (int kind=0; kind<NUM_CONNECTION_KINDS; ++kind) {
    for (int sharing=0; sharing<NUM_CONNECTION_SHARINGS; ++sharing) {
      m_num_connections(etoi(ConnectionSharing::ANY),kind) += m_num_connections(sharing,kind);
      m_num_connections(sharing,etoi(ConnectionKind::ANY)) += m_num_connections(sharing,kind);
    }
    m_num_connections(etoi(ConnectionKind::ANY),etoi(ConnectionKind::ANY)) += m_num_connections(etoi(ConnectionSharing::ANY),kind);
  }

  m_finalized = true;
}

void Connectivity::clean_up()
{
  m_connections = HostViewManaged<ConnectionInfo*[NUM_CONNECTIONS]>("",0);

  Kokkos::deep_copy(m_num_connections,0);

  m_finalized = false;
}

Connectivity& get_connectivity()
{
  static Connectivity c;
  return c;
}

} // namespace Homme
