#include "Connectivity.hpp"

#include <algorithm>

namespace Homme
{

bool operator<(const ConnectionInfo& a, const ConnectionInfo& b)
{
  if (a.remote_pid==-1 && b.remote_pid>=0) {
    // Local connections come first, then shared
    return true;
  } else if (b.remote_pid==-1 && a.remote_pid>=0) {
    // Local connections come first, then shared
    return false;
  } else if (a.kind!=b.kind) {
    // If both local or both shared with the same pid, put corner connections first
    return a.kind==CORNER_KIND;
  } else if (a.local.lid!=b.local.lid){
    // Within corner/edges, order by local element lid
    return a.local.lid<b.local.lid;
  } else {
    // If same connection type within the same element, oder by neighbor position
    return a.local.pos<b.local.pos;
  }
}

Connectivity::Connectivity ()
 : m_finalized (false)
 , m_num_my_elems (-1)
{
  // TODO: change this to allow integration of Hommexx in acme, and
  //       pass the number of processes dedicated to homme
  m_comm.init();
}

void Connectivity::set_num_my_elems (const int num_my_elems)
{
  m_num_my_elems = num_my_elems;
}

void Connectivity::set_num_connections (const int num_local_connections, const int num_shared_connections)
{
  m_connections.resize(num_local_connections+num_shared_connections);

  // We do not set these two. We will use them as progressing index, updating them as we add connections.
  // When finalize() is called, we will set them to match the size of the corresponding vector.
  m_num_connections = 0;
  m_num_local_connections  = 0;
  m_num_shared_connections = 0;
  m_num_local_corner_connections  = 0;
  m_num_local_edge_connections    = 0;
  m_num_shared_corner_connections = 0;
  m_num_shared_edge_connections   = 0;
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
    // Check we are not adding more connections than we promised
    assert (static_cast<size_t>(m_num_connections)<m_connections.size());

    // There is no edge-to-corner connection. Either the elements share a corner or an edge.
    assert (IS_EDGE_NEIGHBOR[first_elem_pos]==IS_EDGE_NEIGHBOR[second_elem_pos]);
#endif

    // Recall: corner=0, edge_contiguous_fwd=1, edge_contiguous_bwd=2, edge_strided_fwd=3, edge_strided_bwd=4
    int is_edge    = IS_EDGE_NEIGHBOR[first_elem_pos];
    int direction  = is_edge==1 ? NEIGHBOR_EDGE_DIRECTION[first_elem_pos][second_elem_pos] : 0;

    ConnectionInfo& info = m_connections[m_num_connections];
    LidPosType& local  = info.local;
    LidPosType& remote = info.remote;

    // Local and remote elements lid and connection position
    local.lid  = first_elem_lid;
    local.pos  = first_elem_pos % NUM_EDGES;
    remote.lid = second_elem_lid;
    remote.pos = second_elem_pos % NUM_EDGES;

    // Kind
    info.kind = is_edge==1 ? EDGE_KIND : CORNER_KIND;

    // Direction and remote element pid (set to -1 for now)
    info.remote_pid = -1;
    info.direction  = direction;

    if (second_elem_pid!=m_comm.m_rank)
    {
      // It is an connection between different processes. We need some extra info
      // Local and remote connection type
      local.type = info.kind==CORNER_KIND ?
                     CORNER :
                     (IS_STRIDED_EDGE[local.pos] ?
                        EDGE_STRIDED:
                        EDGE_CONTIGUOUS_FWD);

      remote.type = info.kind==CORNER_KIND ?
                      CORNER :
                        (direction==DIRECTION_FWD ? EDGE_CONTIGUOUS_FWD : EDGE_CONTIGUOUS_BWD);

      // Remote element pid
      info.remote_pid = second_elem_pid;

    }
    ++m_num_connections;
  }
}

void Connectivity::finalize()
{
  // We sort the connections, so we can easily subview them by type (shared/local and corner/edge)
  std::sort(m_connections.begin(),m_connections.end());

  // Count local/shared connections and, within local/shared, count corner/edge connections
  for (const ConnectionInfo& info : m_connections) {
    if (info.remote_pid==-1) {
      ++(info.kind==CORNER_KIND ? m_num_local_corner_connections : m_num_local_edge_connections);
      ++m_num_local_connections;
    } else {
      ++(info.kind==CORNER_KIND ? m_num_shared_corner_connections : m_num_shared_edge_connections);
      ++m_num_shared_connections;
    }
  }

  m_local_connections  = HostViewUnmanaged<const ConnectionInfo*>(m_connections.data(),m_num_local_connections);
  m_shared_connections = HostViewUnmanaged<const ConnectionInfo*>(m_connections.data()+m_num_local_connections,m_num_shared_connections);

  m_local_corner_connections  = HostViewUnmanaged<const ConnectionInfo*>(m_local_connections.data(),m_num_local_corner_connections);
  m_local_edge_connections    = HostViewUnmanaged<const ConnectionInfo*>(m_local_connections.data()+m_num_local_corner_connections,m_num_local_edge_connections);
  m_shared_corner_connections = HostViewUnmanaged<const ConnectionInfo*>(m_shared_connections.data(),m_num_shared_corner_connections);
  m_shared_edge_connections   = HostViewUnmanaged<const ConnectionInfo*>(m_shared_connections.data()+m_num_shared_corner_connections,m_num_shared_edge_connections);

  m_finalized = true;
}

void Connectivity::clean_up()
{
  m_local_corner_connections  = HostViewUnmanaged<const ConnectionInfo*>(nullptr,0);
  m_local_edge_connections    = HostViewUnmanaged<const ConnectionInfo*>(nullptr,0);
  m_shared_corner_connections = HostViewUnmanaged<const ConnectionInfo*>(nullptr,0);
  m_shared_edge_connections   = HostViewUnmanaged<const ConnectionInfo*>(nullptr,0);

  m_local_connections  = HostViewUnmanaged<const ConnectionInfo*>(nullptr,0);
  m_shared_connections = HostViewUnmanaged<const ConnectionInfo*>(nullptr,0);

  m_connections.resize(0);

  m_num_connections = 0;
  m_num_local_connections = m_num_shared_connections = 0;
  m_num_local_edge_connections = m_num_local_corner_connections = 0;
  m_num_shared_edge_connections = m_num_shared_corner_connections = 0;

  m_finalized = false;
}

Connectivity& get_connectivity()
{
  static Connectivity c;
  return c;
}

} // namespace Homme
