#ifndef HOMMEXX_CONNECTIVITY_HPP
#define HOMMEXX_CONNECTIVITY_HPP

#include "ConnectivityHelpers.hpp"

#include "Comm.hpp"

#include "Types.hpp"

namespace Homme
{
// A simple struct to store, for a connection between elements, the local id of the element
// the position, meaning which neighbor this connection refers to (W/E/S/N/SW/SE/NW/NE) and the type
// of the connection (see NEIGHBOR_TYPES in ConnectivityHelpers.hpp).
struct LidPosType
{
  int lid;
  int pos;
  int type;
};

// A simple struct, storing a connection info. In addition to LidPosType (on both local and
// remote element), it stores also whether the ordering is the same on both the element
// (relevant only for edge-type connections), and the process id of the remote element,
// which is only used if  the remote element is on a different process.
struct ConnectionInfo
{
  LidPosType local;
  LidPosType remote;

  ConnectionKind kind;

  // The following is needed only for W/E/S/N edges, in case the ordering of the NP points is different in the two elements
  int direction;  //0=forward, 1=reverse

  // This is only needed if the neighboring element is owned by a different process
  int remote_pid;
};

// The connectivity class. It stores two lists of ConnectionInfo objects, one for
// local connections (both elements on process) and one for shared connections
// (one element is on a remote process). The latter require MPI work, while the
// former can be handled locally.
class Connectivity
{
public:

  Connectivity ();

  //@name Methods
  //@{

  void set_num_my_elems (const int num_my_elems);

  // TODO: pass just the number of connections (we figure out the rest later)
  void set_num_connections (const int num_local_connections, const int num_shared_connections);

  void add_connection (const int first_elem_lid,  const int first_elem_pos,  const int first_elem_pid,
                       const int second_elem_lid, const int second_elem_pos, const int second_elem_pid);

  void finalize ();

  void clean_up ();
  //@}

  //@name Getters
  //@{
  const std::vector<ConnectionInfo>& get_connections () const { return m_connections;   }

  HostViewUnmanaged<const ConnectionInfo*> get_local_connections  () const { return m_local_connections;  }
  HostViewUnmanaged<const ConnectionInfo*> get_shared_connections () const { return m_shared_connections; }

  HostViewUnmanaged<const ConnectionInfo*> get_local_corner_connections  () const { return m_local_corner_connections;  }
  HostViewUnmanaged<const ConnectionInfo*> get_local_edge_connections    () const { return m_local_edge_connections;    }
  HostViewUnmanaged<const ConnectionInfo*> get_shared_corner_connections () const { return m_shared_corner_connections; }
  HostViewUnmanaged<const ConnectionInfo*> get_shared_edge_connections   () const { return m_shared_edge_connections;   }

  int get_num_connections               () const { return m_num_connections; }

  int get_num_shared_connections        () const { return m_num_shared_connections; }
  int get_num_local_connections         () const { return m_num_local_connections;  }

  int get_num_local_corner_connections  () const { return m_num_local_corner_connections;  }
  int get_num_local_edge_connections    () const { return m_num_local_edge_connections;  }
  int get_num_shared_corner_connections () const { return m_num_shared_corner_connections; }
  int get_num_shared_edge_connections   () const { return m_num_shared_edge_connections; }

  int get_num_my_elems                  () const { return m_num_my_elems; }

  const Comm& get_comm () const { return m_comm; }
  //@}

private:

  Comm    m_comm;

  bool    m_finalized;

  int     m_num_my_elems;

  int     m_num_connections;

  int     m_num_local_connections;
  int     m_num_shared_connections;

  int     m_num_local_corner_connections;
  int     m_num_local_edge_connections;
  int     m_num_shared_corner_connections;
  int     m_num_shared_edge_connections;

  std::vector<ConnectionInfo>  m_connections;

  HostViewUnmanaged<const ConnectionInfo*>  m_local_connections;
  HostViewUnmanaged<const ConnectionInfo*>  m_shared_connections;

  HostViewUnmanaged<const ConnectionInfo*>  m_local_corner_connections;
  HostViewUnmanaged<const ConnectionInfo*>  m_local_edge_connections;
  HostViewUnmanaged<const ConnectionInfo*>  m_shared_corner_connections;
  HostViewUnmanaged<const ConnectionInfo*>  m_shared_edge_connections;
};

Connectivity& get_connectivity();

} // namespace Homme

#endif // HOMMEXX_CONNECTIVITY_HPP
