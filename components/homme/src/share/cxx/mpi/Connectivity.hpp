#ifndef HOMMEXX_CONNECTIVITY_HPP
#define HOMMEXX_CONNECTIVITY_HPP

#include "ConnectivityHelpers.hpp"

#include "Comm.hpp"

#include "Types.hpp"

namespace Homme
{
// A simple struct to store, for a connection between elements, the local id of the element
// the position, meaning which neighbor this connection refers to (W/E/S/N/SW/SE/NW/NE)
// Much like std::pair, but with more verbose members' names
struct LidPos
{
  int lid;
  int pos;
};

// A simple struct, storing a connection info. In addition to LidPos (on both local and
// remote element), it stores also whether the ordering is the same on both the element
// (relevant only for edge-type connections), and the process id of the remote element,
// which is only used if  the remote element is on a different process.
// Note: we store kind, sharing and direction already converted to ints
struct ConnectionInfo
{
  LidPos local;
  LidPos remote;

  int kind;     // etoi(ConnectionKind::EDGE)=0, etoi(ConnectionKind::CORNER)=1,  etoi(ConnectionSharing::MISSING)=2
  int sharing;  // etoi(ConnectionSharing::LOCAL)=0, etoi(ConnectionSharing::SHARED)=1, etoi(ConnectionSharing::MISSING)=2


  // The following is needed only for W/E/S/N edges, in case the ordering of the NP points is different in the two elements
  int direction;  //0=forward, 1=backward

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

  void set_num_elements (const int num_elements);

  void add_connection (const int first_elem_lid,  const int first_elem_pos,  const int first_elem_pid,
                       const int second_elem_lid, const int second_elem_pos, const int second_elem_pid);

  void finalize ();

  void clean_up ();
  //@}

  //@name Getters
  //@{

  HostViewUnmanaged<const ConnectionInfo*[NUM_CONNECTIONS]> get_connections () const { return m_connections; }

  // Get number of connections with given kind and sharing
  int get_num_connections (const ConnectionSharing sharing, const ConnectionKind kind) const {
    return m_num_connections(etoi(sharing), etoi(kind));
  }

  // Shortcuts for common sharing/kind pairs
  int get_num_connections        () const { return get_num_connections(ConnectionSharing::ANY,   ConnectionKind::ANY); }
  int get_num_shared_connections () const { return get_num_connections(ConnectionSharing::SHARED,ConnectionKind::ANY); }
  int get_num_local_connections  () const { return get_num_connections(ConnectionSharing::LOCAL, ConnectionKind::ANY); }

  int get_num_elements           () const { return m_num_elements; }

  const Comm& get_comm () const { return m_comm; }
  //@}

private:

  Comm    m_comm;

  bool    m_finalized;

  int     m_num_elements;

  HostViewManaged<int[NUM_CONNECTION_SHARINGS+1][NUM_CONNECTION_KINDS+1]> m_num_connections;

  HostViewManaged<ConnectionInfo*[NUM_CONNECTIONS]> m_connections;
};

Connectivity& get_connectivity();

} // namespace Homme

#endif // HOMMEXX_CONNECTIVITY_HPP
