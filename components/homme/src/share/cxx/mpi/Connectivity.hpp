/*********************************************************************************
 *
 * Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC
 * (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
 * Government retains certain rights in this software.
 *
 * For five (5) years from  the United States Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide
 * license in this data to reproduce, prepare derivative works, and perform
 * publicly and display publicly, by or on behalf of the Government. There is
 * provision for the possible extension of the term of this license. Subsequent
 * to that period or any extension granted, the United States Government is
 * granted for itself and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable worldwide license in this data to reproduce, prepare derivative
 * works, distribute copies to the public, perform publicly and display publicly,
 * and to permit others to do so. The specific term of the license can be
 * identified by inquiry made to National Technology and Engineering Solutions of
 * Sandia, LLC or DOE.
 *
 * NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF
 * ENERGY, NOR NATIONAL TECHNOLOGY AND ENGINEERING SOLUTIONS OF SANDIA, LLC, NOR
 * ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY
 * LEGAL RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY
 * INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS
 * USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
 *
 * Any licensee of this software has the obligation and responsibility to abide
 * by the applicable export control laws, regulations, and general prohibitions
 * relating to the export of technical data. Failure to obtain an export control
 * license or other authority from the Government may result in criminal
 * liability under U.S. laws.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 *     - Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *     - Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimers in the documentation
 *       and/or other materials provided with the distribution.
 *     - Neither the name of Sandia Corporation,
 *       nor the names of its contributors may be used to endorse or promote
 *       products derived from this Software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ********************************************************************************/

#ifndef HOMMEXX_CONNECTIVITY_HPP
#define HOMMEXX_CONNECTIVITY_HPP

#include "ConnectivityHelpers.hpp"

#include "Comm.hpp"

#include "Types.hpp"

namespace Homme
{
// A simple struct to store, for a connection between elements, the local/global id of the element
// and the position, meaning which neighbor this connection refers to (W/E/S/N/SW/SE/NW/NE)
// Much like std::tuple, but with more verbose members' names
struct LidGidPos
{
  int lid;
  int gid;
  int pos;
};

// An invalid id
constexpr int INVALID_ID = -1;

// A simple struct, storing a connection info. In addition to LidGidPos (on both local and
// remote element), it stores also whether the ordering is the same on both the element
// (relevant only for edge-type connections), and the process id of the remote element,
// which is only used if  the remote element is on a different process.
// Note: we store kind, sharing and direction already converted to ints
struct ConnectionInfo
{
  LidGidPos local;
  LidGidPos remote;

  int kind;     // etoi(ConnectionKind::EDGE)=0, etoi(ConnectionKind::CORNER)=1,  etoi(ConnectionSharing::MISSING)=2
  int sharing;  // etoi(ConnectionSharing::LOCAL)=0, etoi(ConnectionSharing::SHARED)=1, etoi(ConnectionSharing::MISSING)=2


  // The following is needed only for W/E/S/N edges, in case the ordering of the NP points is different in the two elements
  int direction;  //0=forward, 1=backward

  // This is only needed if the neighboring element is owned by a different process
  int remote_pid; // Process id owning the other side of the connection
};

// The connectivity class. It stores two lists of ConnectionInfo objects, one for
// local connections (both elements on process) and one for shared connections
// (one element is on a remote process). The latter require MPI work, while the
// former can be handled locally.
class Connectivity
{
public:

  Connectivity ();
  Connectivity& operator= (const Connectivity& src) = default;

  //@name Methods
  //@{

  void set_comm (const Comm& comm);

  void set_num_elements (const int num_local_elements);

  void add_connection (const int first_elem_lid,  const int first_elem_gid,  const int first_elem_pos,  const int first_elem_pid,
                       const int second_elem_lid, const int second_elem_gid, const int second_elem_pos, const int second_elem_pid);

  void finalize ();

  void clean_up ();
  //@}

  //@name Getters
  //@{

  // Get the view with all connections
  template<typename MemSpace>
  KOKKOS_INLINE_FUNCTION
  typename std::enable_if<std::is_same<MemSpace,HostMemSpace>::value,
                 HostViewUnmanaged<const ConnectionInfo*[NUM_CONNECTIONS]>>::type
  get_connections () const { return h_connections; }

  template<typename MemSpace>
  KOKKOS_INLINE_FUNCTION
  typename std::enable_if<std::is_same<MemSpace,ExecMemSpace>::value && !std::is_same<ExecMemSpace,HostMemSpace>::value,
                 ExecViewUnmanaged<const ConnectionInfo*[NUM_CONNECTIONS]>>::type
  get_connections () const { return m_connections; }

  // Get a particular connection
  template<typename MemSpace>
  KOKKOS_INLINE_FUNCTION
  const ConnectionInfo& get_connection (const int ie, const int iconn) const { return get_connections<MemSpace>()(ie,iconn); }

  // Get number of connections with given kind and sharing
  template<typename MemSpace>
  KOKKOS_INLINE_FUNCTION
  typename std::enable_if<std::is_same<MemSpace,HostMemSpace>::value,int>::type
  get_num_connections (const ConnectionSharing sharing, const ConnectionKind kind) const { return h_num_connections(etoi(sharing), etoi(kind)); }

  template<typename MemSpace>
  KOKKOS_INLINE_FUNCTION
  typename std::enable_if<std::is_same<MemSpace,ExecMemSpace>::value && !std::is_same<ExecMemSpace,HostMemSpace>::value,int>::type
  get_num_connections (const ConnectionSharing sharing, const ConnectionKind kind) const { return m_num_connections(etoi(sharing), etoi(kind)); }

  // Shortcuts of the previous getter for common sharing/kind pairs
  template<typename MemSpace>
  KOKKOS_INLINE_FUNCTION
  int get_num_connections        () const { return get_num_connections<MemSpace>(ConnectionSharing::ANY,   ConnectionKind::ANY); }
  template<typename MemSpace>
  KOKKOS_INLINE_FUNCTION
  int get_num_shared_connections () const { return get_num_connections<MemSpace>(ConnectionSharing::SHARED,ConnectionKind::ANY); }
  template<typename MemSpace>
  KOKKOS_INLINE_FUNCTION
  int get_num_local_connections  () const { return get_num_connections<MemSpace>(ConnectionSharing::LOCAL, ConnectionKind::ANY); }

  int get_num_local_elements     () const { return m_num_local_elements;  }

  bool is_initialized () const { return m_initialized; }
  bool is_finalized   () const { return m_finalized;   }

  const Comm& get_comm () const { return m_comm; }
  //@}

private:

  Comm    m_comm;

  bool    m_finalized;
  bool    m_initialized;

  int     m_num_local_elements;

  ConnectionHelpers m_helpers;

  // TODO: do we need the counters on the device? It appears we never use them...
  ExecViewManaged<int[NUM_CONNECTION_SHARINGS+1][NUM_CONNECTION_KINDS+1]>             m_num_connections;
  ExecViewManaged<int[NUM_CONNECTION_SHARINGS+1][NUM_CONNECTION_KINDS+1]>::HostMirror h_num_connections;

  ExecViewManaged<ConnectionInfo*[NUM_CONNECTIONS]>             m_connections;
  ExecViewManaged<ConnectionInfo*[NUM_CONNECTIONS]>::HostMirror h_connections;
};

} // namespace Homme

#endif // HOMMEXX_CONNECTIVITY_HPP
