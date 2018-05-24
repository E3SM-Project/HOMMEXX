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

#ifndef HOMMEXX_CONNECTIVITY_HELPERS_HPP
#define HOMMEXX_CONNECTIVITY_HELPERS_HPP

#include "Dimensions.hpp"
#include "Types.hpp"
#include "HommexxEnums.hpp"

#include <Kokkos_Array.hpp>

namespace Homme
{

// +--------------------------------------------------------------------------------------------------------------+
// |                                                                                                              |
// |                                  REMARKS ON BOUNDARY EXCHANGE                                                |
// |                                                                                                              |
// | Each element has 8 neighbors: west(W), east(E), south(S), north(N),                                          |
// |                               south-west(SW), south-east(SE), north-west(NW), north-east(NE)                 |
// | The first 4 correspond to the neighbors sharing a full edge with this element,                               |
// | while the latters correspond to the neighbors sharing only a corner with this element.                       |
// | NOTE: if the # of elements on each face of the cube sphere (ne*ne in homme) is greater than one, then        |
// | there are 24 elements that miss one corner neighbor. These are the element that touch one                    |
// | of the cube vertices (if ne=1, then all elements miss all the corner neighbors).                             |
// | The numeration of dofs on each element is the following                                                      |
// |                                                                                                              |
// |  (NW)     -(N)->   (NE)                                                                                      |
// |      12--13--14--15                                                                                          |
// |       |   |   |   |                                                                                          |
// |       |   |   |   |                                                                                          |
// |    ^  8---9--10--11  ^                                                                                       |
// |    |  |   |   |   |  |                                                                                       |
// |   (W) |   |   |   | (E)                                                                                      |
// |    |  4---5---6---7  |                                                                                       |
// |       |   |   |   |                                                                                          |
// |       |   |   |   |                                                                                          |
// |       0---1---2---3                                                                                          |
// |  (SW)     -(S)->   (SE)                                                                                      |
// |                                                                                                              |
// | The arrows around an edge neighbor refer to the 'natural' ordering of the dofs within an edge.               |
// | From the picture we can immediately state the following:                                                     |
// |                                                                                                              |
// |  1) edge neighbors contain 4 points/dofs, while corners contain only 1                                       |
// |  2) the S/N edges store the points contiguously, while the W/E points are strided (stride is NP)             |
// |     (Note: this is relevant only if we decide to switch to RMA, avoiding buffers, and copying                |
// |            data directly to/from host views)                                                                 |
// |                                                                                                              |
// | In addition, we need to understand that the local ordering of edge points may differ on two                  |
// | neighboring elements. For instance, consider two elements sharing their S edge. This is depicted             |
// | in the following:                                                                                            |
// |                                                                                                              |
// |          elem 1                                                                                              |
// |       0---1---2---3                                                                                          |
// |                                                                                                              |
// |       3---2---1---0                                                                                          |
// |          elem 2                                                                                              |
// |                                                                                                              |
// | So the 1st dof on the S edge of elem 1 does not correspond to the 1st dof on the S edge of elem 2.           |
// | This always happen for same-edge neighbors (W/W, E/E, S/S/, N/N), and also for W/N, E/S. For these           |
// | neighbors we need to store a flag marking the neighbor as 'backward' ordered. The other neighbors            |
// | (S/W, S/N, W/E, E/N) are marked as 'forward', meaning that the 1st dof on elem1's edge matches the           |
// | 1st dof on elem2's edge.                                                                                     |
// | NOTE: in F90, in this context, 'edge' means an edge of the dual mesh, i.e., a connection between elements.   |
// |       Here, we reserve the word 'edge' for the mesh edges, and we use 'connection' to refer to the 8         |
// |       possible connections with neighboring elements.                                                        |
// |                                                                                                              |
// | The W/E/S/N/SW/SE/NW/NE values from a neighboring element are stored on a buffer, to allow all MPI           |
// | operations to be over before the local view is updated. The values are then summed into the local field      |
// | view in a pre-established order, to guarantee reproducibility of the accumulation.                           |
// |                                                                                                              |
// +--------------------------------------------------------------------------------------------------------------+

// ============ Constexpr counters =========== //

constexpr int NUM_CONNECTION_KINDS     = 3;
constexpr int NUM_CONNECTION_SHARINGS  = 3;
constexpr int NUM_CONNECTIONS_PER_KIND = 4;

constexpr int NUM_CORNERS     = NUM_CONNECTIONS_PER_KIND;
constexpr int NUM_EDGES       = NUM_CONNECTIONS_PER_KIND;
constexpr int NUM_CONNECTIONS = NUM_CORNERS + NUM_EDGES;

// =========== A simple type for a Gauss Point =========== //

// A simple struct to store i,j indices of a gauss point. This is much like an std::pair,
// but with shorter and more meaningful member names than 'first' and 'second'.
// Note: we want to allow aggregate initialization, so no explitit constructors (and no non-static methods)!
struct GaussPoint
{
  int ip;   // i
  int jp;   // j
};
using ArrayGP = Kokkos::Array<GaussPoint,NP>;

// =========== A container struct for the information about connections =========== //

// Here we define a bunch of conxtexpr int's and arrays (of arrays (of arrays)) of ints, which we can
// use to easily retrieve information about a connection, such as the kind (corner or edge), the ordering
// on the remote (only relevant for edges), the (i,j) coordinates of the Gauss point(s) in the connection,
// and more.
struct ConnectionHelpers {

  ConnectionHelpers () {}

  ConnectionHelpers& operator= (const ConnectionHelpers& src) { return *this; }

  // Unpacking edges in the following order: S, N, W, E. For corners, order doesn't really matter
  const int UNPACK_EDGES_ORDER  [NUM_EDGES]   = { etoi(ConnectionName::SOUTH), etoi(ConnectionName::NORTH), etoi(ConnectionName::WEST),  etoi(ConnectionName::EAST) };
  const int UNPACK_CORNERS_ORDER[NUM_CORNERS] = { etoi(ConnectionName::SWEST), etoi(ConnectionName::SEAST), etoi(ConnectionName::NWEST), etoi(ConnectionName::NEAST)};

  const int CONNECTION_SIZE[NUM_CONNECTION_KINDS] = {
    NP,   // EDGE
    1,    // CORNER
    0     // MISSING (for completeness, but probably never used)
  };

  const ConnectionKind CONNECTION_KIND[NUM_CONNECTIONS] = {
      ConnectionKind::EDGE,     // S
      ConnectionKind::EDGE,     // N
      ConnectionKind::EDGE,     // W
      ConnectionKind::EDGE,     // E
      ConnectionKind::CORNER,   // SW
      ConnectionKind::CORNER,   // SE
      ConnectionKind::CORNER,   // NW
      ConnectionKind::CORNER    // NE
  };

  const Direction CONNECTION_DIRECTION[NUM_CONNECTIONS][NUM_CONNECTIONS] = {
    {Direction::BACKWARD, Direction::FORWARD , Direction::FORWARD,  Direction::BACKWARD, Direction::INVALID, Direction::INVALID, Direction::INVALID, Direction::INVALID}, // S/(S-N-W-E)
    {Direction::FORWARD,  Direction::BACKWARD, Direction::BACKWARD, Direction::FORWARD,  Direction::INVALID, Direction::INVALID, Direction::INVALID, Direction::INVALID}, // N/(S-N-W-E)
    {Direction::FORWARD,  Direction::BACKWARD, Direction::BACKWARD, Direction::FORWARD,  Direction::INVALID, Direction::INVALID, Direction::INVALID, Direction::INVALID}, // W/(S-N-W-E)
    {Direction::BACKWARD, Direction::FORWARD , Direction::FORWARD,  Direction::BACKWARD, Direction::INVALID, Direction::INVALID, Direction::INVALID, Direction::INVALID}, // E/(S-N-W-E)
    {Direction::INVALID,  Direction::INVALID,  Direction::INVALID,  Direction::INVALID,  Direction::FORWARD, Direction::FORWARD, Direction::FORWARD, Direction::FORWARD},
    {Direction::INVALID,  Direction::INVALID,  Direction::INVALID,  Direction::INVALID,  Direction::FORWARD, Direction::FORWARD, Direction::FORWARD, Direction::FORWARD},
    {Direction::INVALID,  Direction::INVALID,  Direction::INVALID,  Direction::INVALID,  Direction::FORWARD, Direction::FORWARD, Direction::FORWARD, Direction::FORWARD},
    {Direction::INVALID,  Direction::INVALID,  Direction::INVALID,  Direction::INVALID,  Direction::FORWARD, Direction::FORWARD, Direction::FORWARD, Direction::FORWARD}
  };

  // We only need 12 out of these 16, but for clarity, we define them all, plus an invalid one
  const GaussPoint GP_0       {  0,  0 };
  const GaussPoint GP_1       {  0,  1 };
  const GaussPoint GP_2       {  0,  2 };
  const GaussPoint GP_3       {  0,  3 };
  const GaussPoint GP_4       {  1,  0 };
  const GaussPoint GP_5       {  1,  1 };
  const GaussPoint GP_6       {  1,  2 };
  const GaussPoint GP_7       {  1,  3 };
  const GaussPoint GP_8       {  2,  0 };
  const GaussPoint GP_9       {  2,  1 };
  const GaussPoint GP_10      {  2,  2 };
  const GaussPoint GP_11      {  2,  3 };
  const GaussPoint GP_12      {  3,  0 };
  const GaussPoint GP_13      {  3,  1 };
  const GaussPoint GP_14      {  3,  2 };
  const GaussPoint GP_15      {  3,  3 };
  const GaussPoint GP_INVALID { -1, -1 };

  const ArrayGP SOUTH_PTS_FWD = {{ GP_0 , GP_1 , GP_2 , GP_3  }};
  const ArrayGP NORTH_PTS_FWD = {{ GP_12, GP_13, GP_14, GP_15 }};
  const ArrayGP WEST_PTS_FWD  = {{ GP_0 , GP_4 , GP_8 , GP_12 }};
  const ArrayGP EAST_PTS_FWD  = {{ GP_3 , GP_7 , GP_11, GP_15 }};

  const ArrayGP SOUTH_PTS_BWD = {{ GP_3 , GP_2 , GP_1 , GP_0  }};
  const ArrayGP NORTH_PTS_BWD = {{ GP_15, GP_14, GP_13, GP_12 }};
  const ArrayGP WEST_PTS_BWD  = {{ GP_12, GP_8 , GP_4 , GP_0  }};
  const ArrayGP EAST_PTS_BWD  = {{ GP_15, GP_11, GP_7 , GP_3  }};

  const ArrayGP SWEST_PTS = {{ GP_0 , GP_INVALID, GP_INVALID, GP_INVALID }};
  const ArrayGP SEAST_PTS = {{ GP_3 , GP_INVALID, GP_INVALID, GP_INVALID }};
  const ArrayGP NWEST_PTS = {{ GP_12, GP_INVALID, GP_INVALID, GP_INVALID }};
  const ArrayGP NEAST_PTS = {{ GP_15, GP_INVALID, GP_INVALID, GP_INVALID }};

  const ArrayGP NO_PTS = {{ }}; // Used as a placeholder later on

  // Now we pack all the connection points

  // Connections fwd
  const ArrayGP CONNECTION_PTS_FWD [NUM_CONNECTIONS] =
    { SOUTH_PTS_FWD, NORTH_PTS_FWD, WEST_PTS_FWD, EAST_PTS_FWD, SWEST_PTS, SEAST_PTS, NWEST_PTS, NEAST_PTS };

  // Connections bwd
  const ArrayGP CONNECTION_PTS_BWD [NUM_CONNECTIONS] =
    { SOUTH_PTS_BWD, NORTH_PTS_BWD, WEST_PTS_BWD, EAST_PTS_BWD, SWEST_PTS, SEAST_PTS, NWEST_PTS, NEAST_PTS };

  // All connections
  // You should never access CONNECTIONS_PTS with Direction=INVALID
  const ArrayGP CONNECTION_PTS[NUM_DIRECTIONS][NUM_CONNECTIONS] =
    {
      { SOUTH_PTS_FWD, NORTH_PTS_FWD, WEST_PTS_FWD, EAST_PTS_FWD, SWEST_PTS, SEAST_PTS, NWEST_PTS, NEAST_PTS },
      { SOUTH_PTS_BWD, NORTH_PTS_BWD, WEST_PTS_BWD, EAST_PTS_BWD, SWEST_PTS, SEAST_PTS, NWEST_PTS, NEAST_PTS },
      { NO_PTS }
    };

  // Edges and corners (fwd), used in the unpacking
  const ArrayGP EDGE_PTS_FWD [NUM_CONNECTIONS_PER_KIND] =
    { SOUTH_PTS_FWD, NORTH_PTS_FWD, WEST_PTS_FWD, EAST_PTS_FWD };

  const ArrayGP CORNER_PTS_FWD [NUM_CONNECTIONS_PER_KIND] =
    { SWEST_PTS, SEAST_PTS, NWEST_PTS, NEAST_PTS};
};

} // namespace Homme

#endif // HOMMEXX_CONNECTIVITY_HELPERS_HPP
