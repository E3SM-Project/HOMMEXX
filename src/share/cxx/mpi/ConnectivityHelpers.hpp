#ifndef HOMMEXX_CONNECTIVITY_HELPERS_HPP
#define HOMMEXX_CONNECTIVITY_HELPERS_HPP

#include "Dimensions.hpp"
#ifdef HOMMEXX_DEBUG
#include <assert.h>
#endif

#include <array>

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
// |  2) the S/N edges store the points contiguously, while the W/E points are strided (stride is 4=NP)           |
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
// |       Here, we reserve the word 'edge' for the mesh edges, and we use words like 'neighbor' or 'connection'  |
// |       to refer to the 8 possible connections with neighboring elements.                                      |
// |                                                                                                              |
// | The W/E/S/N edges and SW/SE/NW/NE from a neighboring element are stored on a buffer, to allow all MPI        |
// | operations to be over before the local view is updated. Edges and corners are stored on different buffers,   |
// | to allow each process to then sum the buffers in the view always in the same order, and maintain             |
// | reproducibility of the boundary exchange. The corners will be stored in a buffer with 4 entries per          |
// | element/level. The edges will be stored in a buffer with 16 entries. We decide to store the edges            |
// | horizontally in the buffer, so that each edge is stored contiguously (this may help when kokkos copies       |
// | the buffer into the field later). In other words, at each elem/level, the 4 edges are stored                 |
// | in the buffer in the following way:                                                                          |
// |                                                                                                              |
// |     -- j -- >                                                                                                |
// | |  W1 W2 W3 W4                                                                                               |
// | i  E1 E2 E3 E4                                                                                               |
// | |  S1 S2 S3 S4                                                                                               |
// | V  N1 N2 N3 N4                                                                                               |
// |                                                                                                              |
// +--------------------------------------------------------------------------------------------------------------+

// Here we define a bunch of conxtexpr int's and arrays (of arrays (of arrays)) of ints, which we can
// use to easily detect the type of neighbor (corner or edge), whether an edge is strided (W/E) or
// contiguous (S/N), whether the remote edge has a reverse direction, the idx of the first point of
// an edge, and more.

// The kind of connection: corner or edge
enum ConnectionKind
{
  CORNER_KIND = 0,
  EDGE_KIND   = 1
};

// Number of neighbors
constexpr int NUM_CORNERS = 4;
constexpr int NUM_EDGES   = 4;
constexpr int NUM_NEIGHBORS = NUM_CORNERS + NUM_EDGES;
constexpr int cornerId (const int connection) { return connection - NUM_EDGES; }

// Edges
constexpr int WEST  = 0;
constexpr int EAST  = 1;
constexpr int SOUTH = 2;
constexpr int NORTH = 3;

// Corners
constexpr int SWEST = 0;
constexpr int SEAST = 1;
constexpr int NWEST = 2;
constexpr int NEAST = 3;

// Direction (of an edge)
constexpr int DIRECTION_FWD = 0;
constexpr int DIRECTION_BWD = 1;

// We define local and remote neighbor types. This is because the way we store/read/write things on local strucutres
// is not the same as we store/read/write things on remote structures. In particular, we will have
// - local types: CORNER, EDGE_CONTIGUOUS_FWD, EDGE_STRIDED
// - remtoe types: CORNER, EDGE_CONTIGUOUS_FWD, EDGE_CONTIGUOUS_BWD
constexpr int CORNER              = 0;
constexpr int EDGE_CONTIGUOUS_FWD = 1;
constexpr int EDGE_CONTIGUOUS_BWD = 2;
constexpr int EDGE_STRIDED        = 2;

constexpr int NUM_NEIGHBOR_TYPES = 3;

//                                               W  E  S  N  SW  SE  NW  NE
constexpr int IS_EDGE_NEIGHBOR[NUM_NEIGHBORS] = {1, 1, 1, 1, 0,  0,  0,  0};
constexpr int IS_STRIDED_EDGE[NUM_EDGES] = {1, 1, 0, 0};

constexpr int NEIGHBOR_EDGE_DIRECTION[NUM_EDGES][NUM_EDGES] = {
                                     {DIRECTION_BWD, DIRECTION_FWD, DIRECTION_FWD, DIRECTION_BWD},  // W/(W-E-S-N)
                                     {DIRECTION_FWD, DIRECTION_BWD, DIRECTION_BWD, DIRECTION_FWD},  // E/(W-E-S-N)
                                     {DIRECTION_FWD, DIRECTION_BWD, DIRECTION_BWD, DIRECTION_FWD},  // S/(W-E-S-N)
                                     {DIRECTION_BWD, DIRECTION_FWD, DIRECTION_FWD, DIRECTION_BWD}}; // N/(W-E-S-N)

// A simple struct to store i,j indices of a gauss point. This is much like an std::pair (or, better, an std::tuple),
// but with shorter and more meaningful member names than 'first' and 'second' or 'get(0)'
// Note: we want to allow aggregate initialization, so no explitit constructors (and no non-static methods)!
struct GaussPoint
{
  int ip;   // i
  int jp;   // j
  int idx;  // = ip*NP+jp
};

// We only need 12 out of these 16, but for clarity, we define them all
constexpr GaussPoint GP_0  = {0, 0,  0};
constexpr GaussPoint GP_1  = {0, 1,  1};
constexpr GaussPoint GP_2  = {0, 2,  2};
constexpr GaussPoint GP_3  = {0, 3,  3};
constexpr GaussPoint GP_4  = {1, 0,  4};
constexpr GaussPoint GP_5  = {1, 1,  5};
constexpr GaussPoint GP_6  = {1, 2,  6};
constexpr GaussPoint GP_7  = {1, 3,  7};
constexpr GaussPoint GP_8  = {2, 0,  8};
constexpr GaussPoint GP_9  = {2, 1,  9};
constexpr GaussPoint GP_10 = {2, 2, 10};
constexpr GaussPoint GP_11 = {2, 3, 11};
constexpr GaussPoint GP_12 = {3, 0, 12};
constexpr GaussPoint GP_13 = {3, 1, 13};
constexpr GaussPoint GP_14 = {3, 2, 14};
constexpr GaussPoint GP_15 = {3, 3, 15};

constexpr std::array<GaussPoint,NP> WEST_EDGE_PTS_FWD  = {{ GP_0 , GP_4 , GP_8 , GP_12 }};
constexpr std::array<GaussPoint,NP> EAST_EDGE_PTS_FWD  = {{ GP_3 , GP_7 , GP_11, GP_15 }};
constexpr std::array<GaussPoint,NP> SOUTH_EDGE_PTS_FWD = {{ GP_0 , GP_1 , GP_2 , GP_3  }};
constexpr std::array<GaussPoint,NP> NORTH_EDGE_PTS_FWD = {{ GP_12, GP_13, GP_14, GP_15 }};

constexpr std::array<GaussPoint,NP> WEST_EDGE_PTS_BWD  = {{ GP_12, GP_8 , GP_4 , GP_0  }};
constexpr std::array<GaussPoint,NP> EAST_EDGE_PTS_BWD  = {{ GP_15, GP_11, GP_7 , GP_3  }};
constexpr std::array<GaussPoint,NP> SOUTH_EDGE_PTS_BWD = {{ GP_3 , GP_2 , GP_1 , GP_0  }};
constexpr std::array<GaussPoint,NP> NORTH_EDGE_PTS_BWD = {{ GP_15, GP_14, GP_13, GP_12 }};

// Now we pack all the indices
constexpr std::array<std::array<GaussPoint,NP>,NUM_EDGES> EDGE_PTS_FWD =
          { WEST_EDGE_PTS_FWD, EAST_EDGE_PTS_FWD, SOUTH_EDGE_PTS_FWD, NORTH_EDGE_PTS_FWD};

constexpr std::array<std::array<GaussPoint,NP>,NUM_EDGES> EDGE_PTS_BWD =
          { WEST_EDGE_PTS_BWD, EAST_EDGE_PTS_BWD, SOUTH_EDGE_PTS_BWD, NORTH_EDGE_PTS_BWD};

// Finally, we pack the two arrays (F and B) in one
constexpr std::array<std::array<GaussPoint,NP>,NUM_EDGES> EDGE_PTS[2] = {EDGE_PTS_FWD, EDGE_PTS_BWD};

// The corner gauss points
constexpr std::array<GaussPoint,NUM_CORNERS> CORNER_PTS = {GP_0, GP_3, GP_12, GP_15};

} // namespace Homme

#endif // HOMMEXX_CONNECTIVITY_HELPERS_HPP
