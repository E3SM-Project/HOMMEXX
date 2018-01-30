#ifndef HOMMEXX_ENUMS_HPP
#define HOMMEXX_ENUMS_HPP

#include "Kokkos_Core.hpp"

namespace Homme
{

// Convert strong typed enum to the underlying int value
// TODO: perhaps move this to Utility.hpp
template<typename E>
constexpr
KOKKOS_FORCEINLINE_FUNCTION
typename std::underlying_type<E>::type etoi(E e) {
  return static_cast<typename std::underlying_type<E>::type>(e);
}

// =================== Run parameters enums ====================== //

enum class MoistDry {
  MOIST,
  DRY
};

enum class RemapAlg {
  PPM_MIRRORED = 1,
  PPM_FIXED    = 2
};

enum class TestCase {
  ASP_BAROCLINIC,
  ASP_GRAVITY_WAVE,
  ASP_MOUNTAIN,
  ASP_ROSSBY,
  ASP_TRACER,
  BAROCLINIC,
  DCMIP2012_TEST1_1,
  DCMIP2012_TEST1_2,
  DCMIP2012_TEST1_3,
  DCMIP2012_TEST2_0,
  DCMIP2012_TEST2_1,
  DCMIP2012_TEST2_2,
  DCMIP2012_TEST3,
  HELD_SUAREZ0,
  JW_BAROCLINIC
};

// =================== Mesh connectivity enums ====================== //

// The kind of connection: edge, corner or missing (one of the corner connections on one of the 8 cube vertices)
enum class ConnectionKind : int {
  EDGE    = 0,
  CORNER  = 1,
  MISSING = 2,  // Used to detect missing connections
  ANY     = 3   // Used when the kind of connection is not needed
};

// The locality of connection: local, shared or missing
enum class ConnectionSharing : int {
  LOCAL   = 0,
  SHARED  = 1,
  MISSING = 2,  // Used to detect missing connections
  ANY     = 3   // Used when the kind of connection is not needed
};

enum class ConnectionName : int {
  // Edges
  WEST  = 0,
  EAST  = 1,
  SOUTH = 2,
  NORTH = 3,

  // Corners
  SWEST = 4,
  SEAST = 5,
  NWEST = 6,
  NEAST = 7
};

// Direction (useful only for an edge)
constexpr int NUM_DIRECTIONS = 3;
enum class Direction : int {
  FORWARD  = 0,
  BACKWARD = 1,
  INVALID  = 2
};


} // namespace Homme

#endif // HOMMEXX_ENUMS_HPP
