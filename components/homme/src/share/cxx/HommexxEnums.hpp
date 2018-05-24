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

// ============== Run options check utility enum ================== //

namespace Errors {

enum class ComparisonOp {
  EQ = 0,   // EQUAL
  NE,       // NOT EQUAL
  GT,       // GREATHER THAN
  LT,       // LESS THAN
  GE,       // GREATHER THAN OR EQUAL
  LE        // LESS THAN OR EQUAL
};

} // namespace Errors

// =================== Run parameters enums ====================== //

enum class MoistDry {
  MOIST,
  DRY
};

enum class RemapAlg {
  PPM_MIRRORED = 1,
  PPM_FIXED_PARABOLA = 2,
  PPM_FIXED_MEANS = 3,
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

enum class UpdateType {
  LEAPFROG,
  FORWARD
};

// =================== Euler Step DSS Option ====================== //

enum class DSSOption {
  ETA,
  OMEGA,
  DIV_VDP_AVE
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
  SOUTH = 0,
  NORTH = 1,
  WEST  = 2,
  EAST  = 3,

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
