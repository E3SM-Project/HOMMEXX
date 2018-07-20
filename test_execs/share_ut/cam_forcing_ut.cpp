
#include <catch/catch.hpp>

#include "Context.hpp"
#include "Tracers.hpp"
#include "Elements.hpp"
#include "TimeLevel.hpp"
#include "HybridVCoord.hpp"
#include "SimulationParams.hpp"
#include "KernelVariables.hpp"
#include "Types.hpp"
#include "vector/vector_pragmas.hpp"
#include "profiling.hpp"

// Fortran implementation signatures
extern "C" {

}

namespace Homme {
void tracer_forcing(
    const ExecViewUnmanaged<const Scalar * [QSIZE_D][NP][NP][NUM_LEV]> &f_q,
    const HybridVCoord &hvcoord, const TimeLevel &tl, const int &num_q,
    const MoistDry &moisture, const double &dt,
    const ExecViewManaged<Real * [NUM_TIME_LEVELS][NP][NP]> &ps_v,
    const ExecViewManaged<
        Scalar * [Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]> &qdp,
    const ExecViewManaged<Scalar * [QSIZE_D][NP][NP][NUM_LEV]> &Q);

void state_forcing(
    const ExecViewUnmanaged<const Scalar * [NP][NP][NUM_LEV]> &f_t,
    const ExecViewUnmanaged<const Scalar * [2][NP][NP][NUM_LEV]> &f_m,
    const TimeLevel &tl, const Real &dt,
    const ExecViewUnmanaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> &t,
    const ExecViewUnmanaged<Scalar * [NUM_TIME_LEVELS][2][NP][NP][NUM_LEV]> &v);
}

TEST_CASE("cam_forcing", "apply_cam_forcing") {}
