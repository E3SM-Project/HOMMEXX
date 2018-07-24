
#include <catch/catch.hpp>

#include "Context.hpp"
#include "Tracers.hpp"
#include "Elements.hpp"
#include "TimeLevel.hpp"
#include "HybridVCoord.hpp"
#include "SimulationParams.hpp"
#include "KernelVariables.hpp"
#include "Types.hpp"
#include "utilities/SubviewUtils.hpp"
#include "utilities/SyncUtils.hpp"
#include "utilities/TestUtils.hpp"

#include <random>

using rngAlg = std::mt19937_64;

// Fortran implementation signatures
extern "C" {
// cam_forcing_tracers
// real (kind=real_kind), intent(in) :: dt_q, ps0
// integer (kind=c_int), intent(in) :: np1, np1_qdp
// real (kind=real_kind), intent(in) :: hyai(nlev)
// real (kind=real_kind), intent(in) :: hybi(nlev)
// real (kind=real_kind), intent(in) :: hyai(nlev)
// real (kind=real_kind), intent(in) :: hybi(nlev)
// real (kind=real_kind), intent(in) :: FQ(np, np, nlev, qsize)
// real (kind=real_kind), intent(inout) :: Qdp(np, np, nlev, qsize, timelevels)
// real (kind=real_kind), intent(inout) :: ps_v(np, np, timelevels)

void cam_forcing_tracers(const double &dt_q, const double &ps0, const int &np1,
                         const int &np1_qdp, const bool &wet,
                         const double *hyai, const double *hybi,
                         const double *FQ, double *Qdp, double *ps_v);

// cam_forcing_states
// real (kind=real_kind), intent(in) :: dt_q
// integer (kind=c_int), intent(in) :: np1
// real (kind=real_kind), intent(in) :: FT(np, np, nlev)
// real (kind=real_kind), intent(in) :: FM(np, np, 2, nlev)
// real (kind=real_kind), intent(inout) :: T(np, np, nlev, timelevels)
// real (kind=real_kind), intent(inout) :: v(np, np, 2, nlev, timelevels)

void cam_forcing_states(const double &dt_q, const int &np1, const double *FT,
                        const double *FM, double *T, double *v);
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
    const int &np1, const Real &dt,
    const ExecViewUnmanaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> &t,
    const ExecViewUnmanaged<Scalar * [NUM_TIME_LEVELS][2][NP][NP][NUM_LEV]> &v);

TEST_CASE("cam_forcing", "states") {
  constexpr const int num_elems = 4;
  const Real dt_q = 1.0;
  const int np1 = 1;

  std::random_device rd;
  rngAlg engine(rd());
  std::uniform_real_distribution<Real> dist(0.125, 1000.0);

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][NP][NP]> ft_f90(
      "Temperature Forcing F90", num_elems);
  ExecViewManaged<Scalar * [NP][NP][NUM_LEV]> ft_cxx("Temperature Forcing cxx",
                                                     num_elems);
  genRandArray(ft_f90, engine, dist);
  sync_to_device(ft_f90, ft_cxx);

  HostViewManaged<Real * [NUM_PHYSICAL_LEV][2][NP][NP]> fm_f90(
      "Velocity Forcing F90", num_elems);
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]> fm_cxx("Velocity Forcing cxx",
                                                        num_elems);
  genRandArray(fm_f90, engine, dist);
  sync_to_device(fm_f90, fm_cxx);

  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]> t_f90(
      "Temperature F90", num_elems);
  ExecViewManaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> t_cxx(
      "Temperature cxx", num_elems);
  genRandArray(t_f90, engine, dist);
  sync_to_device(t_f90, t_cxx);

  HostViewManaged<Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][2][NP][NP]> v_f90(
      "Velocity F90", num_elems);
  ExecViewManaged<Scalar * [NUM_TIME_LEVELS][2][NP][NP][NUM_LEV]> v_cxx(
      "Velocity cxx", num_elems);
  genRandArray(v_f90, engine, dist);
  sync_to_device(v_f90, v_cxx);

  state_forcing(ft_cxx, fm_cxx, np1 - 1, dt_q, t_cxx, v_cxx);
  auto t_mirror = Kokkos::create_mirror_view(t_cxx);
  Kokkos::deep_copy(t_mirror, t_cxx);
  auto v_mirror = Kokkos::create_mirror_view(v_cxx);
  Kokkos::deep_copy(v_mirror, v_cxx);
  for (int ie = 0; ie < num_elems; ++ie) {
    cam_forcing_states(dt_q, np1, Homme::subview(ft_f90, ie).data(),
                       Homme::subview(fm_f90, ie).data(),
                       Homme::subview(t_f90, ie).data(),
                       Homme::subview(v_f90, ie).data());
    for (int k = 0; k < NUM_PHYSICAL_LEV; ++k) {
      const int ilev = k / VECTOR_SIZE;
      const int vlev = k % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          REQUIRE(t_mirror(ie, np1, igp, jgp, ilev)[vlev] ==
                  t_f90(ie, np1, k, igp, jgp));
          REQUIRE(v_mirror(ie, np1, 0, igp, jgp, ilev)[vlev] ==
                  v_f90(ie, np1, k, 0, igp, jgp));
          REQUIRE(v_mirror(ie, np1, 1, igp, jgp, ilev)[vlev] ==
                  v_f90(ie, np1, k, 1, igp, jgp));
        }
      }
    }
  }
}
}
