/********************************************************************************
 * HOMMEXX 1.0: Copyright of Sandia Corporation
 * This software is released under the BSD license
 * See the file 'COPYRIGHT' in the HOMMEXX/src/share/cxx directory
 *******************************************************************************/

#include "Context.hpp"
#include "Tracers.hpp"
#include "Elements.hpp"
#include "TimeLevel.hpp"
#include "HybridVCoord.hpp"
#include "SimulationParams.hpp"
#include "KernelVariables.hpp"
#include "vector/vector_pragmas.hpp"

namespace Homme {

// ----------------- SIGNATURES ---------------- //

void apply_cam_forcing(const Real &dt);
void tracer_forcing(const Elements &elems, const Tracers &tracers,
                    const HybridVCoord &hvcoord, const TimeLevel &tl,
                    const MoistDry &moisture, const Real &dt);
void state_forcing(const Elements &elems, const TimeLevel &tl, const Real &dt);

// -------------- IMPLEMENTATIONS -------------- //

void apply_cam_forcing(const Real &dt) {
  const Elements &elems = Context::singleton().get_elements();
  const Tracers &tracers = Context::singleton().get_tracers();
  const TimeLevel &tl = Context::singleton().get_time_level();
  const SimulationParams &sim_params =
      Context::singleton().get_simulation_params();
  const HybridVCoord &hvcoord = Context::singleton().get_hvcoord();

  tracer_forcing(elems, tracers, hvcoord, tl, sim_params.moisture, dt);
  state_forcing(elems, tl, dt);
}

void state_forcing(const Elements &elems, const TimeLevel &tl, const Real &dt) {
  const int num_e = elems.num_elems();
  Kokkos::parallel_for(
      "state temperature forcing",
      Kokkos::RangePolicy<ExecSpace>(0, num_e * NP * NP * NUM_LEV),
      KOKKOS_LAMBDA(const int & idx) {
        const int ie = ((idx / NUM_LEV) / NP) / NP;
        const int igp = ((idx / NUM_LEV) / NP) % NP;
        const int jgp = (idx / NUM_LEV) % NP;
        const int k = idx % NUM_LEV;
        elems.m_t(ie, tl.np1, igp, jgp, k) += dt * elems.m_ft(ie, igp, jgp, k);
      });
  Kokkos::parallel_for(
      "state velocity forcing",
      Kokkos::RangePolicy<ExecSpace>(0, num_e * 2 * NP * NP * NUM_LEV),
      KOKKOS_LAMBDA(const int & idx) {
        const int ie = (((idx / NUM_LEV) / NP) / NP) / 2;
        const int dim = (((idx / NUM_LEV) / NP) / NP) % 2;
        const int igp = ((idx / NUM_LEV) / NP) % NP;
        const int jgp = (idx / NUM_LEV) % NP;
        const int k = idx % NUM_LEV;
        elems.m_v(ie, tl.np1, dim, igp, jgp, k) +=
            dt * elems.m_fm(ie, dim, igp, jgp, k);
      });
}

void tracer_forcing(const Elements &elems, const Tracers &tracers,
                    const HybridVCoord &hvcoord, const TimeLevel &tl,
                    const MoistDry &moisture, const double &dt) {
  const int num_e = elems.num_elems();

  const auto policy = Homme::get_default_team_policy<ExecSpace>(num_e);
  Kokkos::parallel_for("tracer FQ_PS forcing", policy,
                       KOKKOS_LAMBDA(const TeamMember & team) {
    KernelVariables kv(team);
    const int &ie = kv.team_idx;

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int pt_idx) {
      const int igp = pt_idx / NP;
      const int jgp = pt_idx % NP;

      Dispatch<ExecSpace>::parallel_scan(
          kv.team, NUM_PHYSICAL_LEV,
          [&](const int &k, Real &accumulator, const bool &last) {
            const int &&ilev = k / VECTOR_SIZE;
            const int &&vlev = k % VECTOR_SIZE;
            double v1 = tracers.fq(ie, 0, igp, jgp, ilev)[vlev];
            const double &qdp =
                tracers.qdp(ie, tl.np1_qdp, 0, igp, jgp, ilev)[vlev];
            if (qdp + v1 * dt < 0.0 && v1 < 0.0) {
              if (qdp < 0.0) {
                v1 = 0.0;
              } else {
                v1 = -qdp / dt;
              }
            }
            accumulator += v1;
            if (last) {
              elems.m_fq_ps(ie, igp, jgp) = accumulator;
            }
          });
    });
  });

  const int num_q = tracers.num_tracers();
  Kokkos::parallel_for(
      "tracer qdp forcing",
      Kokkos::RangePolicy<ExecSpace>(0, num_e * num_q * NP * NP * NUM_LEV),
      KOKKOS_LAMBDA(const int & idx) {
        const int ie = (((idx / NUM_LEV) / NP) / NP) / num_q;
        const int iq = (((idx / NUM_LEV) / NP) / NP) % num_q;
        const int igp = ((idx / NUM_LEV) / NP) % NP;
        const int jgp = (idx / NUM_LEV) % NP;
        const int k = idx % NUM_LEV;
        Scalar &v1 = tracers.fq(ie, iq, igp, jgp, k);
        Scalar &qdp = tracers.qdp(ie, tl.np1_qdp, iq, igp, jgp, k);
        VECTOR_SIMD_LOOP
        for (int vlev = 0; vlev < VECTOR_SIZE; ++vlev) {
          if (qdp[vlev] + v1[vlev] * dt < 0.0 && v1[vlev] < 0.0) {
            if (qdp[vlev] < 0.0) {
              v1[vlev] = 0.0;
            } else {
              v1[vlev] = -qdp[vlev] / dt;
            }
          }
          qdp[vlev] += v1[vlev];
        }
      });

  if (moisture == MoistDry::MOIST) {
    // This conserves the dry mass in the process of forcing tracer 0
    Kokkos::parallel_for("tracer forcing ps_v",
                         Kokkos::RangePolicy<ExecSpace>(0, num_e * NP * NP),
                         KOKKOS_LAMBDA(const int & idx) {
      const int ie = (idx / NP) / NP;
      const int igp = (idx / NP) % NP;
      const int jgp = idx % NP;
      elems.m_ps_v(ie, tl.np1, igp, jgp) += elems.m_fq_ps(ie, igp, jgp);
    });
  }

  Kokkos::parallel_for(
      "tracer forcing ps_v",
      Kokkos::RangePolicy<ExecSpace>(0, num_e * num_q * NP * NP * NUM_LEV),
      KOKKOS_LAMBDA(const int & idx) {
        const int ie = (((idx / NUM_LEV) / NP) / NP) / num_q;
        const int iq = (((idx / NUM_LEV) / NP) / NP) % num_q;
        const int igp = ((idx / NUM_LEV) / NP) % NP;
        const int jgp = (idx / NUM_LEV) % NP;
        const int k = idx % NUM_LEV;

        const Scalar dp =
            hvcoord.hybrid_ai_delta(k) * hvcoord.ps0 +
            hvcoord.hybrid_bi_delta(k) * elems.m_ps_v(ie, tl.np1, igp, jgp);
        tracers.q(ie, iq, igp, jgp, k) =
            tracers.qdp(ie, tl.np1_qdp, iq, igp, jgp, k) / dp;
      });
}

// ---------------------------------------- //

}
