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
#include "profiling.hpp"

namespace Homme {

// ----------------- SIGNATURES ---------------- //

void tracer_forcing(const Elements &elems, const Tracers &tracers,
                    const HybridVCoord &hvcoord, const TimeLevel &tl,
                    const MoistDry &moisture, const Real &dt);
void state_forcing(
    const ExecViewUnmanaged<const Scalar * [NP][NP][NUM_LEV]> &f_t,
    const ExecViewUnmanaged<const Scalar * [2][NP][NP][NUM_LEV]> &f_m,
    const TimeLevel &tl, const Real &dt,
    const ExecViewUnmanaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> &t,
    const ExecViewUnmanaged<Scalar * [NUM_TIME_LEVELS][2][NP][NP][NUM_LEV]> &v);

// -------------- IMPLEMENTATIONS -------------- //

void apply_cam_forcing(const Real &dt) {
  GPTLstart("ApplyCAMForcing");
  const Elements &elems = Context::singleton().get_elements();
  const TimeLevel &tl = Context::singleton().get_time_level();

  state_forcing(elems.m_ft, elems.m_fm, tl, dt, elems.m_t, elems.m_v);

  const SimulationParams &sim_params =
      Context::singleton().get_simulation_params();
  const HybridVCoord &hvcoord = Context::singleton().get_hvcoord();
  const Tracers &tracers = Context::singleton().get_tracers();
  tracer_forcing(elems, tracers, hvcoord, tl, sim_params.moisture, dt);
  GPTLstop("ApplyCAMForcing");
}

void state_forcing(
    const ExecViewUnmanaged<const Scalar * [NP][NP][NUM_LEV]> &f_t,
    const ExecViewUnmanaged<const Scalar * [2][NP][NP][NUM_LEV]> &f_m,
    const TimeLevel &tl, const Real &dt,
    const ExecViewUnmanaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]> &t,
    const ExecViewUnmanaged<Scalar * [NUM_TIME_LEVELS][2][NP][NP][NUM_LEV]> &
        v) {
  const int num_e = f_t.extent_int(0);
  Kokkos::parallel_for(
      "state temperature forcing",
      Kokkos::RangePolicy<ExecSpace>(0, num_e * NP * NP * NUM_LEV),
      KOKKOS_LAMBDA(const int & idx) {
        const int ie = ((idx / NUM_LEV) / NP) / NP;
        const int igp = ((idx / NUM_LEV) / NP) % NP;
        const int jgp = (idx / NUM_LEV) % NP;
        const int k = idx % NUM_LEV;
        t(ie, tl.np1, igp, jgp, k) += dt * f_t(ie, igp, jgp, k);
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
        v(ie, tl.np1, dim, igp, jgp, k) += dt * f_m(ie, dim, igp, jgp, k);
      });
}

void tracer_forcing(const Elements &elems, const Tracers &tracers,
                    const HybridVCoord &hvcoord, const TimeLevel &tl,
                    const MoistDry &moisture, const double &dt) {
  const int num_e = elems.num_elems();

  const auto policy = Homme::get_default_team_policy<ExecSpace>(num_e);

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
    // Remove the m_fq_ps_v buffer since it's not actually needed.
    // Instead apply the forcing to m_ps_v directly
    // Bonus - one less parallel reduce in dry cases!

    // This conserves the dry mass in the process of forcing tracer 0
    Kokkos::parallel_for("tracer FQ_PS forcing", policy,
                         KOKKOS_LAMBDA(const TeamMember & team) {
      KernelVariables kv(team);
      const int &ie = kv.team_idx;

      Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                           [&](const int pt_idx) {
        const int igp = pt_idx / NP;
        const int jgp = pt_idx % NP;

        Real ps_v_forcing = 0.0;

        Dispatch<ExecSpace>::parallel_reduce(
            kv.team, Kokkos::ThreadVectorRange(kv.team, NUM_PHYSICAL_LEV),
            [&](const int &k, Real &accumulator) {
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
            }, ps_v_forcing);
        elems.m_ps_v(ie, tl.np1, igp, jgp) += ps_v_forcing;
      });
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
        tracers.Q(ie, iq, igp, jgp, k) =
            tracers.qdp(ie, tl.np1_qdp, iq, igp, jgp, k) / dp;
      });
}

// ---------------------------------------- //
}
