#ifndef HOMMEXX_EULER_STEP_FUNCTOR_HPP
#define HOMMEXX_EULER_STEP_FUNCTOR_HPP

#include "Context.hpp"
#include "Elements.hpp"
#include "Derivative.hpp"
#include "Control.hpp"
#include "SphereOperators.hpp"

namespace Homme {

class EulerStepFunctor {
  Control           m_data;
  const Elements    m_elements;
  const Derivative  m_deriv;

  enum { m_mem_per_team = 2 * NP * NP * sizeof(Real) };

public:

  static size_t team_shmem_size (const int team_size) {
    return Memory<ExecSpace>::on_gpu ? team_size * m_mem_per_team : 0;
  }

  EulerStepFunctor (const Control& data)
   : m_data    (data)
   , m_elements(Context::singleton().get_elements())
   , m_deriv   (Context::singleton().get_derivative())
  {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const TeamMember& team) const {
    start_timer("esf compute");
    KernelVariables kv(team, m_data.qsize);
    compute_vstar_qdp(kv);
    compute_qtens(kv);
    kv.team_barrier();
    if (m_data.rhs_viss != 0) {
      add_hyperviscosity(kv);
      kv.team_barrier();
    }
    if (m_data.limiter_option == 8) {
      limiter_optim_iter_full(kv);
      kv.team_barrier();
    }
    apply_mass_matrix(kv);
    stop_timer("esf compute");
  }

  static void run () {
    Control& data = Context::singleton().get_control();
    EulerStepFunctor func(data);

    profiling_resume();
    start_timer("esf run");
    Kokkos::parallel_for(get_policy(data), func);
    stop_timer("esf run");

    ExecSpace::fence();
    profiling_pause();
  }

private:

  static Kokkos::TeamPolicy<ExecSpace, void> get_policy(const Control& data) {
    return Homme::get_default_team_policy<ExecSpace>(data.num_elems * data.qsize);
  }

  KOKKOS_INLINE_FUNCTION
  void compute_vstar_qdp (const KernelVariables& kv) const {
    const auto NP2 = NP * NP;
    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(kv.team, NP2),
      [&] (const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;

        const ExecViewUnmanaged<const Scalar[NP][NP][NUM_LEV]>
          qdp   = Homme::subview(m_elements.m_qdp, kv.ie, m_data.qn0, kv.iq);
        const ExecViewUnmanaged<Scalar[NP][NP][NUM_LEV]>
          q_buf = Homme::subview(m_elements.buffers.qtens, kv.ie, kv.iq);
        const ExecViewUnmanaged<Scalar[2][NP][NP][NUM_LEV]>
          v_buf = Homme::subview(m_elements.buffers.vstar_qdp, kv.ie, kv.iq);

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV), [&] (const int& ilev) {
          v_buf(0,igp,jgp,ilev) = (m_elements.buffers.vstar(kv.ie, 0, igp, jgp, ilev) *
                                   qdp(igp, jgp, ilev));
          v_buf(1,igp,jgp,ilev) = (m_elements.buffers.vstar(kv.ie, 1, igp, jgp, ilev) *
                                   qdp(igp, jgp, ilev));
          q_buf(igp,jgp,ilev) = qdp(igp,jgp,ilev);
        });
      }
    );
  }

  KOKKOS_INLINE_FUNCTION
  void compute_qtens (const KernelVariables& kv) const {
    const auto dvv = m_deriv.get_dvv();
    const ExecViewUnmanaged<const Real[NP][NP]>
      metdet = Homme::subview(m_elements.m_metdet, kv.ie);
    const ExecViewUnmanaged<const Real[2][2][NP][NP]>
      dinv = Homme::subview(m_elements.m_dinv, kv.ie);
    divergence_sphere_update(
      kv, -m_data.dt, 1.0, dinv, metdet, dvv,
      Homme::subview(m_elements.buffers.vstar_qdp, kv.ie, kv.iq),
      Homme::subview(m_elements.buffers.qwrk, kv.ie, kv.iq),
      Homme::subview(m_elements.buffers.qtens, kv.ie, kv.iq));
  }

  KOKKOS_INLINE_FUNCTION
  void add_hyperviscosity (const KernelVariables& kv) const {
    const auto qtens = Homme::subview(m_elements.buffers.qtens, kv.ie, kv.iq);
    const auto qtens_biharmonic = Homme::subview(m_elements.buffers.qtens_biharmonic, kv.ie, kv.iq);
    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(kv.team, NP * NP),
      [&] (const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;
        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
          [&] (const int& ilev) {
            qtens(igp, jgp, ilev) += qtens_biharmonic(igp, jgp, ilev);
          });
      });
  }

  KOKKOS_INLINE_FUNCTION
  void limiter_optim_iter_full (const KernelVariables& kv) const {
    const auto sphweights = Homme::subview(m_elements.m_spheremp, kv.ie);
    const auto dpmass = Homme::subview(m_elements.buffers.dpdissk, kv.ie);
    const auto ptens = Homme::subview(m_elements.buffers.qtens, kv.ie, kv.iq);
    const auto qlim = Homme::subview(m_elements.buffers.qlim, kv.ie, kv.iq);

    limiter_optim_iter_full(kv.team, sphweights, dpmass, qlim, ptens);
  }

  //! apply mass matrix, overwrite np1 with solution:
  //! dont do this earlier, since we allow np1_qdp == n0_qdp
  //! and we dont want to overwrite n0_qdp until we are done using it
  KOKKOS_INLINE_FUNCTION
  void apply_mass_matrix (const KernelVariables& kv) const {
    const auto qdp = Homme::subview(m_elements.m_qdp, kv.ie, m_data.np1_qdp, kv.iq);
    const auto qtens = Homme::subview(m_elements.buffers.qtens, kv.ie, kv.iq);
    const auto spheremp = Homme::subview(m_elements.m_spheremp, kv.ie);
    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(kv.team, NP * NP),
      [&] (const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;
        Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
          [&] (const int& ilev) {
            qdp(igp, jgp, ilev) = spheremp(igp, jgp) * qtens(igp, jgp, ilev);
          });
      });
  }

  // Do all the setup and teardown associated with a limiter. Call a limiter
  // functor to do the actual math given the problem data (mass, minp, maxp, c,
  // x), where the limiter possibly alters x to place it in the constraint set
  //    {x: (i) minp <= x_k <= maxp and (ii) c'x = mass }.
  template <typename Limit, typename ArrayGll, typename ArrayGllLvl, typename Array2Lvl>
  KOKKOS_INLINE_FUNCTION static void
  with_limiter_shell (const TeamMember& team, const Limit& limit,
                      const ArrayGll& sphweights, const ArrayGllLvl& dpmass,
                      const Array2Lvl& qlim, const ArrayGllLvl& ptens) {
    const int NP2 = NP * NP;

    // Size doesn't matter; just need to get a pointer to the start of the
    // shared memory.
    Real* const team_data = Memory<ExecSpace>::get_shmem<Real>(team);

    Kokkos::parallel_for (
      Kokkos::TeamThreadRange(team, NUM_PHYSICAL_LEV),
      [&] (const int ilev) {
        const int vpi = ilev / VECTOR_SIZE, vsi = ilev % VECTOR_SIZE;

        const auto tvr = Kokkos::ThreadVectorRange(team, NP2);
        using Kokkos::parallel_for;
        using Kokkos::parallel_reduce;

        Real* const data = team_data ?
          team_data + 2 * NP2 * team.team_rank() :
          nullptr;
        Memory<ExecSpace>::AutoArray<Real, NP2> x(data), c(data + NP2);

        parallel_for(tvr, [&] (const int& k) {
#ifdef REBASELINE
            const int i = k / NP, j = k % NP;
#else
            const int i = k % NP, j = k / NP;
#endif
            const auto& dpm = dpmass(i,j,vpi)[vsi];
            c[k] = sphweights(i,j)*dpm;
            x[k] = ptens(i,j,vpi)[vsi]/dpm;
          });

        Real sumc = 0;
        parallel_reduce(tvr, [&] (const int& k, Real& isumc) { isumc += c[k]; }, sumc);
        if (sumc <= 0) return; //! this should never happen, but if it does, dont limit
        Real mass = 0;
        parallel_reduce(tvr, [&] (const int& k, Real& imass) { imass += x[k]*c[k]; }, mass);

        Real minp = qlim(0,vpi)[vsi], maxp = qlim(1,vpi)[vsi];
        //! relax constraints to ensure limiter has a solution:
        //! This is only needed if running with the SSP CFL>1 or
        //! due to roundoff errors
        // This is technically a write race condition, but the same value is
        // being written, so it doesn't matter.
        if (mass < minp*sumc)
          minp = qlim(0,vpi)[vsi] = mass/sumc;
        if (mass > maxp*sumc)
          maxp = qlim(1,vpi)[vsi] = mass/sumc;
    
        limit(team, mass, minp, maxp, x.data(), c.data());

        parallel_for(tvr, [&] (const int& k) {
#ifdef REBASELINE
            const int i = k / NP, j = k % NP;
#else
            const int i = k % NP, j = k / NP;
#endif
            ptens(i,j,vpi)[vsi] = x[k]*dpmass(i,j,vpi)[vsi];
          });        
      });
  }

public:

  // limiter_option = 8.
  template <typename ArrayGll, typename ArrayGllLvl, typename Array2Lvl>
  KOKKOS_INLINE_FUNCTION static void
  limiter_optim_iter_full (const TeamMember& team,
                           const ArrayGll& sphweights, const ArrayGllLvl& dpmass,
                           const Array2Lvl& qlim, const ArrayGllLvl& ptens) {
    struct Limit {
      KOKKOS_INLINE_FUNCTION void
      operator() (const TeamMember& team, const Real& mass,
                  const Real& minp, const Real& maxp,
                  Real* const x, Real const* const c) const {
        const int NP2 = NP * NP;
        const int maxiter = NP*NP - 1;
        const Real tol_limiter = 5e-14;

        const auto tvr = Kokkos::ThreadVectorRange(team, NP2);
        using Kokkos::parallel_for;
        using Kokkos::parallel_reduce;

        for (int iter = 0; iter < maxiter; ++iter) {
          Real addmass = 0;
          parallel_reduce(tvr, [&] (const int& k, Real& iaddmass) {
              if (x[k] > maxp) {
                iaddmass += (x[k] - maxp)*c[k];
                x[k] = maxp;
              } else if (x[k] < minp) {
                iaddmass += (x[k] - minp)*c[k];
                x[k] = minp;
              }
            }, addmass);

          if (std::abs(addmass) <= tol_limiter*std::abs(mass))
            break;

          Real weightssum = 0;
          if (addmass > 0) {
            parallel_reduce(tvr, [&] (const int& k, Real& iweightssum) {
                if (x[k] < maxp)
                  iweightssum += c[k];
              }, weightssum);
            const auto adw = addmass/weightssum;
            parallel_for(tvr, [&] (const int& k) {
                if (x[k] < maxp)
                  x[k] += adw;
              });
          } else {
            parallel_reduce(tvr, [&] (const int& k, Real& iweightssum) {
                if (x[k] > minp)
                  iweightssum += c[k];
              }, weightssum);
            const auto adw = addmass/weightssum;
            parallel_for(tvr, [&] (const int& k) {
                if (x[k] > minp)
                  x[k] += adw;
              });
          }
        }
      }
    };

    with_limiter_shell(team, Limit(), sphweights, dpmass, qlim, ptens);
  }

  // This is limiter_option = 9 in ACME master. For now, just unit test it.
  template <typename ArrayGll, typename ArrayGllLvl, typename Array2Lvl>
  KOKKOS_INLINE_FUNCTION static void
  limiter_clip_and_sum (const TeamMember& team,
                        const ArrayGll& sphweights, const ArrayGllLvl& dpmass,
                        const Array2Lvl& qlim, const ArrayGllLvl& ptens) {
    struct Limit {
      KOKKOS_INLINE_FUNCTION void
      operator() (const TeamMember& team, const Real& mass,
                  const Real& minp, const Real& maxp,
                  Real* const x, Real const* const c) const {
        const int NP2 = NP * NP;

        const auto tvr = Kokkos::ThreadVectorRange(team, NP2);
        using Kokkos::parallel_for;
        using Kokkos::parallel_reduce;

        // Clip.
        Real addmass = 0;
        parallel_reduce(tvr, [&] (const int& k, Real& iaddmass) {
            if (x[k] > maxp) {
              iaddmass += (x[k] - maxp)*c[k];
              x[k] = maxp;
            } else if (x[k] < minp) {
              iaddmass += (x[k] - minp)*c[k];
              x[k] = minp;
            }
          }, addmass);

        // No need for a tol: this isn't iterative. If it happens to be exactly
        // 0, then return early.
        if (addmass == 0) return;

        Real fac = 0;
        if (addmass > 0) {
          // Get sum of weights. Don't store them; we don't want another array.
          parallel_reduce(tvr, [&] (const int& k, Real& ifac) {
              ifac += c[k]*(maxp - x[k]);
            }, fac);
          if (fac > 0) {
            // Update.
            fac = addmass/fac;
            parallel_for(tvr, [&] (const int& k) { x[k] += fac*(maxp - x[k]); });
          }
        } else {
          parallel_reduce(tvr, [&] (const int& k, Real& ifac) {
              ifac += c[k]*(x[k] - minp);
            }, fac);
          if (fac > 0) {
            fac = addmass/fac;
            parallel_for(tvr, [&] (const int& k) { x[k] += fac*(x[k] - minp); });
          }
        }
      }
    };

    with_limiter_shell(team, Limit(), sphweights, dpmass, qlim, ptens);
  }
};

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_HPP
