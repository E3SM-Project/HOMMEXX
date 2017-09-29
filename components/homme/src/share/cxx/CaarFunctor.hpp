#ifndef CAAR_FUNCTOR_HPP
#define CAAR_FUNCTOR_HPP

#include "Types.hpp"
#include "Control.hpp"
#include "Elements.hpp"
#include "Derivative.hpp"
#include "KernelVariables.hpp"
#include "SphereOperators.hpp"

#include "Utility.hpp"

namespace Homme {

struct CaarFunctor {
  const Control     m_data;
  const Elements&   m_elements;
  const Derivative& m_deriv;

  static constexpr Kokkos::Impl::ALL_t ALL = Kokkos::ALL;

  CaarFunctor() : m_data(), m_elements(get_elements()), m_deriv(get_derivative()) {
    // Nothing to be done here
  }

  KOKKOS_INLINE_FUNCTION
  CaarFunctor(const Control &data)
      : m_data(data), m_elements(get_elements()), m_deriv(get_derivative()) {
    // Nothing to be done here
  }

  // Depends on PHI (after preq_hydrostatic), PECND
  // Modifies Ephi_grad
  // Computes \nabla (E + phi) + \nabla (P) * Rgas * T_v / P
  KOKKOS_FORCEINLINE_FUNCTION void compute_energy_grad(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      // Kinetic energy + PHI (geopotential energy) +
      // PECND (potential energy?)
      Scalar k_energy =
          0.5 * (m_elements.m_u(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
                     m_elements.m_u(kv.ie, m_data.n0, kv.ilev, igp, jgp) +
                 m_elements.m_v(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
                     m_elements.m_v(kv.ie, m_data.n0, kv.ilev, igp, jgp));
      m_elements.buffers.ephi(kv.ie, kv.ilev, igp, jgp) =
          k_energy + (m_elements.m_phi(kv.ie, kv.ilev, igp, jgp) +
                      m_elements.m_pecnd(kv.ie, kv.ilev, igp, jgp));
    });

    //amb _nsv == "no subview". Subviews are not free. They are useful
    // when the cost of constructing one is amortized over a
    // sufficiently large number of uses of the resulting view. I
    // think there's a very new feature in Kokkos -- I haven't tested
    // it yet -- to use the same subview metadata with multiple block
    // data views that might make it feasible to use a subview just
    // once. See Kokkos issue #648 for more. In any case, here I
    // create _nsv variants of these functions to remove the subview
    // cost.
    gradient_sphere_update_nsv(
        kv, m_elements.m_dinv, m_deriv.get_dvv(),
        m_elements.buffers.ephi,
        m_elements.buffers.energy_grad);
  }

  // Depends on pressure, PHI, U_current, V_current, METDET,
  // D, DINV, U, V, FCOR, SPHEREMP, T_v, ETA_DPDN
  KOKKOS_INLINE_FUNCTION void compute_phase_3(KernelVariables &kv) const {
    using Kokkos::ALL;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
                         [&](const int &ilev) {
      kv.ilev = ilev;
      compute_eta_dpdn(kv);
      compute_omega_p(kv);
      compute_temperature_np1(kv);
      compute_velocity_np1(kv);
      compute_dp3d_np1(kv);
    });
  }

  // Depends on pressure, PHI, U_current, V_current, METDET,
  // D, DINV, U, V, FCOR, SPHEREMP, T_v
  KOKKOS_INLINE_FUNCTION
  void compute_velocity_np1(KernelVariables &kv) const {
    gradient_sphere_nsv(
        kv, m_elements.m_dinv, m_deriv.get_dvv(),
        m_elements.buffers.pressure,
        m_elements.buffers.pressure_grad);
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, 2 * NP * NP),
                         [&](const int idx) {
      const int hgp = (idx / NP) / NP;
      const int igp = (idx / NP) % NP;
      const int jgp = idx % NP;

      m_elements.buffers.energy_grad(kv.ie, kv.ilev, hgp, igp, jgp) =
          PhysicalConstants::Rgas *
          (m_elements.buffers.temperature_virt(kv.ie, kv.ilev, igp, jgp) /
           m_elements.buffers.pressure(kv.ie, kv.ilev, igp, jgp)) *
          m_elements.buffers.pressure_grad(kv.ie, kv.ilev, hgp, igp, jgp);
    });

    compute_energy_grad(kv);

    vorticity_sphere_nsv(
        kv, m_elements.m_d, m_elements.m_metdet, m_deriv.get_dvv(),
        m_elements.m_u,
        m_elements.m_v,
        m_elements.buffers.vorticity, m_data.n0);

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      //amb Use refs one possible to avoid indexing repeatedly.
      auto& energy_grad0 = m_elements.buffers.energy_grad(kv.ie, kv.ilev, 0, igp, jgp);
      auto& energy_grad1 = m_elements.buffers.energy_grad(kv.ie, kv.ilev, 1, igp, jgp);
      const auto& spheremp = m_elements.m_spheremp(kv.ie, igp, jgp);

      // Recycle vort to contain (fcor+vort)
      m_elements.buffers.vorticity(kv.ie, kv.ilev, igp, jgp) +=
          m_elements.m_fcor(kv.ie, igp, jgp);

      energy_grad0 *= -1;
      energy_grad0 +=
          /* v_vadv(igp, jgp) + */ m_elements.m_v(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
          m_elements.buffers.vorticity(kv.ie, kv.ilev, igp, jgp);
      energy_grad1 *= -1;
      energy_grad1 +=
          /* v_vadv(igp, jgp) + */ -m_elements.m_u(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
          m_elements.buffers.vorticity(kv.ie, kv.ilev, igp, jgp);

      energy_grad0 *= m_data.dt;
      energy_grad0 += m_elements.m_u(kv.ie, m_data.nm1, kv.ilev, igp, jgp);
      energy_grad1 *= m_data.dt;
      energy_grad1 += m_elements.m_v(kv.ie, m_data.nm1, kv.ilev, igp, jgp);

      // Velocity at np1 = spheremp * buffer
      m_elements.m_u(kv.ie, m_data.np1, kv.ilev, igp, jgp) = spheremp * energy_grad0;
      m_elements.m_v(kv.ie, m_data.np1, kv.ilev, igp, jgp) = spheremp * energy_grad1;
    });
  }

  // Depends on ETA_DPDN
  KOKKOS_INLINE_FUNCTION
  void compute_eta_dpdn(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         KOKKOS_LAMBDA(const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      // TODO: Compute the actual value for this if
      // rsplit=0.
      // m_elements.ETA_DPDN += eta_ave_w*eta_dot_dpdn

      m_elements.m_eta_dot_dpdn(kv.ie, kv.ilev, igp, jgp) = 0;
    });
  }

  static void threadstuff (KernelVariables &kv, int& lib, int& lie) {
    // Could abstract this to a thread-specific KernelVariables-like
    // object. None of this should be in this function
    const int tid = kv.team_rank();
    const int nth = kv.team_size();
    // Works only for powers of 2. If hardcoded to 8 (2 threads), I
    // get Fortran performance. That's because the SIMD loop before is
    // known at compile time to be perfect. But we care about thread
    // scalability more than perfect auto-vectorization in the (i,j)
    // dimension.
    const int wsz = (NP * NP) / nth;
    lib = wsz*tid;
    lie = lib + wsz;
  }

  // Depends on PHIS, DP3D, PHI, pressure, T_v
  // Modifies PHI
  KOKKOS_INLINE_FUNCTION
  void preq_hydrostatic(KernelVariables &kv) const {
    //amb Get loop ranges for our hand-rolled ThreadVectorRange.
    int lib, lie;
    threadstuff(kv, lib, lie);

    // Thread-specific stack-alloc'ed space, with some going to
    // waste. That's OK, but we could abstract it for GPU vs KNL.
    Real phii1[NP * NP] = {0};

#if 1
    //amb Invert level pack and (i,j) loop for locality.
    for (int ilevel = NUM_LEV - 1; ilevel >= 0; --ilevel) {
      kv.ilev = ilevel;

      //amb pragma simd is needed to make the compiler vectorize the
      // loop. Kokkos::ThreadVectorRange doesn't seem to be enough.
      // Could abstract this to Homme::team_parallel_for coupled with
      // the above.
#     pragma simd
      for (int li = lib; li < lie; ++li) {
        const int i = li / NP;
        const int j = li % NP;

        //amb Use references to vector packs.
        const auto& phis = m_elements.m_phis(kv.ie, i, j);
        const auto& dp3d = m_elements.m_dp3d(kv.ie, m_data.n0, kv.ilev, i, j);
        const auto& pressure = m_elements.buffers.pressure(kv.ie, kv.ilev, i, j);
        auto& phi = m_elements.m_phi(kv.ie, kv.ilev, i, j);
        auto& temperature_virt = m_elements.buffers.temperature_virt(kv.ie, kv.ilev, i, j);

        auto& phii = phii1[li];

        //amb The indexing above amortizes over the SIMD pack
        // calculation in what follows, which has as little indexing
        // as possible: just the v index.
        for (int v = VECTOR_SIZE - 1; v >= 0; --v) {
          const auto R_Tv_hkl = PhysicalConstants::Rgas * temperature_virt[v] * (dp3d[v] / pressure[v]);
          // compute phi
          phi[v] = phis + phii + 0.5 * R_Tv_hkl;
          // update phii
          phii += R_Tv_hkl;
        }

      }
    }
#else
    // Invert level pack and (i,j) loop for locality.
    for (int ilevel = NUM_LEV - 1; ilevel >= 0; --ilevel) {
      kv.ilev = ilevel;

      // 2. Could abstract this to ij-slice class:
      //     const ConstIJSlice phis1 = constslice(m_elements.m_phis);
      const auto* phis1 = &m_elements.m_phis(kv.ie, 0, 0);
      const auto* dp3d1 = &m_elements.m_dp3d(kv.ie, m_data.n0, kv.ilev, 0, 0);
      const auto* pressure1 = &m_elements.buffers.pressure(kv.ie, kv.ilev, 0, 0);
      auto* phi1 = &m_elements.m_phi(kv.ie, kv.ilev, 0, 0);
      auto* temperature_virt1 = &m_elements.buffers.temperature_virt(kv.ie, kv.ilev, 0, 0);

      // 1. Could abstract this to Homme::team_parallel_for coupled
      // with the above.
#     pragma simd
      for (int li = lib; li < lie; ++li) {

        const auto& phis = phis1[li];
        const auto& dp3d = dp3d1[li];
        const auto& pressure = pressure1[li];
        auto& phi = phi1[li];
        auto& temperature_virt = temperature_virt1[li];
        auto& phii = phii1[li];

        // The indexing above amortizes over the SIMD pack
        // calculation, which has as little indexing as possible: just
        // the v index.
        for (int v = VECTOR_SIZE - 1; v >= 0; --v) {
          const auto R_Tv_hkl = PhysicalConstants::Rgas * temperature_virt[v] * (dp3d[v] / pressure[v]);
          // compute phi
          phi[v] = phis + phii + 0.5 * R_Tv_hkl;
          // update phii
          phii += R_Tv_hkl;
        }

      }
    }
#endif
  }

  // Depends on pressure, U_current, V_current, div_vdp,
  // omega_p
  KOKKOS_INLINE_FUNCTION
  void preq_omega_ps(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
                         [&](const int ilev) {
      kv.ilev = ilev;
      gradient_sphere_nsv(
          kv, m_elements.m_dinv, m_deriv.get_dvv(),
          m_elements.buffers.pressure,
          m_elements.buffers.pressure_grad);

      const auto f = [&](const int li) {                           
        const int igp = li / NP;
        const int jgp = li % NP;

        const Scalar vgrad_p = ( (m_elements.m_u(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
                                  m_elements.buffers.pressure_grad(kv.ie, kv.ilev, 0, igp, jgp)) +
                                 (m_elements.m_v(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
                                  m_elements.buffers.pressure_grad(kv.ie, kv.ilev, 1, igp, jgp)) );

        m_elements.buffers.omega_p(kv.ie, kv.ilev, igp, jgp) =
        vgrad_p / m_elements.buffers.pressure(kv.ie, kv.ilev, igp, jgp);
      };
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP), f);
    });

    kv.team_barrier();

    int lib, lie;
    threadstuff(kv, lib, lie);

    Real integration[NP * NP] = {0};

    for (int ilevel = 0; ilevel < NUM_LEV; ++ilevel) {
      kv.ilev = ilevel;

#     pragma simd
      for (int li = lib; li < lie; ++li) {
        const int igp = li / NP;
        const int jgp = li % NP;
        const auto& div_vdp = m_elements.buffers.div_vdp(kv.ie, kv.ilev, igp, jgp);
        const auto& pressure = m_elements.buffers.pressure(kv.ie, kv.ilev, igp, jgp);
        auto& omega_p = m_elements.buffers.omega_p(kv.ie, kv.ilev, igp, jgp);
        auto& ili = integration[li];

        for (int vec = 0; vec < VECTOR_SIZE; ++vec) {
          omega_p[vec] -= (ili + 0.5 * div_vdp[vec]) / pressure[vec];
          ili += div_vdp[vec];
        }
      }
    }
  }

  // Depends on DP3D
  KOKKOS_INLINE_FUNCTION
  void compute_pressure(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int idx) {
      Kokkos::single(Kokkos::PerThread(kv.team), [&]() {
        const int igp = idx / NP;
        const int jgp = idx % NP;
        m_elements.buffers.pressure(kv.ie, 0, igp, jgp)[0] =
            m_data.hybrid_a0 * m_data.ps0 +
            0.5 * m_elements.m_dp3d(kv.ie, m_data.n0, 0, igp, jgp)[0];

        // TODO: change the sum into p(k) = p(k-1) + 0.5*(
        // dp(k)+dp(k-1) ) to
        // increase accuracy
        for (kv.ilev = 1; kv.ilev < NUM_PHYSICAL_LEV; ++kv.ilev) {
          const int lev = kv.ilev / VECTOR_SIZE;
          const int vec = kv.ilev % VECTOR_SIZE;

          const int lev_prev = (kv.ilev - 1) / VECTOR_SIZE;
          const int vec_prev = (kv.ilev - 1) % VECTOR_SIZE;
          m_elements.buffers.pressure(kv.ie, lev, igp, jgp)[vec] =
              m_elements.buffers.pressure(kv.ie, lev_prev, igp, jgp)[vec_prev] +
              0.5 * m_elements.m_dp3d(kv.ie, m_data.n0, lev_prev, igp, jgp)[vec_prev] +
              0.5 * m_elements.m_dp3d(kv.ie, m_data.n0, lev, igp, jgp)[vec];
        }
      });
    });
  }

  // Depends on DP3D, PHIS, DP3D, PHI, T_v
  // Modifies pressure, PHI
  KOKKOS_INLINE_FUNCTION
  void compute_scan_properties(KernelVariables &kv) const {
    compute_pressure(kv);
    kv.team_barrier();
    preq_hydrostatic(kv);
    kv.team_barrier();
    preq_omega_ps(kv);
  }

  KOKKOS_INLINE_FUNCTION
  void compute_temperature_no_tracers_helper(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      m_elements.buffers.temperature_virt(kv.ie, kv.ilev, igp, jgp) =
          m_elements.m_t(kv.ie, m_data.n0, kv.ilev, igp, jgp);
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_temperature_tracers_helper(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      Scalar Qt = m_elements.m_qdp(kv.ie, m_data.qn0, 0, kv.ilev, igp, jgp) /
                  m_elements.m_dp3d(kv.ie, m_data.n0, kv.ilev, igp, jgp);
      Qt *= (PhysicalConstants::Rwater_vapor / PhysicalConstants::Rgas - 1.0);
      Qt += 1.0;
      m_elements.buffers.temperature_virt(kv.ie, kv.ilev, igp, jgp) =
          m_elements.m_t(kv.ie, m_data.n0, kv.ilev, igp, jgp) * Qt;
    });
  }

  // Depends on DERIVED_UN0, DERIVED_VN0, METDET, DINV
  // Initializes div_vdp, which is used 2 times afterwards
  // Modifies DERIVED_UN0, DERIVED_VN0
  // Requires NUM_LEV * 5 * NP * NP
  KOKKOS_INLINE_FUNCTION
  void compute_div_vdp(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      m_elements.buffers.vdp(kv.ie, kv.ilev, 0, igp, jgp) =
          m_elements.m_u(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
          m_elements.m_dp3d(kv.ie, m_data.n0, kv.ilev, igp, jgp);

      m_elements.buffers.vdp(kv.ie, kv.ilev, 1, igp, jgp) =
          m_elements.m_v(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
          m_elements.m_dp3d(kv.ie, m_data.n0, kv.ilev, igp, jgp);

      m_elements.m_derived_un0(kv.ie, kv.ilev, igp, jgp) =
          m_elements.m_derived_un0(kv.ie, kv.ilev, igp, jgp) +
          m_data.eta_ave_w * m_elements.buffers.vdp(kv.ie, kv.ilev, 0, igp, jgp);

      m_elements.m_derived_vn0(kv.ie, kv.ilev, igp, jgp) =
          m_elements.m_derived_vn0(kv.ie, kv.ilev, igp, jgp) +
          m_data.eta_ave_w * m_elements.buffers.vdp(kv.ie, kv.ilev, 1, igp, jgp);
    });

    divergence_sphere_nsv(
        kv, m_elements.m_dinv, m_elements.m_metdet, m_deriv.get_dvv(),
        m_elements.buffers.vdp,
        m_elements.buffers.div_vdp);
  }

  // Depends on T_current, DERIVE_UN0, DERIVED_VN0, METDET,
  // DINV
  // Might depend on QDP, DP3D_current
  KOKKOS_INLINE_FUNCTION
  void compute_temperature_div_vdp(KernelVariables &kv) const {
    if (m_data.qn0 == -1) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
                           [&](const int ilev) {
        kv.ilev = ilev;
        compute_temperature_no_tracers_helper(kv);
        compute_div_vdp(kv);
      });
    } else {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NUM_LEV),
                           [&](const int ilev) {
        kv.ilev = ilev;
        compute_temperature_tracers_helper(kv);
        compute_div_vdp(kv);
      });
    }
  }

  KOKKOS_INLINE_FUNCTION
  void compute_omega_p(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      m_elements.m_omega_p(kv.ie, kv.ilev, igp, jgp) +=
          m_data.eta_ave_w * m_elements.buffers.omega_p(kv.ie, kv.ilev, igp, jgp);
    });
  }

  // Depends on T (global), OMEGA_P (global), U (global), V
  // (global),
  // SPHEREMP (global), T_v, and omega_p
  // block_3d_scalars
  KOKKOS_INLINE_FUNCTION
  void compute_temperature_np1(KernelVariables &kv) const {
    gradient_sphere_nsv(
        kv, m_elements.m_dinv, m_deriv.get_dvv(),
        m_elements.m_t,
        m_elements.buffers.temperature_grad, m_data.n0);

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      Scalar vgrad_t =
          m_elements.m_u(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
              m_elements.buffers.temperature_grad(kv.ie, kv.ilev, 0, igp, jgp) +
          m_elements.m_v(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
              m_elements.buffers.temperature_grad(kv.ie, kv.ilev, 1, igp, jgp);

      // vgrad_t + kappa * T_v * omega_p
      Scalar ttens;
      ttens = -vgrad_t +
              PhysicalConstants::kappa *
                  m_elements.buffers.temperature_virt(kv.ie, kv.ilev, igp, jgp) *
                  m_elements.buffers.omega_p(kv.ie, kv.ilev, igp, jgp);

      Scalar temp_np1 = ttens * m_data.dt +
                        m_elements.m_t(kv.ie, m_data.nm1, kv.ilev, igp, jgp);
      temp_np1 *= m_elements.m_spheremp(kv.ie, igp, jgp);
      m_elements.m_t(kv.ie, m_data.np1, kv.ilev, igp, jgp) = temp_np1;
    });
  }

  // Depends on DERIVED_UN0, DERIVED_VN0, U, V,
  // Modifies DERIVED_UN0, DERIVED_VN0, OMEGA_P, T, and DP3D
  KOKKOS_INLINE_FUNCTION
  void compute_dp3d_np1(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      Scalar tmp = m_elements.m_dp3d(kv.ie, m_data.nm1, kv.ilev, igp, jgp);
      tmp -= m_data.dt * m_elements.buffers.div_vdp(kv.ie, kv.ilev, igp, jgp);
      m_elements.m_dp3d(kv.ie, m_data.np1, kv.ilev, igp, jgp) =
          m_elements.m_spheremp(kv.ie, igp, jgp) * tmp;
    });
  }

  // Computes the vertical advection of T and v
  // Not currently used
  KOKKOS_INLINE_FUNCTION
  void preq_vertadv(
      const TeamMember &,
      const ExecViewUnmanaged<const Scalar[NUM_LEV][NP][NP]> T,
      const ExecViewUnmanaged<const Scalar[NUM_LEV][2][NP][NP]> v,
      const ExecViewUnmanaged<const Scalar[NUM_LEV_P][NP][NP]> eta_dp_deta,
      const ExecViewUnmanaged<const Scalar[NUM_LEV][NP][NP]> rpdel,
      ExecViewUnmanaged<Scalar[NUM_LEV][NP][NP]> T_vadv,
      ExecViewUnmanaged<Scalar[NUM_LEV][2][NP][NP]> v_vadv) {
    constexpr const int k_0 = 0;
    for (int j = 0; j < NP; ++j) {
      for (int i = 0; i < NP; ++i) {
        Scalar facp = 0.5 * rpdel(k_0, j, i) * eta_dp_deta(k_0 + 1, j, i);
        T_vadv(k_0, j, i) = facp * (T(k_0 + 1, j, i) - T(k_0, j, i));
        for (int h = 0; h < 2; ++h) {
          v_vadv(k_0, h, j, i) = facp * (v(k_0 + 1, h, j, i) - v(k_0, h, j, i));
        }
      }
    }
    constexpr const int k_f = NUM_LEV - 1;
    for (int k = k_0 + 1; k < k_f; ++k) {
      for (int j = 0; j < NP; ++j) {
        for (int i = 0; i < NP; ++i) {
          Scalar facp = 0.5 * rpdel(k, j, i) * eta_dp_deta(k + 1, j, i);
          Scalar facm = 0.5 * rpdel(k, j, i) * eta_dp_deta(k, j, i);
          T_vadv(k, j, i) = facp * (T(k + 1, j, i) - T(k, j, i)) +
                            facm * (T(k, j, i) - T(k - 1, j, i));
          for (int h = 0; h < 2; ++h) {
            v_vadv(k, h, j, i) = facp * (v(k + 1, h, j, i) - v(k, h, j, i)) +
                                 facm * (v(k, h, j, i) - v(k - 1, h, j, i));
          }
        }
      }
    }
    for (int j = 0; j < NP; ++j) {
      for (int i = 0; i < NP; ++i) {
        Scalar facm = 0.5 * rpdel(k_f, j, i) * eta_dp_deta(k_f, j, i);
        T_vadv(k_f, j, i) = facm * (T(k_f, j, i) - T(k_f - 1, j, i));
        for (int h = 0; h < 2; ++h) {
          v_vadv(k_f, h, j, i) = facm * (v(k_f, h, j, i) - v(k_f - 1, h, j, i));
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(TeamMember team) const {
    KernelVariables kv(team);

    compute_temperature_div_vdp(kv);
    kv.team_barrier();
    compute_scan_properties(kv);
    kv.team_barrier();
    compute_phase_3(kv);
  }

  KOKKOS_INLINE_FUNCTION
  size_t shmem_size(const int team_size) const {
    //amb No shared memory needed.
    return 0; //KernelVariables::shmem_size(team_size);
  }
};

} // Namespace Homme

#endif // CAAR_FUNCTOR_HPP
