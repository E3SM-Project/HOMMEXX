#ifndef CAAR_FUNCTOR_HPP
#define CAAR_FUNCTOR_HPP

#include "Types.hpp"
#include "Control.hpp"
#include "Region.hpp"
#include "Derivative.hpp"
#include "KernelVariables.hpp"
#include "SphereOperators.hpp"

#include "Utility.hpp"

namespace Homme {

struct CaarFunctor {
  Control           m_data;
  const Region      m_region;
  const Derivative  m_deriv;

  static constexpr Kokkos::Impl::ALL_t ALL = Kokkos::ALL;

  CaarFunctor() : m_data(), m_region(get_region()), m_deriv(get_derivative()) {
    // Nothing to be done here
  }

  KOKKOS_INLINE_FUNCTION
  CaarFunctor(const Control &data)
      : m_data(data), m_region(get_region()), m_deriv(get_derivative()) {
    // Nothing to be done here
  }

  // Depends on PHI (after preq_hydrostatic), PECND
  // Modifies Ephi_grad
  // Computes \nabla (E + phi) + \nabla (P) * Rgas * T_v / P
  KOKKOS_INLINE_FUNCTION void compute_energy_grad(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      // Kinetic energy + PHI (geopotential energy) +
      // PECND (potential energy?)
      Scalar k_energy =
          0.5 * (m_region.m_u(kv.ie, m_data.n0, igp, jgp, kv.ilev) *
                     m_region.m_u(kv.ie, m_data.n0, igp, jgp, kv.ilev) +
                 m_region.m_v(kv.ie, m_data.n0, igp, jgp, kv.ilev) *
                     m_region.m_v(kv.ie, m_data.n0, igp, jgp, kv.ilev));
      m_region.buffers.ephi(kv.ie, igp, jgp, kv.ilev) =
          k_energy + (m_region.m_phi(kv.ie, igp, jgp, kv.ilev) +
                      m_region.m_pecnd(kv.ie, igp, jgp, kv.ilev));
    });

    gradient_sphere_update(
        kv, m_region.m_dinv, m_deriv.get_dvv(),
        Homme::subview(m_region.buffers.ephi, kv.ie),
        Homme::subview(m_region.buffers.energy_grad, kv.ie));
  }

  // Depends on pressure, PHI, U_current, V_current, METDET,
  // D, DINV, U, V, FCOR, SPHEREMP, T_v, ETA_DPDN
  KOKKOS_INLINE_FUNCTION void compute_phase_3(KernelVariables &kv) const {
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
    gradient_sphere(
        kv, m_region.m_dinv, m_deriv.get_dvv(),
        Homme::subview(m_region.buffers.pressure, kv.ie),
        Homme::subview(m_region.buffers.pressure_grad, kv.ie));
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, 2 * NP * NP),
                         [&](const int idx) {
      const int hgp = (idx / NP) / NP;
      const int igp = (idx / NP) % NP;
      const int jgp = idx % NP;

      m_region.buffers.energy_grad(kv.ie, hgp, igp, jgp, kv.ilev) =
          PhysicalConstants::Rgas *
          (m_region.buffers.temperature_virt(kv.ie, igp, jgp, kv.ilev) /
           m_region.buffers.pressure(kv.ie, igp, jgp, kv.ilev)) *
          m_region.buffers.pressure_grad(kv.ie, hgp, igp, jgp, kv.ilev);
    });

    compute_energy_grad(kv);

    vorticity_sphere(
        kv, m_region.m_d, m_region.m_metdet, m_deriv.get_dvv(),
        Homme::subview(m_region.m_u, kv.ie, m_data.n0),
        Homme::subview(m_region.m_v, kv.ie, m_data.n0),
        Homme::subview(m_region.buffers.vorticity, kv.ie));

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      // Recycle vort to contain (fcor+vort)
      m_region.buffers.vorticity(kv.ie, igp, jgp, kv.ilev) +=
          m_region.m_fcor(kv.ie, igp, jgp);

      m_region.buffers.energy_grad(kv.ie, 0, igp, jgp, kv.ilev) *= -1;
      m_region.buffers.energy_grad(kv.ie, 0, igp, jgp, kv.ilev) +=
          /* v_vadv(igp, jgp) + */ m_region.m_v(kv.ie, m_data.n0, igp, jgp,
                                                kv.ilev) *
          m_region.buffers.vorticity(kv.ie, igp, jgp, kv.ilev);
      m_region.buffers.energy_grad(kv.ie, 1, igp, jgp, kv.ilev) *= -1;
      m_region.buffers.energy_grad(kv.ie, 1, igp, jgp, kv.ilev) +=
          /* v_vadv(igp, jgp) + */ -m_region.m_u(kv.ie, m_data.n0, igp, jgp,
                                                 kv.ilev) *
          m_region.buffers.vorticity(kv.ie, igp, jgp, kv.ilev);

      m_region.buffers.energy_grad(kv.ie, 0, igp, jgp, kv.ilev) *= m_data.dt;
      m_region.buffers.energy_grad(kv.ie, 0, igp, jgp, kv.ilev) +=
          m_region.m_u(kv.ie, m_data.nm1, igp, jgp, kv.ilev);
      m_region.buffers.energy_grad(kv.ie, 1, igp, jgp, kv.ilev) *= m_data.dt;
      m_region.buffers.energy_grad(kv.ie, 1, igp, jgp, kv.ilev) +=
          m_region.m_v(kv.ie, m_data.nm1, igp, jgp, kv.ilev);

      // Velocity at np1 = spheremp * buffer
      m_region.m_u(kv.ie, m_data.np1, igp, jgp, kv.ilev) =
          m_region.m_spheremp(kv.ie, igp, jgp) *
          m_region.buffers.energy_grad(kv.ie, 0, igp, jgp, kv.ilev);
      m_region.m_v(kv.ie, m_data.np1, igp, jgp, kv.ilev) =
          m_region.m_spheremp(kv.ie, igp, jgp) *
          m_region.buffers.energy_grad(kv.ie, 1, igp, jgp, kv.ilev);
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
      // m_region.ETA_DPDN += eta_ave_w*eta_dot_dpdn

      m_region.m_eta_dot_dpdn(kv.ie, jgp, igp, kv.ilev) = 0;
    });
  }

  // Depends on PHIS, DP3D, PHI, pressure, T_v
  // Modifies PHI
  KOKKOS_INLINE_FUNCTION
  void preq_hydrostatic(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Real integration = 0.0;
      for (kv.ilev = NUM_LEV - 1; kv.ilev >= 0; --kv.ilev) {
        // compute phi
        for (int v = VECTOR_SIZE - 1; v >= 0; --v) {
          m_region.m_phi(kv.ie, jgp, igp, kv.ilev)[v] =
              m_region.m_phis(kv.ie, jgp, igp) + integration +
              PhysicalConstants::Rgas * m_region.buffers.temperature_virt(
                                            kv.ie, jgp, igp, kv.ilev)[v] *
                  (m_region.m_dp3d(kv.ie, m_data.n0, jgp, igp, kv.ilev)[v] *
                   0.5 /
                   m_region.buffers.pressure(kv.ie, jgp, igp, kv.ilev)[v]);

          // update phii
          integration +=
              PhysicalConstants::Rgas *
              m_region.buffers.temperature_virt(kv.ie, jgp, igp, kv.ilev)[v] *
              2.0 *
              (m_region.m_dp3d(kv.ie, m_data.n0, jgp, igp, kv.ilev)[v] * 0.5 /
               m_region.buffers.pressure(kv.ie, jgp, igp, kv.ilev)[v]);
        }
      }
    });
  }

  // Depends on pressure, U_current, V_current, div_vdp,
  // omega_p
  KOKKOS_INLINE_FUNCTION
  void preq_omega_ps(KernelVariables &kv) const {
    // NOTE: we can't use a single TeamThreadRange loop,
    // since gradient_sphere requires a 'consistent'
    // pressure, meaning that we cannot update the different
    // pressure points within a level before the gradient is
    // complete!
    ExecViewUnmanaged<Real[NP][NP]> integration = kv.scratch_mem;
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      integration(igp, jgp) = 0.0;
    });

    for (kv.ilev = 0; kv.ilev < NUM_LEV; ++kv.ilev) {
      gradient_sphere(
          kv, m_region.m_dinv, m_deriv.get_dvv(),
          Homme::subview(m_region.buffers.pressure, kv.ie),
          Homme::subview(m_region.buffers.pressure_grad, kv.ie));

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                           [&](const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;
        Scalar vgrad_p =
            m_region.m_u(kv.ie, m_data.n0, igp, jgp, kv.ilev) *
                m_region.buffers.pressure_grad(kv.ie, 0, igp, jgp, kv.ilev) +
            m_region.m_v(kv.ie, m_data.n0, igp, jgp, kv.ilev) *
                m_region.buffers.pressure_grad(kv.ie, 1, igp, jgp, kv.ilev);

        m_region.buffers.omega_p(kv.ie, igp, jgp, kv.ilev) =
            vgrad_p / m_region.buffers.pressure(kv.ie, igp, jgp, kv.ilev);

        for (int vec = 0; vec < VECTOR_SIZE; ++vec) {
          Real div_vdp =
              m_region.buffers.div_vdp(kv.ie, igp, jgp, kv.ilev)[vec];
          Real ckk =
              0.5 / m_region.buffers.pressure(kv.ie, igp, jgp, kv.ilev)[vec];
          m_region.buffers.omega_p(kv.ie, igp, jgp, kv.ilev)[vec] -=
              (2.0 * ckk * integration(igp, jgp) + ckk * div_vdp);
          integration(igp, jgp) += div_vdp;
        }
      });
    }
  }

  // Depends on DP3D
  KOKKOS_INLINE_FUNCTION
  void compute_pressure(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      m_region.buffers.pressure(kv.ie, igp, jgp, 0)[0] =
          m_data.hybrid_a(0) * m_data.ps0 +
          0.5 * m_region.m_dp3d(kv.ie, m_data.n0, igp, jgp, 0)[0];

      // TODO: change the sum into p(k) = p(k-1) + 0.5*(
      // dp(k)+dp(k-1) ) to
      // increase accuracy
      for (kv.ilev = 1; kv.ilev < NUM_PHYSICAL_LEV; ++kv.ilev) {
        const int lev = kv.ilev / VECTOR_SIZE;
        const int vec = kv.ilev % VECTOR_SIZE;

        const int lev_prev = (kv.ilev - 1) / VECTOR_SIZE;
        const int vec_prev = (kv.ilev - 1) % VECTOR_SIZE;
        m_region.buffers.pressure(kv.ie, igp, jgp, lev)[vec] =
            m_region.buffers.pressure(kv.ie, igp, jgp, lev_prev)[vec_prev] +
            0.5 * m_region.m_dp3d(kv.ie, m_data.n0, igp, jgp,
                                  lev_prev)[vec_prev] +
            0.5 * m_region.m_dp3d(kv.ie, m_data.n0, igp, jgp, lev)[vec];
      }
    });
  }

  // Depends on DP3D, PHIS, DP3D, PHI, T_v
  // Modifies pressure, PHI
  KOKKOS_INLINE_FUNCTION
  void compute_scan_properties(KernelVariables &kv) const {
    if (kv.team.team_rank() == 0) {
      compute_pressure(kv);
      preq_hydrostatic(kv);
      preq_omega_ps(kv);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void compute_temperature_no_tracers_helper(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      m_region.buffers.temperature_virt(kv.ie, igp, jgp, kv.ilev) =
          m_region.m_t(kv.ie, m_data.n0, igp, jgp, kv.ilev);
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_temperature_tracers_helper(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      Scalar Qt = m_region.m_qdp(kv.ie, m_data.qn0, 0, igp, jgp, kv.ilev) /
                  m_region.m_dp3d(kv.ie, m_data.n0, igp, jgp, kv.ilev);
      Qt *= (PhysicalConstants::Rwater_vapor / PhysicalConstants::Rgas - 1.0);
      Qt += 1.0;
      m_region.buffers.temperature_virt(kv.ie, igp, jgp, kv.ilev) =
          m_region.m_t(kv.ie, m_data.n0, igp, jgp, kv.ilev) * Qt;
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

      m_region.buffers.vdp(kv.ie, 0, jgp, igp, kv.ilev) =
          m_region.m_u(kv.ie, m_data.n0, jgp, igp, kv.ilev) *
          m_region.m_dp3d(kv.ie, m_data.n0, jgp, igp, kv.ilev);

      m_region.buffers.vdp(kv.ie, 1, jgp, igp, kv.ilev) =
          m_region.m_v(kv.ie, m_data.n0, jgp, igp, kv.ilev) *
          m_region.m_dp3d(kv.ie, m_data.n0, jgp, igp, kv.ilev);

      m_region.m_derived_un0(kv.ie, jgp, igp, kv.ilev) =
          m_region.m_derived_un0(kv.ie, jgp, igp, kv.ilev) +
          m_data.eta_ave_w * m_region.buffers.vdp(kv.ie, 0, jgp, igp, kv.ilev);

      m_region.m_derived_vn0(kv.ie, jgp, igp, kv.ilev) =
          m_region.m_derived_vn0(kv.ie, jgp, igp, kv.ilev) +
          m_data.eta_ave_w * m_region.buffers.vdp(kv.ie, 1, jgp, igp, kv.ilev);
    });

    divergence_sphere(
        kv, m_region.m_dinv, m_region.m_metdet, m_deriv.get_dvv(),
        Homme::subview(m_region.buffers.vdp, kv.ie),
        Homme::subview(m_region.buffers.div_vdp, kv.ie));
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
      m_region.m_omega_p(kv.ie, jgp, igp, kv.ilev) +=
          m_data.eta_ave_w * m_region.buffers.omega_p(kv.ie, jgp, igp, kv.ilev);
    });
  }

  // Depends on T (global), OMEGA_P (global), U (global), V
  // (global),
  // SPHEREMP (global), T_v, and omega_p
  // block_3d_scalars
  KOKKOS_INLINE_FUNCTION
  void compute_temperature_np1(KernelVariables &kv) const {

    gradient_sphere(
        kv, m_region.m_dinv, m_deriv.get_dvv(),
        Homme::subview(m_region.m_t, kv.ie, m_data.n0),
        Homme::subview(m_region.buffers.temperature_grad, kv.ie));

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      Scalar vgrad_t =
          m_region.m_u(kv.ie, m_data.n0, igp, jgp, kv.ilev) *
              m_region.buffers.temperature_grad(kv.ie, 0, igp, jgp, kv.ilev) +
          m_region.m_v(kv.ie, m_data.n0, igp, jgp, kv.ilev) *
              m_region.buffers.temperature_grad(kv.ie, 1, igp, jgp, kv.ilev);

      // vgrad_t + kappa * T_v * omega_p
      Scalar ttens;
      ttens = -vgrad_t +
              PhysicalConstants::kappa *
                  m_region.buffers.temperature_virt(kv.ie, igp, jgp, kv.ilev) *
                  m_region.buffers.omega_p(kv.ie, igp, jgp, kv.ilev);

      Scalar temp_np1 = ttens * m_data.dt +
                        m_region.m_t(kv.ie, m_data.nm1, igp, jgp, kv.ilev);
      temp_np1 *= m_region.m_spheremp(kv.ie, igp, jgp);
      m_region.m_t(kv.ie, m_data.np1, igp, jgp, kv.ilev) = temp_np1;
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
      Scalar tmp = m_region.m_dp3d(kv.ie, m_data.nm1, jgp, igp, kv.ilev);
      tmp -= m_data.dt * m_region.buffers.div_vdp(kv.ie, jgp, igp, kv.ilev);
      m_region.m_dp3d(kv.ie, m_data.np1, jgp, igp, kv.ilev) =
          m_region.m_spheremp(kv.ie, jgp, igp) * tmp;
    });
  }

  // Computes the vertical advection of T and v
  // Not currently used
  //KOKKOS_INLINE_FUNCTION
  //void preq_vertadv(
  //    const TeamMember &,
  //    const ExecViewUnmanaged<const Scalar[NUM_LEV][NP][NP]> T,
  //    const ExecViewUnmanaged<const Scalar[NUM_LEV][2][NP][NP]> v,
  //    const ExecViewUnmanaged<const Scalar[NUM_LEV_P][NP][NP]> eta_dp_deta,
  //    const ExecViewUnmanaged<const Scalar[NUM_LEV][NP][NP]> rpdel,
  //    ExecViewUnmanaged<Scalar[NUM_LEV][NP][NP]> T_vadv,
  //    ExecViewUnmanaged<Scalar[NUM_LEV][2][NP][NP]> v_vadv) {
  //  constexpr const int k_0 = 0;
  //  for (int j = 0; j < NP; ++j) {
  //    for (int i = 0; i < NP; ++i) {
  //      Scalar facp = 0.5 * rpdel(k_0, j, i) * eta_dp_deta(k_0 + 1, j, i);
  //      T_vadv(k_0, j, i) = facp * (T(k_0 + 1, j, i) - T(k_0, j, i));
  //      for (int h = 0; h < 2; ++h) {
  //        v_vadv(k_0, h, j, i) = facp * (v(k_0 + 1, h, j, i) - v(k_0, h, j, i));
  //      }
  //    }
  //  }
  //  constexpr const int k_f = NUM_LEV - 1;
  //  for (int k = k_0 + 1; k < k_f; ++k) {
  //    for (int j = 0; j < NP; ++j) {
  //      for (int i = 0; i < NP; ++i) {
  //        Scalar facp = 0.5 * rpdel(k, j, i) * eta_dp_deta(k + 1, j, i);
  //        Scalar facm = 0.5 * rpdel(k, j, i) * eta_dp_deta(k, j, i);
  //        T_vadv(k, j, i) = facp * (T(k + 1, j, i) - T(k, j, i)) +
  //                          facm * (T(k, j, i) - T(k - 1, j, i));
  //        for (int h = 0; h < 2; ++h) {
  //          v_vadv(k, h, j, i) = facp * (v(k + 1, h, j, i) - v(k, h, j, i)) +
  //                               facm * (v(k, h, j, i) - v(k - 1, h, j, i));
  //        }
  //      }
  //    }
  //  }
  //  for (int j = 0; j < NP; ++j) {
  //    for (int i = 0; i < NP; ++i) {
  //      Scalar facm = 0.5 * rpdel(k_f, j, i) * eta_dp_deta(k_f, j, i);
  //      T_vadv(k_f, j, i) = facm * (T(k_f, j, i) - T(k_f - 1, j, i));
  //      for (int h = 0; h < 2; ++h) {
  //        v_vadv(k_f, h, j, i) = facm * (v(k_f, h, j, i) - v(k_f - 1, h, j, i));
  //      }
  //    }
  //  }
  //}

  KOKKOS_INLINE_FUNCTION
  void operator()(TeamMember team) const {
    KernelVariables kv(team);

    compute_temperature_div_vdp(kv);
    kv.team.team_barrier();

    compute_scan_properties(kv);
    kv.team.team_barrier();

    compute_phase_3(kv);
  }

  KOKKOS_INLINE_FUNCTION
  size_t shmem_size(const int team_size) const {
    return KernelVariables::shmem_size(team_size);
  }
};

} // Namespace Homme

#endif // CAAR_FUNCTOR_HPP
