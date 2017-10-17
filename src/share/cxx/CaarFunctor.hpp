#ifndef CAAR_FUNCTOR_HPP
#define CAAR_FUNCTOR_HPP

#include "Types.hpp"
#include "Control.hpp"
#include "Elements.hpp"
#include "Derivative.hpp"
#include "KernelVariables.hpp"
#include "SphereOperators.hpp"

#include "Utility.hpp"
#include "profiling.hpp"

#include <assert.h>

namespace Homme {

struct CaarFunctor {
  Control           m_data;
  const Elements    m_elements;
  const Derivative  m_deriv;

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
  KOKKOS_INLINE_FUNCTION void compute_energy_grad(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      for (int ilev=0; ilev<NUM_LEV; ++ilev) {
        // pre-fill energy_grad with the pressure(_grad)-temperature part
        m_elements.buffers.energy_grad(kv.ie, 0, igp, jgp, ilev) =
            PhysicalConstants::Rgas *
            (m_elements.buffers.temperature_virt(kv.ie, igp, jgp, ilev) /
             m_elements.buffers.pressure(kv.ie, igp, jgp, ilev)) *
            m_elements.buffers.pressure_grad(kv.ie, 0, igp, jgp, ilev);

        m_elements.buffers.energy_grad(kv.ie, 1, igp, jgp, ilev) =
            PhysicalConstants::Rgas *
            (m_elements.buffers.temperature_virt(kv.ie, igp, jgp, ilev) /
             m_elements.buffers.pressure(kv.ie, igp, jgp, ilev)) *
            m_elements.buffers.pressure_grad(kv.ie, 1, igp, jgp, ilev);

        // Kinetic energy + PHI (geopotential energy) +
        // PECND (potential energy?)
        Scalar k_energy =
            0.5 * (m_elements.m_u(kv.ie, m_data.n0, igp, jgp, ilev) *
                       m_elements.m_u(kv.ie, m_data.n0, igp, jgp, ilev) +
                   m_elements.m_v(kv.ie, m_data.n0, igp, jgp, ilev) *
                       m_elements.m_v(kv.ie, m_data.n0, igp, jgp, ilev));
        m_elements.buffers.ephi(kv.ie, igp, jgp, ilev) =
            k_energy + (m_elements.m_phi(kv.ie, igp, jgp, ilev) +
                        m_elements.m_pecnd(kv.ie, igp, jgp, ilev));
      }
    });
    kv.team_barrier();

    gradient_sphere_update(
        kv, m_elements.m_dinv, m_deriv.get_dvv(),
        Kokkos::subview(m_elements.buffers.ephi, kv.ie, ALL, ALL, ALL),
        m_elements.buffers.grad_buf, Kokkos::subview(m_elements.buffers.energy_grad,
                                                   kv.ie, ALL, ALL, ALL, ALL));
  } // TESTED 1

#ifdef NDEBUG
  KOKKOS_INLINE_FUNCTION void check_dp3d(KernelVariables &kv) const {}
#else
  KOKKOS_INLINE_FUNCTION void check_dp3d(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team,
                                                   NP * NP * NUM_PHYSICAL_LEV),
                         [&](const int &idx) {
      const int igp = (idx / NUM_PHYSICAL_LEV) / NP;
      const int jgp = (idx / NUM_PHYSICAL_LEV) % NP;
      const int ilev = idx / VECTOR_SIZE;
      const int ivec = idx % VECTOR_SIZE;
      assert(m_elements.m_dp3d(kv.ie, m_data.np1, igp, jgp, ilev)[ivec] > 0.0);
    });
    kv.team_barrier();
  }
#endif

  // Depends on pressure, PHI, U_current, V_current, METDET,
  // D, DINV, U, V, FCOR, SPHEREMP, T_v, ETA_DPDN
  KOKKOS_INLINE_FUNCTION void compute_phase_3(KernelVariables &kv) const {
    compute_eta_dpdn_rsplit(kv);
    compute_omega_p(kv);
    compute_temperature_np1(kv);
    compute_velocity_np1(kv);
    // Note this is dependent on eta_dot_dpdn from other levels and will cause
    // issues when rsplit is 0
    compute_dp3d_np1(kv);
    check_dp3d(kv);
  } // TRIVIAL

  // Depends on pressure, PHI, U_current, V_current, METDET,
  // D, DINV, U, V, FCOR, SPHEREMP, T_v
  KOKKOS_INLINE_FUNCTION
  void compute_velocity_np1(KernelVariables &kv) const {
    compute_energy_grad(kv);

    vorticity_sphere(
        kv, m_elements.m_d, m_elements.m_metdet, m_deriv.get_dvv(),
        Kokkos::subview(m_elements.m_u, kv.ie, m_data.n0, ALL, ALL, ALL),
        Kokkos::subview(m_elements.m_v, kv.ie, m_data.n0, ALL, ALL, ALL),
        m_elements.buffers.vort_buf,
        Kokkos::subview(m_elements.buffers.vorticity, kv.ie, ALL, ALL, ALL));

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      for (int ilev=0; ilev<NUM_LEV; ++ilev) {
        // Recycle vort to contain (fcor+vort)
        m_elements.buffers.vorticity(kv.ie, igp, jgp, ilev) +=
            m_elements.m_fcor(kv.ie, igp, jgp);

        m_elements.buffers.energy_grad(kv.ie, 0, igp, jgp, ilev) *= -1;
        m_elements.buffers.energy_grad(kv.ie, 0, igp, jgp, ilev) +=
            /* v_vadv(igp, jgp) + */ m_elements.m_v(kv.ie, m_data.n0, igp, jgp,
                                                  ilev) *
            m_elements.buffers.vorticity(kv.ie, igp, jgp, ilev);
        m_elements.buffers.energy_grad(kv.ie, 1, igp, jgp, ilev) *= -1;
        m_elements.buffers.energy_grad(kv.ie, 1, igp, jgp, ilev) +=
            /* v_vadv(igp, jgp) + */ -m_elements.m_u(kv.ie, m_data.n0, igp, jgp,
                                                   ilev) *
            m_elements.buffers.vorticity(kv.ie, igp, jgp, ilev);

        m_elements.buffers.energy_grad(kv.ie, 0, igp, jgp, ilev) *= m_data.dt;
        m_elements.buffers.energy_grad(kv.ie, 0, igp, jgp, ilev) +=
            m_elements.m_u(kv.ie, m_data.nm1, igp, jgp, ilev);
        m_elements.buffers.energy_grad(kv.ie, 1, igp, jgp, ilev) *= m_data.dt;
        m_elements.buffers.energy_grad(kv.ie, 1, igp, jgp, ilev) +=
            m_elements.m_v(kv.ie, m_data.nm1, igp, jgp, ilev);

        // Velocity at np1 = spheremp * buffer
        m_elements.m_u(kv.ie, m_data.np1, igp, jgp, ilev) =
            m_elements.m_spheremp(kv.ie, igp, jgp) *
            m_elements.buffers.energy_grad(kv.ie, 0, igp, jgp, ilev);
        m_elements.m_v(kv.ie, m_data.np1, igp, jgp, ilev) =
            m_elements.m_spheremp(kv.ie, igp, jgp) *
            m_elements.buffers.energy_grad(kv.ie, 1, igp, jgp, ilev);
      }
    });
    kv.team_barrier();
  } // UNTESTED 2

  // TODO: Use partial template specialization to determine if we need this
  // Make a templated subclass of an untemplated version of CaarFunctor
  // Specialize the templated subclass to implement these based on rsplit
  KOKKOS_INLINE_FUNCTION
  void compute_eta_dpdn_rsplit(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         KOKKOS_LAMBDA(const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      for (int ilev=0; ilev<NUM_INTERFACE_LEV; ++ilev) {
        m_elements.m_eta_dot_dpdn(kv.ie, jgp, igp, ilev) = 0;
      }
    });
    kv.team_barrier();
  } // TRIVIAL

  KOKKOS_INLINE_FUNCTION
  void compute_eta_dpdn_no_rsplit(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         KOKKOS_LAMBDA(const int idx) {
      // TODO: Compute the actual value for this if rsplit=0.
      // Note this will be unsafe to thread over levels,
      // so thread over points instead
      // for (int ilev=0; ilev<NUM_INTERFACE_LEV; ++ilev) {
      //   m_elements.eta_dot_dpdn += eta_ave_w*eta_dot_dpdn
      // }
    });
  } // Unimplemented

  // Depends on PHIS, DP3D, PHI, pressure, T_v
  // Modifies PHI
  KOKKOS_INLINE_FUNCTION
  void preq_hydrostatic(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      Real integration = 0.0;
      for (int ilevel = NUM_PHYSICAL_LEV - 1; ilevel >= 0; --ilevel) {
        int ilev = ilevel / VECTOR_SIZE;
        int v   = ilevel % VECTOR_SIZE;
        // compute phi
        m_elements.m_phi(kv.ie, jgp, igp, ilev)[v] =
            m_elements.m_phis(kv.ie, jgp, igp) + integration +
            PhysicalConstants::Rgas * m_elements.buffers.temperature_virt(
                                          kv.ie, jgp, igp, ilev)[v] *
                (m_elements.m_dp3d(kv.ie, m_data.n0, jgp, igp, ilev)[v] *
                 0.5 /
                 m_elements.buffers.pressure(kv.ie, jgp, igp, ilev)[v]);

        // update phii
        integration +=
            PhysicalConstants::Rgas *
            m_elements.buffers.temperature_virt(kv.ie, jgp, igp, ilev)[v] *
            2.0 *
            (m_elements.m_dp3d(kv.ie, m_data.n0, jgp, igp, ilev)[v] * 0.5 /
             m_elements.buffers.pressure(kv.ie, jgp, igp, ilev)[v]);

      }
    });
    kv.team_barrier();
  } // TESTED 3

  // Depends on pressure, U_current, V_current, div_vdp,
  // omega_p
  KOKKOS_INLINE_FUNCTION
  void preq_omega_ps(KernelVariables &kv) const {
    // NOTE: we can't use a single TeamThreadRange loop,
    // since gradient_sphere requires a 'consistent'
    // pressure, meaning that we cannot update the different
    // pressure points within a level before the gradient is
    // complete!
    ExecViewUnmanaged<Real *[NP][NP]> integration = m_elements.buffers.preq_buf;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int loop_idx) {
      const int igp = loop_idx / NP;
      const int jgp = loop_idx % NP;
      integration(kv.ie, igp, jgp) = 0.0;
    });

    for (int ilev = 0; ilev < NUM_LEV; ++ilev) {
      gradient_sphere(
          kv, m_elements.m_dinv, m_deriv.get_dvv(),
          Kokkos::subview(m_elements.buffers.pressure, kv.ie, ALL, ALL, ALL),
          m_elements.buffers.grad_buf,
          Kokkos::subview(m_elements.buffers.pressure_grad, kv.ie, ALL, ALL, ALL,
                          ALL));
      Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                           [&](const int loop_idx) {
        const int igp = loop_idx / NP;
        const int jgp = loop_idx % NP;
        Scalar vgrad_p =
            m_elements.m_u(kv.ie, m_data.n0, igp, jgp, ilev) *
                m_elements.buffers.pressure_grad(kv.ie, 0, igp, jgp, ilev) +
            m_elements.m_v(kv.ie, m_data.n0, igp, jgp, ilev) *
                m_elements.buffers.pressure_grad(kv.ie, 1, igp, jgp, ilev);

        m_elements.buffers.omega_p(kv.ie, igp, jgp, ilev) =
            vgrad_p / m_elements.buffers.pressure(kv.ie, igp, jgp, ilev);

        for (int vec = 0; vec < VECTOR_SIZE; ++vec) {
          Real div_vdp =
              m_elements.buffers.div_vdp(kv.ie, igp, jgp, ilev)[vec];
          Real ckk =
              0.5 / m_elements.buffers.pressure(kv.ie, igp, jgp, ilev)[vec];
          m_elements.buffers.omega_p(kv.ie, igp, jgp, ilev)[vec] -=
              (2.0 * ckk * integration(kv.ie, igp, jgp) + ckk * div_vdp);
          integration(kv.ie, igp, jgp) += div_vdp;
        }
      });
      kv.team_barrier();
    }
  } // TESTED 4

  // Depends on DP3D
  KOKKOS_INLINE_FUNCTION
  void compute_pressure(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      m_elements.buffers.pressure(kv.ie, igp, jgp, 0)[0] =
          m_data.hybrid_a(0) * m_data.ps0 +
          0.5 * m_elements.m_dp3d(kv.ie, m_data.n0, igp, jgp, 0)[0];

      // TODO: change the sum into
      // p(k) = p(k-1) + 0.5*(dp(k)+dp(k-1))
      // to increase accuracy
      for (int ilev = 1; ilev < NUM_PHYSICAL_LEV; ++ilev) {
        const int lev = ilev / VECTOR_SIZE;
        const int vec = ilev % VECTOR_SIZE;

        const int lev_prev = (ilev - 1) / VECTOR_SIZE;
        const int vec_prev = (ilev - 1) % VECTOR_SIZE;
        m_elements.buffers.pressure(kv.ie, igp, jgp, lev)[vec] =
            m_elements.buffers.pressure(kv.ie, igp, jgp, lev_prev)[vec_prev] +
            0.5 * m_elements.m_dp3d(kv.ie, m_data.n0, igp, jgp,
                                    lev_prev)[vec_prev] +
            0.5 * m_elements.m_dp3d(kv.ie, m_data.n0, igp, jgp, lev)[vec];
      }
    });
    kv.team_barrier();
  } // TESTED 5

  // Depends on DP3D, PHIS, DP3D, PHI, T_v
  // Modifies pressure, PHI
  KOKKOS_INLINE_FUNCTION
  void compute_scan_properties(KernelVariables &kv) const {
    // Use this instead of Kokkos::single(Kokkos::PerTeam
    // due to Kokkos failing to execute the TeamThreadRange parallel for
    // on CUDA
    compute_pressure(kv);
    preq_hydrostatic(kv);
    preq_omega_ps(kv);
  } // TRIVIAL

  KOKKOS_INLINE_FUNCTION
  void compute_temperature_no_tracers_helper(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      for (int ilev = 0; ilev < NUM_LEV; ++ilev) {
        m_elements.buffers.temperature_virt(kv.ie, igp, jgp, ilev) =
            m_elements.m_t(kv.ie, m_data.n0, igp, jgp, ilev);
      }
    });
    kv.team_barrier();
  } // TESTED 6

  KOKKOS_INLINE_FUNCTION
  void compute_temperature_tracers_helper(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      for (int ilev = 0; ilev < NUM_LEV; ++ilev) {
        Scalar Qt = m_elements.m_qdp(kv.ie, m_data.qn0, 0, igp, jgp, ilev) /
                    m_elements.m_dp3d(kv.ie, m_data.n0, igp, jgp, ilev);
        Qt *= (PhysicalConstants::Rwater_vapor / PhysicalConstants::Rgas - 1.0);
        Qt += 1.0;
        m_elements.buffers.temperature_virt(kv.ie, igp, jgp, ilev) =
            m_elements.m_t(kv.ie, m_data.n0, igp, jgp, ilev) * Qt;
      }
    });
    kv.team_barrier();
  } // TESTED 7

  // Depends on DERIVED_UN0, DERIVED_VN0, METDET, DINV
  // Initializes div_vdp, which is used 2 times afterwards
  // Modifies DERIVED_UN0, DERIVED_VN0
  // Requires NUM_LEV * 5 * NP * NP
  KOKKOS_INLINE_FUNCTION
  void compute_div_vdp(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      for (int ilev = 0; ilev < NUM_LEV; ++ilev) {
        m_elements.buffers.vdp(kv.ie, 0, igp, jgp, ilev) =
            m_elements.m_u(kv.ie, m_data.n0, igp, jgp, ilev) *
            m_elements.m_dp3d(kv.ie, m_data.n0, igp, jgp, ilev);

        m_elements.buffers.vdp(kv.ie, 1, igp, jgp, ilev) =
            m_elements.m_v(kv.ie, m_data.n0, igp, jgp, ilev) *
            m_elements.m_dp3d(kv.ie, m_data.n0, igp, jgp, ilev);

        m_elements.m_derived_un0(kv.ie, igp, jgp, ilev) +=
            m_data.eta_ave_w * m_elements.buffers.vdp(kv.ie, 0, igp, jgp, ilev);

        m_elements.m_derived_vn0(kv.ie, igp, jgp, ilev) +=
            m_data.eta_ave_w * m_elements.buffers.vdp(kv.ie, 1, igp, jgp, ilev);
      }
    });
    kv.team_barrier();

    divergence_sphere(
        kv, m_elements.m_dinv, m_elements.m_metdet, m_deriv.get_dvv(),
        Kokkos::subview(m_elements.buffers.vdp, kv.ie, ALL, ALL, ALL, ALL),
        m_elements.buffers.div_buf,
        Kokkos::subview(m_elements.buffers.div_vdp, kv.ie, ALL, ALL, ALL));
  } // TESTED 8

  // Depends on T_current, DERIVE_UN0, DERIVED_VN0, METDET,
  // DINV
  // Might depend on QDP, DP3D_current
  KOKKOS_INLINE_FUNCTION
  void compute_temperature_div_vdp(KernelVariables &kv) const {
    if (m_data.qn0 == -1) {
      compute_temperature_no_tracers_helper(kv);
    } else {
      compute_temperature_tracers_helper(kv);
    }
    compute_div_vdp(kv);
  } // TESTED 9

  KOKKOS_INLINE_FUNCTION
  void compute_omega_p(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      for (int ilev=0; ilev<NUM_LEV; ++ilev) {
        m_elements.m_omega_p(kv.ie, jgp, igp, ilev) +=
            m_data.eta_ave_w * m_elements.buffers.omega_p(kv.ie, jgp, igp, ilev);
      }
    });
    kv.team_barrier();
  } // TESTED 10

  // Depends on T (global), OMEGA_P (global), U (global), V
  // (global),
  // SPHEREMP (global), T_v, and omega_p
  // block_3d_scalars
  KOKKOS_INLINE_FUNCTION
  void compute_temperature_np1(KernelVariables &kv) const {

    gradient_sphere(
        kv, m_elements.m_dinv, m_deriv.get_dvv(),
        Kokkos::subview(m_elements.m_t, kv.ie, m_data.n0, ALL, ALL, ALL),
        m_elements.buffers.grad_buf,
        Kokkos::subview(m_elements.buffers.temperature_grad, kv.ie, ALL, ALL, ALL,
                        ALL));

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      for (int ilev=0; ilev<NUM_LEV; ++ilev) {
        const Scalar vgrad_t =
            m_elements.m_u(kv.ie, m_data.n0, igp, jgp, ilev) *
                m_elements.buffers.temperature_grad(kv.ie, 0, igp, jgp, ilev) +
            m_elements.m_v(kv.ie, m_data.n0, igp, jgp, ilev) *
                m_elements.buffers.temperature_grad(kv.ie, 1, igp, jgp, ilev);

        // vgrad_t + kappa * T_v * omega_p
        const Scalar ttens = -vgrad_t +
                PhysicalConstants::kappa *
                    m_elements.buffers.temperature_virt(kv.ie, igp, jgp, ilev) *
                    m_elements.buffers.omega_p(kv.ie, igp, jgp, ilev);

        Scalar temp_np1 = ttens * m_data.dt +
                          m_elements.m_t(kv.ie, m_data.nm1, igp, jgp, ilev);
        temp_np1 *= m_elements.m_spheremp(kv.ie, igp, jgp);
        m_elements.m_t(kv.ie, m_data.np1, igp, jgp, ilev) = temp_np1;
      }
    });
    kv.team_barrier();
  } // TESTED 11

  // Depends on DERIVED_UN0, DERIVED_VN0, U, V,
  // Modifies DERIVED_UN0, DERIVED_VN0, OMEGA_P, T, and DP3D
  KOKKOS_INLINE_FUNCTION
  void compute_dp3d_np1(KernelVariables &kv) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      for (int ilev=0; ilev<NUM_LEV; ++ilev) {
        Scalar tmp = m_elements.m_eta_dot_dpdn(kv.ie, igp, jgp, ilev);
        tmp.shift_left(1);
        tmp[VECTOR_SIZE - 1] =
            m_elements.m_eta_dot_dpdn(kv.ie, igp, jgp, ilev + 1)[0];
        // Add div_vdp before subtracting the previous value to eta_dot_dpdn
        // This will hopefully reduce numeric error
        tmp += m_elements.buffers.div_vdp(kv.ie, igp, jgp, ilev);
        tmp -= m_elements.m_eta_dot_dpdn(kv.ie, igp, jgp, ilev);
        tmp = m_elements.m_dp3d(kv.ie, m_data.nm1, igp, jgp, ilev) -
              tmp * m_data.dt;

        m_elements.m_dp3d(kv.ie, m_data.np1, igp, jgp, ilev) =
            m_elements.m_spheremp(kv.ie, igp, jgp) * tmp;
      }
    });
    kv.team_barrier();
  } // TESTED 12

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
  } // UNTESTED 13

  KOKKOS_INLINE_FUNCTION
  void operator()(TeamMember team) const {
    start_timer("caar compute");
    KernelVariables kv(team);

    compute_temperature_div_vdp(kv);
    kv.team.team_barrier();

    compute_scan_properties(kv);
    kv.team.team_barrier();

    compute_phase_3(kv);
    stop_timer("caar compute");
  }

  KOKKOS_INLINE_FUNCTION
  size_t shmem_size(const int team_size) const {
    return KernelVariables::shmem_size(team_size);
  }
};

} // Namespace Homme

#endif // CAAR_FUNCTOR_HPP
