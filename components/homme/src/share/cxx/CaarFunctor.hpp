#ifndef CAAR_FUNCTOR_HPP
#define CAAR_FUNCTOR_HPP

#include "CaarControl.hpp"
#include "CaarRegion.hpp"
#include "Derivative.hpp"

#include "SphereOperators.hpp"
#include "Types.hpp"

#include "Utility.hpp"
#include <fstream>
#include <iomanip>

namespace Homme {

struct CaarFunctor {
  CaarControl m_data;
  const CaarRegion m_region;
  const Derivative m_deriv;

  static constexpr Kokkos::Impl::ALL_t ALL = Kokkos::ALL;
  static constexpr int PRESSURE = 0;
  static constexpr int OMEGA_P = 1;
  static constexpr int T_V = 2;
  static constexpr int DIV_VDP = 3;

  struct KernelVariables {
    KOKKOS_INLINE_FUNCTION
    KernelVariables(const TeamMember &team_in)
        : team(team_in), scalar_buf_1(allocate_thread<Real, Real[NP][NP]>()),
          scalar_buf_2(allocate_thread<Real, Real[NP][NP]>()),
          vector_buf_1(allocate_thread<Real, Real[2][NP][NP]>()),
          vector_buf_2(allocate_thread<Real, Real[2][NP][NP]>()),
          ie(team.league_rank()), ilev(-1) {} //, igp(-1), jgp(-1) {}

    template <typename Primitive, typename Data>
    KOKKOS_INLINE_FUNCTION Primitive *allocate_team() const {
      ScratchView<Data> view(team.team_scratch(0));
      return view.data();
    }

    template <typename Primitive, typename Data>
    KOKKOS_INLINE_FUNCTION Primitive *allocate_thread() const {
      ScratchView<Data> view(team.thread_scratch(0));
      return view.data();
    }

    KOKKOS_INLINE_FUNCTION
    static size_t shmem_size(int team_size) {
      size_t mem_size =
          (2 * sizeof(Real[2][NP][NP]) + 2 * sizeof(Real[NP][NP])) * team_size;
      return mem_size;
    }

    const TeamMember &team;

    // Temporary buffers
    ExecViewUnmanaged<Real[NP][NP]> scalar_buf_1;
    ExecViewUnmanaged<Real[NP][NP]> scalar_buf_2;
    ExecViewUnmanaged<Real[2][NP][NP]> vector_buf_1;
    ExecViewUnmanaged<Real[2][NP][NP]> vector_buf_2;
    int ie, ilev;
    Real temp[2];
  }; // KernelVariables

  KOKKOS_INLINE_FUNCTION
  CaarFunctor() : m_data(), m_region(get_region()), m_deriv(get_derivative()) {
    // Nothing to be done here
  }

  KOKKOS_INLINE_FUNCTION
  CaarFunctor(const CaarControl &data)
      : m_data(data), m_region(get_region()), m_deriv(get_derivative()) {
    // Nothing to be done here
  }

  // Depends on PHI (after preq_hydrostatic), PECND
  // Modifies Ephi_grad
  // Computes \nabla (E + phi) + \nabla (P) * Rgas * T_v / P
  KOKKOS_INLINE_FUNCTION void compute_energy_grad(KernelVariables &kv) const {
    ExecViewUnmanaged<Real[NP][NP]> Ephi = kv.scalar_buf_1;
    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(kv.team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          // Kinetic energy + PHI (geopotential energy) +
          // PECND (potential energy?)
          Real k_energy =
              0.5 * (m_region.U(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
                         m_region.U(kv.ie, m_data.n0, kv.ilev, igp, jgp) +
                     m_region.V(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
                         m_region.V(kv.ie, m_data.n0, kv.ilev, igp, jgp));
          Ephi(igp, jgp) = k_energy + m_region.PHI(kv.ie, kv.ilev, igp, jgp) +
                           m_region.PECND(kv.ie, kv.ilev, igp, jgp);
        });

    ExecViewUnmanaged<const Real[NP][NP]> p_ilev =
        m_region.get_3d_buffer(kv.ie, PRESSURE, kv.ilev);

    ExecViewUnmanaged<Real[2][NP][NP]> gpterm = kv.vector_buf_2;
    gradient_sphere(kv.team, p_ilev, m_deriv.get_dvv(), m_region.DINV(kv.ie),
                    kv.vector_buf_1, gpterm);

    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(kv.team, 2 * NP * NP), [&](const int idx) {
          const int hgp = (idx / NP) / NP;
          const int igp = (idx / NP) % NP;
          const int jgp = idx % NP;
          gpterm(hgp, igp, jgp) *=
              PhysicalConstants::Rgas *
              (m_region.get_3d_buffer(kv.ie, T_V, kv.ilev, igp, jgp) /
               p_ilev(igp, jgp));
        });

    gradient_sphere_update(kv.team, Ephi, m_deriv.get_dvv(),
                           m_region.DINV(kv.ie), kv.vector_buf_1, gpterm);
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
    ExecViewUnmanaged<const Real[NP][NP]> p_ilev =
        m_region.get_3d_buffer(kv.ie, PRESSURE, kv.ilev);

    // kv.vector_buf_2 -> grad_E_phi + R T_v grad_p / p
    compute_energy_grad(kv);

    const ExecViewUnmanaged<Real[NP][NP]> vort = kv.scalar_buf_2;
    vorticity_sphere(kv.team, m_region.U(kv.ie, m_data.n0, kv.ilev),
                     m_region.V(kv.ie, m_data.n0, kv.ilev), m_deriv.get_dvv(),
                     m_region.METDET(kv.ie), m_region.D(kv.ie), kv.vector_buf_1,
                     vort);

    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(kv.team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;

          // Recycle vort to contain (fcor+vort)
          vort(igp, jgp) += m_region.FCOR(kv.ie, igp, jgp);

          kv.vector_buf_2(0, igp, jgp) *= -1;
          kv.vector_buf_2(0, igp, jgp) +=
              /* v_vadv(igp, jgp) + */ m_region.V(kv.ie, m_data.n0, kv.ilev,
                                                  igp, jgp) *
              vort(igp, jgp);
          kv.vector_buf_2(1, igp, jgp) *= -1;
          kv.vector_buf_2(1, igp, jgp) +=
              /* v_vadv(igp, jgp) + */ -m_region.U(kv.ie, m_data.n0, kv.ilev,
                                                   igp, jgp) *
              vort(igp, jgp);

          kv.vector_buf_2(0, igp, jgp) *= m_data.dt2;
          kv.vector_buf_2(0, igp, jgp) +=
              m_region.U(kv.ie, m_data.nm1, kv.ilev, igp, jgp);
          kv.vector_buf_2(1, igp, jgp) *= m_data.dt2;
          kv.vector_buf_2(1, igp, jgp) +=
              m_region.V(kv.ie, m_data.nm1, kv.ilev, igp, jgp);

          // Velocity at np1 = spheremp * buffer
          m_region.U(kv.ie, m_data.np1, kv.ilev, igp, jgp) =
              m_region.SPHEREMP(kv.ie, igp, jgp) * kv.vector_buf_2(0, igp, jgp);
          m_region.V(kv.ie, m_data.np1, kv.ilev, igp, jgp) =
              m_region.SPHEREMP(kv.ie, igp, jgp) * kv.vector_buf_2(1, igp, jgp);
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

                           m_region.ETA_DPDN(kv.ie, kv.ilev, igp, jgp) = 0;
                         });
  }

  // Depends on PHIS, DP3D, PHI, pressure, T_v
  // Modifies PHI
  KOKKOS_INLINE_FUNCTION
  void preq_hydrostatic(KernelVariables &kv) const {
    const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> pressure =
        m_region.get_3d_buffer(kv.ie, PRESSURE);
    const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> T_v =
        m_region.get_3d_buffer(kv.ie, T_V);
    const ExecViewUnmanaged<const Real[NP][NP]> phis = m_region.PHIS(kv.ie);
    const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> dp_n0 =
        m_region.DP3D(kv.ie, m_data.n0);

    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi = m_region.PHI(kv.ie);

    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(kv.team, NP * NP), [&](const int loop_idx) {
          const int igp = loop_idx / NP;
          const int jgp = loop_idx % NP;
          kv.scalar_buf_1(igp, jgp) = 0.0;
          for (int ilev = NUM_LEV - 1; ilev >= 0; --ilev) {
            // compute phi
            phi(ilev, igp, jgp) =
                phis(igp, jgp) + kv.scalar_buf_1(igp, jgp) +
                PhysicalConstants::Rgas * T_v(ilev, igp, jgp) *
                    (dp_n0(ilev, igp, jgp) * 0.5 / pressure(ilev, igp, jgp));

            // update phii
            kv.scalar_buf_1(igp, jgp) +=
                PhysicalConstants::Rgas * T_v(ilev, igp, jgp) * 2.0 *
                (dp_n0(ilev, igp, jgp) * 0.5 / pressure(ilev, igp, jgp));
            /*
             *        kv.vector_buf_2(0, igp, jgp) =
             *PhysicalConstants::Rgas /
             *pressure(kv.ilev, igp, jgp);
             *        kv.vector_buf_2(0, igp, jgp) *=
             *T_v(kv.ilev, igp, jgp);
             *        kv.vector_buf_2(0, igp, jgp) *=
             *dp_n0(kv.ilev, igp, jgp);
             *
             *        phi(kv.ilev, igp, jgp) = phis(igp,
             *jgp) +
             *kv.scalar_buf_1(igp,
             *jgp);
             *        // FMA so no temporary register needed
             *        phi(kv.ilev, igp, jgp) += 0.5 *
             *kv.vector_buf_2(0, igp, jgp);
             *
             *        kv.scalar_buf_1(igp, jgp) +=
             *kv.vector_buf_2(0, igp, jgp);
             */
          }
        });
  }

  // Depends on pressure, U_current, V_current, div_vdp,
  // omega_p
  KOKKOS_INLINE_FUNCTION
  void preq_omega_ps(KernelVariables &kv) const {
    // NOTE: we can't use a single TeamThreadRange loop,
    // since
    //       gradient_sphere requires a 'consistent'
    // pressure,
    //       meaning that we cannot update the different
    // pressure
    //       points within a level before the gradient is
    // complete!
    // Uses kv.scalar_buf_1 for intermediate computations
    // registers
    //      kv.scalar_buf_2 to store the intermediate
    // integration
    //      kv.vector_buf_1 to store the gradient
    //      kv.vector_buf_2 for the gradient buffer
    //
    const ExecViewUnmanaged<Real[2][NP][NP]> grad_p = kv.vector_buf_1;
    const ExecViewUnmanaged<Real[NP][NP]> integral = kv.scalar_buf_2;
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NP * NP),
                         [&](const int loop_idx) {
                           const int igp = loop_idx / NP;
                           const int jgp = loop_idx % NP;
                           integral(igp, jgp) = 0.0;
                         });
    for (int ilev = 0; ilev < NUM_LEV; ++ilev) {
      ExecViewUnmanaged<const Real[NP][NP]> p_ilev =
          m_region.get_3d_buffer(kv.ie, PRESSURE, ilev);

      gradient_sphere(kv.team, p_ilev, m_deriv.get_dvv(), m_region.DINV(kv.ie),
                      kv.vector_buf_2, grad_p);

      Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(kv.team, NP * NP), [&](const int loop_idx) {
            const int igp = loop_idx / NP;
            const int jgp = loop_idx % NP;
            Real vgrad_p = m_region.U(kv.ie, m_data.n0, ilev, igp, jgp) *
                               grad_p(0, igp, jgp) +
                           m_region.V(kv.ie, m_data.n0, ilev, igp, jgp) *
                               grad_p(1, igp, jgp);
            /*
             *vgrad_p += m_region.V(ie, m_data.n0, ilev,
             *igp,
             *jgp) *
             *           grad_p(1, igp, jgp);
             *vgrad_p -= integral(igp, jgp);
             *vgrad_p += -0.5 * m_scratch.div_vdp(ie, ilev,
             *igp,
             *jgp);
             */

            m_region.get_3d_buffer(kv.ie, OMEGA_P, ilev, igp, jgp) =
                vgrad_p / p_ilev(igp, jgp);

            ////////////////////////
            m_region.get_3d_buffer(kv.ie, OMEGA_P, ilev, igp, jgp) -=
                (1.0 / p_ilev(igp, jgp) * integral(igp, jgp) +
                 (0.5 / p_ilev(igp, jgp)) *
                     m_region.get_3d_buffer(kv.ie, DIV_VDP, ilev, igp, jgp));
            ////////////////////////
            integral(igp, jgp) +=
                m_region.get_3d_buffer(kv.ie, DIV_VDP, ilev, igp, jgp);
          });
    }
  }

  // Depends on DP3D
  KOKKOS_INLINE_FUNCTION
  void compute_pressure(KernelVariables &kv) const {
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure =
        m_region.get_3d_buffer(kv.ie, PRESSURE);
    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(kv.team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          pressure(0, igp, jgp) =
              m_data.hybrid_a(0) * m_data.ps0 +
              0.5 * m_region.DP3D(kv.ie, m_data.n0, 0, igp, jgp);

          // TODO: change the sum into p(k) = p(k-1) + 0.5*(
          // dp(k)+dp(k-1) ) to
          // increase accuracy
          for (kv.ilev = 1; kv.ilev < NUM_LEV; ++kv.ilev) {
            pressure(kv.ilev, igp, jgp) =
                pressure(kv.ilev - 1, igp, jgp) +
                0.5 * m_region.DP3D(kv.ie, m_data.n0, kv.ilev - 1, igp, jgp) +
                0.5 * m_region.DP3D(kv.ie, m_data.n0, kv.ilev, igp, jgp);
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
    ExecViewUnmanaged<Real[NP][NP]> T_v =
        m_region.get_3d_buffer(kv.ie, T_V, kv.ilev);
    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(kv.team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          T_v(igp, jgp) = m_region.T(kv.ie, m_data.n0, kv.ilev, igp, jgp);
        });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_temperature_tracers_helper(KernelVariables &kv) const {
    ExecViewUnmanaged<Real[NP][NP]> T_v =
        m_region.get_3d_buffer(kv.ie, T_V, kv.ilev);
    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(kv.team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;

          Real Qt = m_region.QDP(kv.ie, m_data.qn0, 0, kv.ilev, igp, jgp) /
                    m_region.DP3D(kv.ie, m_data.n0, kv.ilev, igp, jgp);
          Qt *= PhysicalConstants::Rwater_vapor / PhysicalConstants::Rgas - 1.0;
          Qt += 1.0;
          T_v(igp, jgp) = m_region.T(kv.ie, m_data.n0, kv.ilev, igp, jgp) * Qt;
        });
  }

  // Depends on DERIVED_UN0, DERIVED_VN0, METDET, DINV
  // Initializes div_vdp, which is used 2 times afterwards
  // Modifies DERIVED_UN0, DERIVED_VN0
  // Requires NUM_LEV * 5 * NP * NP
  KOKKOS_INLINE_FUNCTION
  void compute_div_vdp(KernelVariables &kv) const {
    // Create subviews to explicitly have static dimensions
    // ExecViewUnmanaged<Real[2][NP][NP]> vdp_ilev =
    // kv.vector_buf_2;

    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(kv.team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;

          kv.vector_buf_2(0, igp, jgp) =
              m_region.U(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
              m_region.DP3D(kv.ie, m_data.n0, kv.ilev, igp, jgp);
          kv.vector_buf_2(1, igp, jgp) =
              m_region.V(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
              m_region.DP3D(kv.ie, m_data.n0, kv.ilev, igp, jgp);

          kv.scalar_buf_1(igp, jgp) =
              m_data.eta_ave_w * kv.vector_buf_2(0, igp, jgp);
          m_region.DERIVED_UN0(kv.ie, kv.ilev, igp, jgp) =
              m_region.DERIVED_UN0(kv.ie, kv.ilev, igp, jgp) +
              kv.scalar_buf_1(igp, jgp);

          kv.scalar_buf_1(igp, jgp) =
              m_data.eta_ave_w * kv.vector_buf_2(1, igp, jgp);
          m_region.DERIVED_VN0(kv.ie, kv.ilev, igp, jgp) =
              m_region.DERIVED_VN0(kv.ie, kv.ilev, igp, jgp) +
              kv.scalar_buf_1(igp, jgp);
        });

    divergence_sphere(kv.team, kv.vector_buf_2, m_deriv.get_dvv(),
                      m_region.METDET(kv.ie), m_region.DINV(kv.ie),
                      kv.vector_buf_1,
                      m_region.get_3d_buffer(kv.ie, DIV_VDP, kv.ilev));
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
    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(kv.team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          m_region.OMEGA_P(kv.ie, kv.ilev, igp, jgp) +=
              m_data.eta_ave_w *
              m_region.get_3d_buffer(kv.ie, OMEGA_P, kv.ilev, igp, jgp);
        });
  }

  // Depends on T (global), OMEGA_P (global), U (global), V
  // (global),
  // SPHEREMP (global), T_v, and omega_p
  // block_3d_scalars
  KOKKOS_INLINE_FUNCTION
  void compute_temperature_np1(KernelVariables &kv) const {
    const ExecViewUnmanaged<const Real[NP][NP]> temperature =
        Kokkos::subview(m_region.T(kv.ie, m_data.n0), kv.ilev, ALL, ALL);

    const ExecViewUnmanaged<Real[2][NP][NP]> grad_tmp = kv.vector_buf_1;
    gradient_sphere(kv.team, temperature, m_deriv.get_dvv(),
                    m_region.DINV(kv.ie), kv.vector_buf_2, grad_tmp);

    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(kv.team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;

          Real vgrad_t = m_region.U(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
                             grad_tmp(0, igp, jgp) +
                         m_region.V(kv.ie, m_data.n0, kv.ilev, igp, jgp) *
                             grad_tmp(1, igp, jgp);

          // vgrad_t + kappa * T_v * omega_p
          Real &ttens = kv.temp[0];
          ttens = -vgrad_t +
                  PhysicalConstants::kappa *
                      m_region.get_3d_buffer(kv.ie, T_V, kv.ilev, igp, jgp) *
                      m_region.get_3d_buffer(kv.ie, OMEGA_P, kv.ilev, igp, jgp);

          Real temp_np1 = ttens * m_data.dt2 +
                          m_region.T(kv.ie, m_data.nm1, kv.ilev, igp, jgp);
          temp_np1 *= m_region.SPHEREMP(kv.ie, igp, jgp);

          m_region.T(kv.ie, m_data.np1, kv.ilev, igp, jgp) = temp_np1;
        });
  }

  // Depends on DERIVED_UN0, DERIVED_VN0, U, V,
  // Modifies DERIVED_UN0, DERIVED_VN0, OMEGA_P, T, and DP3D
  KOKKOS_INLINE_FUNCTION
  void compute_dp3d_np1(KernelVariables &kv) const {
    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(kv.team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          Real tmp = m_region.DP3D(kv.ie, m_data.nm1, kv.ilev, igp, jgp);
          tmp -= m_data.dt2 *
                 m_region.get_3d_buffer(kv.ie, DIV_VDP, kv.ilev, igp, jgp);
          m_region.DP3D(kv.ie, m_data.np1, kv.ilev, igp, jgp) =
              m_region.SPHEREMP(kv.ie, igp, jgp) * tmp;
        });
  }

  // Computes the vertical advection of T and v
  // Not currently used
  KOKKOS_INLINE_FUNCTION
  void preq_vertadv(
      const TeamMember &team,
      const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> T,
      const ExecViewUnmanaged<const Real[NUM_LEV][2][NP][NP]> v,
      const ExecViewUnmanaged<const Real[NUM_LEV_P][NP][NP]> eta_dp_deta,
      const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> rpdel,
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_vadv,
      ExecViewUnmanaged<Real[NUM_LEV][2][NP][NP]> v_vadv) {
    constexpr const int k_0 = 0;
    for (int j = 0; j < NP; ++j) {
      for (int i = 0; i < NP; ++i) {
        Real facp = 0.5 * rpdel(k_0, j, i) * eta_dp_deta(k_0 + 1, j, i);
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
          Real facp = 0.5 * rpdel(k, j, i) * eta_dp_deta(k + 1, j, i);
          Real facm = 0.5 * rpdel(k, j, i) * eta_dp_deta(k, j, i);
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
        Real facm = 0.5 * rpdel(k_f, j, i) * eta_dp_deta(k_f, j, i);
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
