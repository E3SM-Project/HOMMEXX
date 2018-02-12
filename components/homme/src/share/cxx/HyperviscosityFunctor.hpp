#ifndef HOMMEXX_HYPERVISCOSITY_FUNCTOR_HPP
#define HOMMEXX_HYPERVISCOSITY_FUNCTOR_HPP

#include "Elements.hpp"
#include "Derivative.hpp"
#include "SimulationParams.hpp"
#include "KernelVariables.hpp"
#include "SphereOperators.hpp"

namespace Homme
{

class HyperviscosityFunctor
{
  struct HyperviscosityData {
    HyperviscosityData(const int hypervis_subcycle_in, const Real nu_ratio_in, const Real nu_top_in,
                       const Real nu_in, const Real nu_p_in, const Real nu_s_in)
                      : hypervis_subcycle(hypervis_subcycle_in), nu_ratio(nu_ratio_in)
                      , nu_top(nu_top_in), nu(nu_in), nu_p(nu_p_in), nu_s(nu_s_in) {}


    const int   hypervis_subcycle;

    const Real  nu_ratio;
    const Real  nu_top;
    const Real  nu;
    const Real  nu_p;
    const Real  nu_s;

    int         np1;
    Real        dt;

    Real        eta_ave_w;
  };

public:

  struct TagFirstLaplace {};
  struct TagLaplace {};
  struct TagUpdateStates {};
  struct TagApplyInvMass {};
  struct TagHyperPreExchange {};

  HyperviscosityFunctor (const SimulationParams& params, const Elements& elements, const Derivative& deriv);

  void run (const int np1, const Real dt, const Real eta_ave_w);

  void biharmonic_wk_dp3d () const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagFirstLaplace&, const TeamMember& team) const {
    KernelVariables kv(team);
    // Laplacian of temperature
    laplace_simple(kv.team, m_deriv.get_dvv(),
                   Homme::subview(m_elements.m_dinv,kv.ie),
                   Homme::subview(m_elements.m_spheremp,kv.ie),
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.m_t,kv.ie,m_data.np1),
                   Homme::subview(m_elements.buffers.sphere_vector_buf,kv.ie),
                   Homme::subview(m_elements.buffers.ttens,kv.ie));
    // Laplacian of pressure
    laplace_simple(kv.team, m_deriv.get_dvv(),
                   Homme::subview(m_elements.m_dinv,kv.ie),
                   Homme::subview(m_elements.m_spheremp,kv.ie),
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.m_dp3d,kv.ie,m_data.np1),
                   Homme::subview(m_elements.buffers.sphere_vector_buf,kv.ie),
                   Homme::subview(m_elements.buffers.dptens,kv.ie));

    // Laplacian of velocity
    vlaplace_sphere_wk_contra(kv.team, m_data.nu_ratio, m_deriv.get_dvv(),
                              Homme::subview(m_elements.m_d,kv.ie),
                              Homme::subview(m_elements.m_dinv,kv.ie),
                              Homme::subview(m_elements.m_mp,kv.ie),
                              Homme::subview(m_elements.m_spheremp,kv.ie),
                              Homme::subview(m_elements.m_metinv,kv.ie),
                              Homme::subview(m_elements.m_metdet,kv.ie),
                              Homme::subview(m_elements.buffers.divergence_temp,kv.ie),
                              Homme::subview(m_elements.buffers.grad_buf,kv.ie),
                              Homme::subview(m_elements.buffers.sphere_vector_buf,kv.ie),
                              Homme::subview(m_elements.m_v,kv.ie,m_data.np1),
                              Homme::subview(m_elements.buffers.vtens,kv.ie));
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagLaplace&, const TeamMember& team) const {
    KernelVariables kv(team);
    // Laplacian of temperature
    laplace_simple(kv.team, m_deriv.get_dvv(),
                   Homme::subview(m_elements.m_dinv,kv.ie),
                   Homme::subview(m_elements.m_spheremp,kv.ie),
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.buffers.ttens,kv.ie),
                   Homme::subview(m_elements.buffers.sphere_vector_buf,kv.ie),
                   Homme::subview(m_elements.buffers.ttens,kv.ie));
    // Laplacian of pressure
    laplace_simple(kv.team, m_deriv.get_dvv(),
                   Homme::subview(m_elements.m_dinv,kv.ie),
                   Homme::subview(m_elements.m_spheremp,kv.ie),
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.buffers.dptens,kv.ie),
                   Homme::subview(m_elements.buffers.sphere_vector_buf,kv.ie),
                   Homme::subview(m_elements.buffers.dptens,kv.ie));

    // Laplacian of velocity
    vlaplace_sphere_wk_contra(kv.team, m_data.nu_ratio, m_deriv.get_dvv(),
                              Homme::subview(m_elements.m_d,kv.ie),
                              Homme::subview(m_elements.m_dinv,kv.ie),
                              Homme::subview(m_elements.m_mp,kv.ie),
                              Homme::subview(m_elements.m_spheremp,kv.ie),
                              Homme::subview(m_elements.m_metinv,kv.ie),
                              Homme::subview(m_elements.m_metdet, kv.ie),
                              Homme::subview(m_elements.buffers.divergence_temp,kv.ie),
                              Homme::subview(m_elements.buffers.grad_buf,kv.ie),
                              Homme::subview(m_elements.buffers.sphere_vector_buf,kv.ie),
                              Homme::subview(m_elements.buffers.vtens,kv.ie),
                              Homme::subview(m_elements.buffers.vtens,kv.ie));
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagUpdateStates&, const int idx) const {
    const int ie   =  idx / (NP*NP*NUM_LEV);
    const int igp  = (idx / (NP*NUM_LEV)) % NP;
    const int jgp  = (idx / NUM_LEV) % NP;
    const int ilev =  idx % NUM_LEV;

    // Apply inverse mass matrix
    m_elements.buffers.vtens(ie,0,igp,jgp,ilev) = (m_data.dt * m_elements.buffers.vtens(ie,0,igp,jgp,ilev) *
                                                   m_elements.m_rspheremp(ie,igp,jgp));
    m_elements.buffers.vtens(ie,1,igp,jgp,ilev) = (m_data.dt * m_elements.buffers.vtens(ie,1,igp,jgp,ilev) *
                                                   m_elements.m_rspheremp(ie,igp,jgp));
    m_elements.m_v(ie,m_data.np1,0,igp,jgp,ilev) += m_elements.buffers.vtens(ie,0,igp,jgp,ilev);
    m_elements.m_v(ie,m_data.np1,1,igp,jgp,ilev) += m_elements.buffers.vtens(ie,1,igp,jgp,ilev);

    m_elements.buffers.ttens(ie,igp,jgp,ilev) = (m_data.dt*m_elements.buffers.ttens(ie,igp,jgp,ilev) *
                                                 m_elements.m_rspheremp(ie,igp,jgp));
    const Scalar heating = m_elements.buffers.vtens(ie,0,igp,jgp,ilev)*m_elements.m_v(ie,m_data.np1,0,igp,jgp,ilev)
                         + m_elements.buffers.vtens(ie,1,igp,jgp,ilev)*m_elements.m_v(ie,m_data.np1,1,igp,jgp,ilev);
    m_elements.m_t(ie,m_data.np1,igp,jgp,ilev) =
      m_elements.m_t(ie,m_data.np1,igp,jgp,ilev) + m_elements.buffers.ttens(ie,igp,jgp,ilev) -
      heating/PhysicalConstants::cp;

    m_elements.m_dp3d(ie,m_data.np1,igp,jgp,ilev) = (m_elements.buffers.dptens(ie,igp,jgp,ilev) *
                                                     m_elements.m_rspheremp(ie,igp,jgp));
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagHyperPreExchange, const TeamMember &team) const {
    KernelVariables kv(team);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &point_idx) {
      const int igp = point_idx / NP;
      const int jgp = point_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &lev) {
        m_elements.m_derived_dpdiss_ave(kv.ie, igp, jgp, lev) +=
            m_data.eta_ave_w *
            m_elements.m_dp3d(kv.ie, m_data.np1, igp, jgp, lev) /
            m_data.hypervis_subcycle;
        m_elements.m_derived_dpdiss_biharmonic(kv.ie, igp, jgp, lev) +=
            m_data.eta_ave_w * m_elements.buffers.dptens(kv.ie, igp, jgp, lev) /
            m_data.hypervis_subcycle;
      });
    });
    kv.team_barrier();

    // Alias these for more descriptive names
    auto &laplace_v = m_elements.buffers.div_buf;
    auto &laplace_t = m_elements.buffers.lapl_buf_1;
    auto &laplace_dp3d = m_elements.buffers.lapl_buf_2;
    // laplace subfunctors cannot be called from a TeamThreadRange or
    // ThreadVectorRange
    constexpr int NUM_BIHARMONIC_PHYSICAL_LEVELS = 3;
    constexpr int NUM_BIHARMONIC_LEV = (NUM_BIHARMONIC_PHYSICAL_LEVELS + VECTOR_SIZE - 1) / VECTOR_SIZE;
    if (m_data.nu_top > 0) {

      // TODO: Only run on the levels we need to 0-2
      vlaplace_sphere_wk_contra<NUM_BIHARMONIC_LEV>(
            kv.team, m_data.nu_ratio, m_deriv.get_dvv(),
            Homme::subview(m_elements.m_d,kv.ie),
            Homme::subview(m_elements.m_dinv,kv.ie),
            Homme::subview(m_elements.m_mp,kv.ie),
            Homme::subview(m_elements.m_spheremp,kv.ie),
            Homme::subview(m_elements.m_metinv,kv.ie),
            Homme::subview(m_elements.m_metdet, kv.ie),
            Homme::subview(m_elements.buffers.lapl_buf_1, kv.ie),
            Homme::subview(m_elements.buffers.grad_buf, kv.ie),
            Homme::subview(m_elements.buffers.sphere_vector_buf,kv.ie),
            // input
            Homme::subview(m_elements.m_v, kv.ie, m_data.np1),
            // output
            Homme::subview(laplace_v, kv.ie));

      laplace_simple<NUM_BIHARMONIC_LEV>(
            kv.team, m_deriv.get_dvv(),
            Homme::subview(m_elements.m_dinv,kv.ie),
            Homme::subview(m_elements.m_spheremp,kv.ie),
            Homme::subview(m_elements.buffers.grad_buf, kv.ie),
            // input
            Homme::subview(m_elements.m_t, kv.ie, m_data.np1),
            Homme::subview(m_elements.buffers.sphere_vector_buf, kv.ie),
            // output
            Homme::subview(laplace_t, kv.ie));

      laplace_simple<NUM_BIHARMONIC_LEV>(
            kv.team, m_deriv.get_dvv(),
            Homme::subview(m_elements.m_dinv,kv.ie),
            Homme::subview(m_elements.m_spheremp,kv.ie),
            Homme::subview(m_elements.buffers.grad_buf, kv.ie),
            // input
            Homme::subview(m_elements.m_dp3d, kv.ie, m_data.np1),
            Homme::subview(m_elements.buffers.sphere_vector_buf, kv.ie),
            // output
            Homme::subview(laplace_dp3d, kv.ie));
    }
    kv.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &point_idx) {
      const int igp = point_idx / NP;
      const int jgp = point_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &lev) {
        m_elements.buffers.vtens(kv.ie, 0, igp, jgp, lev) *= -m_data.nu;
        m_elements.buffers.vtens(kv.ie, 1, igp, jgp, lev) *= -m_data.nu;
        m_elements.buffers.ttens(kv.ie, igp, jgp, lev) *= -m_data.nu_s;
        m_elements.buffers.dptens(kv.ie, igp, jgp, lev) *= -m_data.nu_p;
      });


      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, int(NUM_BIHARMONIC_LEV)),
                           [&](const int ilev) {
        m_elements.buffers.vtens(kv.ie, 0, igp, jgp, ilev) +=
            m_nu_scale_top[ilev] *
            laplace_v(kv.ie, 0, igp, jgp, ilev);
        m_elements.buffers.vtens(kv.ie, 1, igp, jgp, ilev) +=
            m_nu_scale_top[ilev] *
            laplace_v(kv.ie, 1, igp, jgp, ilev);

        m_elements.buffers.ttens(kv.ie, igp, jgp, ilev) +=
            m_nu_scale_top[ilev] *
            laplace_t(kv.ie, igp, jgp, ilev);

        m_elements.buffers.dptens(kv.ie, igp, jgp, ilev) +=
            m_nu_scale_top[ilev] *
            laplace_dp3d(kv.ie, igp, jgp, ilev);
      });

      // While for T and v we exchange the tendencies, for dp3d we exchange the updated state.
      // However, since the BE structure already has registerd the *tens quantities, we store
      // the updated state in dptens.
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &lev) {
          m_elements.buffers.dptens(kv.ie, igp, jgp, lev) *= m_data.dt;
          m_elements.buffers.dptens(kv.ie, igp, jgp, lev) += m_elements.m_dp3d(kv.ie,m_data.np1,igp,jgp,lev)
                                                           * m_elements.m_spheremp(kv.ie,igp,jgp);
      });
    });
  }

  Elements            m_elements;
  Derivative          m_deriv;
  HyperviscosityData  m_data;

  ExecViewManaged<Scalar[NUM_LEV]> m_nu_scale_top;
};

} // namespace Homme

#endif // HOMMEXX_HYPERVISCOSITY_FUNCTOR_HPP
