#ifndef HOMMEXX_HYPERVISCOSITY_FUNCTOR_IMPL_HPP
#define HOMMEXX_HYPERVISCOSITY_FUNCTOR_IMPL_HPP

#include "Elements.hpp"
#include "Derivative.hpp"
#include "SimulationParams.hpp"
#include "KernelVariables.hpp"
#include "SphereOperators.hpp"
#include "BoundaryExchange.hpp"

#include <memory>

namespace Homme
{

class HyperviscosityFunctorImpl
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

  using ScalarViewUnmanaged = ExecViewUnmanaged<Scalar *   [NP][NP][NUM_LEV]>;
  using VectorViewUnmanaged = ExecViewUnmanaged<Scalar *[2][NP][NP][NUM_LEV]>;

  struct HvfBuffers {
    ScalarViewUnmanaged       dptens;
    ScalarViewUnmanaged       ttens;
    VectorViewUnmanaged       vtens;

    ScalarViewUnmanaged       laplace_dp;
    ScalarViewUnmanaged       laplace_t;
    VectorViewUnmanaged       laplace_v;
  };

public:

  struct TagFirstLaplace {};
  struct TagLaplace {};
  struct TagUpdateStates {};
  struct TagApplyInvMass {};
  struct TagHyperPreExchange {};

  HyperviscosityFunctorImpl (const SimulationParams& params, const Elements& elements, const Derivative& deriv);

  size_t buffers_size () const;
  void init_buffers (Real* raw_buffer, const size_t buffer_size);

  void init_boundary_exchanges();

  void run (const int np1, const Real dt, const Real eta_ave_w);

  void biharmonic_wk_dp3d () const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagFirstLaplace&, const TeamMember& team) const {
    KernelVariables kv(team);
    // Laplacian of temperature
    m_sphere_ops.laplace_simple(kv,
                   Homme::subview(m_elements.m_t,kv.ie,m_data.np1),
                   Homme::subview(m_buffers.ttens,kv.ie));
    // Laplacian of pressure
    m_sphere_ops.laplace_simple(kv,
                   Homme::subview(m_elements.m_dp3d,kv.ie,m_data.np1),
                   Homme::subview(m_buffers.dptens,kv.ie));

    // Laplacian of velocity
    m_sphere_ops.vlaplace_sphere_wk_contra(kv, m_data.nu_ratio,
                              Homme::subview(m_elements.m_v,kv.ie,m_data.np1),
                              Homme::subview(m_buffers.vtens,kv.ie));
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagLaplace&, const TeamMember& team) const {
    KernelVariables kv(team);
    // Laplacian of temperature
    m_sphere_ops.laplace_simple(kv,
                   Homme::subview(m_buffers.ttens,kv.ie),
                   Homme::subview(m_buffers.ttens,kv.ie));
    // Laplacian of pressure
    m_sphere_ops.laplace_simple(kv,
                   Homme::subview(m_buffers.dptens,kv.ie),
                   Homme::subview(m_buffers.dptens,kv.ie));

    // Laplacian of velocity
    m_sphere_ops.vlaplace_sphere_wk_contra(kv, m_data.nu_ratio,
                              Homme::subview(m_buffers.vtens,kv.ie),
                              Homme::subview(m_buffers.vtens,kv.ie));
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagUpdateStates&, const int idx) const {
    const int ie   =  idx / (NP*NP*NUM_LEV);
    const int igp  = (idx / (NP*NUM_LEV)) % NP;
    const int jgp  = (idx / NUM_LEV) % NP;
    const int ilev =  idx % NUM_LEV;

    // Apply inverse mass matrix
    m_buffers.vtens(ie,0,igp,jgp,ilev) = (m_data.dt * m_buffers.vtens(ie,0,igp,jgp,ilev) *
                                          m_elements.m_rspheremp(ie,igp,jgp));
    m_buffers.vtens(ie,1,igp,jgp,ilev) = (m_data.dt * m_buffers.vtens(ie,1,igp,jgp,ilev) *
                                          m_elements.m_rspheremp(ie,igp,jgp));
    m_elements.m_v(ie,m_data.np1,0,igp,jgp,ilev) += m_buffers.vtens(ie,0,igp,jgp,ilev);
    m_elements.m_v(ie,m_data.np1,1,igp,jgp,ilev) += m_buffers.vtens(ie,1,igp,jgp,ilev);

    m_buffers.ttens(ie,igp,jgp,ilev) = (m_data.dt*m_buffers.ttens(ie,igp,jgp,ilev) *
                                        m_elements.m_rspheremp(ie,igp,jgp));
    const Scalar heating = m_buffers.vtens(ie,0,igp,jgp,ilev)*m_elements.m_v(ie,m_data.np1,0,igp,jgp,ilev)
                         + m_buffers.vtens(ie,1,igp,jgp,ilev)*m_elements.m_v(ie,m_data.np1,1,igp,jgp,ilev);
    m_elements.m_t(ie,m_data.np1,igp,jgp,ilev) =
      m_elements.m_t(ie,m_data.np1,igp,jgp,ilev) + m_buffers.ttens(ie,igp,jgp,ilev) -
      heating/PhysicalConstants::cp;

    m_elements.m_dp3d(ie,m_data.np1,igp,jgp,ilev) = (m_buffers.dptens(ie,igp,jgp,ilev) *
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
            m_data.eta_ave_w * m_buffers.dptens(kv.ie, igp, jgp, lev) /
            m_data.hypervis_subcycle;
      });
    });
    kv.team_barrier();

    // laplace subfunctors cannot be called from a TeamThreadRange or
    // ThreadVectorRange
    constexpr int NUM_BIHARMONIC_PHYSICAL_LEVELS = 3;
    constexpr int NUM_BIHARMONIC_LEV = (NUM_BIHARMONIC_PHYSICAL_LEVELS + VECTOR_SIZE - 1) / VECTOR_SIZE;
    if (m_data.nu_top > 0) {

      // TODO: Only run on the levels we need to 0-2
      m_sphere_ops.vlaplace_sphere_wk_contra<NUM_BIHARMONIC_LEV>(
            kv, m_data.nu_ratio,
            // input
            Homme::subview(m_elements.m_v, kv.ie, m_data.np1),
            // output
            Homme::subview(m_buffers.laplace_v, kv.ie));

      m_sphere_ops.laplace_simple<NUM_BIHARMONIC_LEV>(
            kv,
            // input
            Homme::subview(m_elements.m_t, kv.ie, m_data.np1),
            // output
            Homme::subview(m_buffers.laplace_t, kv.ie));

      m_sphere_ops.laplace_simple<NUM_BIHARMONIC_LEV>(
            kv,
            // input
            Homme::subview(m_elements.m_dp3d, kv.ie, m_data.np1),
            // output
            Homme::subview(m_buffers.laplace_dp, kv.ie));
    }
    kv.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(kv.team, NP * NP),
                         [&](const int &point_idx) {
      const int igp = point_idx / NP;
      const int jgp = point_idx % NP;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &lev) {
        m_buffers.vtens(kv.ie, 0, igp, jgp, lev) *= -m_data.nu;
        m_buffers.vtens(kv.ie, 1, igp, jgp, lev) *= -m_data.nu;
        m_buffers.ttens(kv.ie, igp, jgp, lev) *= -m_data.nu_s;
        m_buffers.dptens(kv.ie, igp, jgp, lev) *= -m_data.nu_p;
      });


      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, int(NUM_BIHARMONIC_LEV)),
                           [&](const int ilev) {
        m_buffers.vtens(kv.ie, 0, igp, jgp, ilev) +=
            m_nu_scale_top[ilev] *
            m_buffers.laplace_v(kv.ie, 0, igp, jgp, ilev);
        m_buffers.vtens(kv.ie, 1, igp, jgp, ilev) +=
            m_nu_scale_top[ilev] *
            m_buffers.laplace_v(kv.ie, 1, igp, jgp, ilev);

        m_buffers.ttens(kv.ie, igp, jgp, ilev) +=
            m_nu_scale_top[ilev] *
            m_buffers.laplace_t(kv.ie, igp, jgp, ilev);

        m_buffers.dptens(kv.ie, igp, jgp, ilev) +=
            m_nu_scale_top[ilev] *
            m_buffers.laplace_dp(kv.ie, igp, jgp, ilev);
      });

      // While for T and v we exchange the tendencies, for dp3d we exchange the updated state.
      // However, since the BE structure already has registerd the *tens quantities, we store
      // the updated state in dptens.
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(kv.team, NUM_LEV),
                           [&](const int &lev) {
          m_buffers.dptens(kv.ie, igp, jgp, lev) *= m_data.dt;
          m_buffers.dptens(kv.ie, igp, jgp, lev) += m_elements.m_dp3d(kv.ie,m_data.np1,igp,jgp,lev)
                                                  * m_elements.m_spheremp(kv.ie,igp,jgp);
      });
    });
  }

private:

  Elements            m_elements;
  Derivative          m_deriv;
  HyperviscosityData  m_data;
  SphereOperators     m_sphere_ops;
  HvfBuffers          m_buffers;

  std::shared_ptr<BoundaryExchange> m_be;

  ExecViewManaged<Scalar[NUM_LEV]> m_nu_scale_top;
};

} // namespace Homme

#endif // HOMMEXX_HYPERVISCOSITY_FUNCTOR_IMPL_HPP
