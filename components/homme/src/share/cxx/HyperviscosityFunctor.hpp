#ifndef HOMMEXX_HYPERVISCOSITY_FUNCTOR_HPP
#define HOMMEXX_HYPERVISCOSITY_FUNCTOR_HPP

#include "Control.hpp"
#include "Elements.hpp"
#include "Derivative.hpp"
#include "KernelVariables.hpp"
#include "SphereOperators.hpp"

namespace Homme
{

class HyperviscosityFunctor
{
public:

  struct TagFetchStates {};
  struct TagLaplace {};
  struct TagUpdateStates {};
  struct TagApplyInvMass {};

  HyperviscosityFunctor (const Control& data, const Elements& elements, const Derivative& deriv);

  void run (const int hypervis_subcycle) const;

  void biharmonic_wk_dp3d () const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagLaplace&, const TeamMember& team) const {
    KernelVariables kv(team);
    // Laplacian of temperature
    laplace_tensor(kv, m_elements.m_dinv, m_elements.m_spheremp, m_deriv.get_dvv(),
                   m_elements.m_tensorVisc,
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.buffers.ttens,kv.ie),
                   m_elements.buffers.sphere_vector_buf,
                   Homme::subview(m_elements.buffers.ttens,kv.ie));
    // Laplacian of pressure
    laplace_tensor(kv, m_elements.m_dinv, m_elements.m_spheremp, m_deriv.get_dvv(),
                   m_elements.m_tensorVisc,
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.buffers.dptens,kv.ie),
                   m_elements.buffers.sphere_vector_buf,
                   Homme::subview(m_elements.buffers.dptens,kv.ie));

    // Laplacian of velocity
    vlaplace_sphere_wk_contra(kv,m_elements.m_d,m_elements.m_dinv,m_elements.m_mp,m_elements.m_spheremp,
                              m_elements.m_metinv, m_elements.m_metdet, m_deriv.get_dvv(), m_data.nu_ratio,
                              Homme::subview(m_elements.buffers.divergence_temp,kv.ie),
                              Homme::subview(m_elements.buffers.vorticity_temp,kv.ie),
                              Homme::subview(m_elements.buffers.grad_buf,kv.ie),
                              Homme::subview(m_elements.buffers.curl_buf,kv.ie),
                              m_elements.buffers.sphere_vector_buf,
                              Homme::subview(m_elements.buffers.vtens,kv.ie),
                              Homme::subview(m_elements.buffers.vtens,kv.ie));
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagUpdateStates&, const int idx) const {
    const int ie   =  idx / (NP*NP*NUM_LEV);
    const int igp  = (idx / (NP*NUM_LEV)) % NP;
    const int jgp  = (idx / NUM_LEV) % NP;
    const int ilev =  idx % NUM_LEV;

    m_elements.m_v(ie,m_data.np1,0,igp,jgp,ilev) += m_elements.buffers.vtens(ie,0,igp,jgp,ilev);
    m_elements.m_v(ie,m_data.np1,1,igp,jgp,ilev) += m_elements.buffers.vtens(ie,1,igp,jgp,ilev);

    Scalar heating = m_elements.buffers.vtens(ie,0,igp,jgp,ilev)*m_elements.m_v(ie,m_data.np1,0,igp,jgp,ilev)
                   + m_elements.buffers.vtens(ie,1,igp,jgp,ilev)*m_elements.m_v(ie,m_data.np1,1,igp,jgp,ilev);
    heating /= PhysicalConstants::cp;
    m_elements.m_t(ie,m_data.np1,igp,jgp,ilev) += m_elements.buffers.ttens(ie,igp,jgp,ilev);
    m_elements.m_t(ie,m_data.np1,igp,jgp,ilev) -= heating;

    m_elements.m_dp3d(ie,m_data.np1,igp,jgp,ilev) = m_elements.buffers.dptens(ie,igp,jgp,ilev);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagFetchStates&, const int idx) const
  {
    const int ie   =  idx / (NP*NP*NUM_LEV);
    const int igp  = (idx / (NP*NUM_LEV)) % NP;
    const int jgp  = (idx / NUM_LEV) % NP;
    const int ilev =  idx % NUM_LEV;

    // Copy states' timelevel np1 into buffers
    m_elements.buffers.dptens(ie,  igp,jgp,ilev) = m_elements.m_dp3d(ie,m_data.np1,  igp,jgp,ilev);
    m_elements.buffers.ttens (ie,  igp,jgp,ilev) = m_elements.m_t   (ie,m_data.np1,  igp,jgp,ilev);
    m_elements.buffers.vtens (ie,0,igp,jgp,ilev) = m_elements.m_v   (ie,m_data.np1,0,igp,jgp,ilev);
    m_elements.buffers.vtens (ie,1,igp,jgp,ilev) = m_elements.m_v   (ie,m_data.np1,1,igp,jgp,ilev);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagApplyInvMass&, const int idx) const
  {
    const int ie   =  idx / (NP*NP*NUM_LEV);
    const int igp  = (idx / (NP*NUM_LEV)) % NP;
    const int jgp  = (idx / NUM_LEV) % NP;
    const int ilev =  idx % NUM_LEV;

    // Apply inverse mass matrix
    m_elements.buffers.dptens(ie,  igp,jgp,ilev) *= m_elements.m_rspheremp(ie,igp,jgp);
    m_elements.buffers.ttens (ie,  igp,jgp,ilev) *= m_elements.m_rspheremp(ie,igp,jgp);
    m_elements.buffers.vtens (ie,0,igp,jgp,ilev) *= m_elements.m_rspheremp(ie,igp,jgp);
    m_elements.buffers.vtens (ie,1,igp,jgp,ilev) *= m_elements.m_rspheremp(ie,igp,jgp);
  }

protected:

  Control       m_data;
  Elements      m_elements;
  Derivative    m_deriv;
};

} // namespace Homme

#endif // HOMMEXX_HYPERVISCOSITY_FUNCTOR_HPP
