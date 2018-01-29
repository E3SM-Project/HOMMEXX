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

  struct TagLaplaceSimple_T_Contra_V {};
  struct TagLaplaceSimple_T_DP3D_Contra_V {};
  struct TagLaplaceTensor_T_Contra_V {};
  struct TagLaplaceTensor_T_DP3D_Contra_V {};
  struct TagLaplaceTensor_T_Cartesian_V {};
  struct TagLaplaceTensor_T_DP3D_Cartesian_V {};
  struct TagApplyInvMass_T_V {};
  struct TagApplyInvMass_T_DP3D_V {};

  HyperviscosityFunctor (const Control& data, const Elements& elements, const Derivative& deriv);

  void compute_t_v_laplace      (const int m_itl, const bool var_coeff, const Real nu_ratio, const Real hypervis_scaling);
  void compute_t_v_dp3d_laplace (const int m_itl, const bool var_coeff, const Real nu_ratio, const Real hypervis_scaling);

  ExecViewManaged<Scalar *    [NP][NP][NUM_LEV]>  m_laplace_t;
  ExecViewManaged<Scalar *    [NP][NP][NUM_LEV]>  m_laplace_dp3d;
  ExecViewManaged<Scalar * [2][NP][NP][NUM_LEV]>  m_laplace_v;

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagLaplaceSimple_T_Contra_V&, const TeamMember& team) const {
    KernelVariables kv(team);

    // Laplacian of temperature
    laplace_simple(kv, m_elements.m_dinv, m_elements.m_spheremp, m_deriv.get_dvv(),
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.m_t, kv.ie, m_itl),
                   m_elements.buffers.sphere_vector_buf,
                   Homme::subview(m_laplace_t,kv.ie));
    // Laplacian of velocity
    vlaplace_sphere_wk_contra(kv,m_elements.m_d,m_elements.m_dinv,m_elements.m_mp,m_elements.m_spheremp,
                              m_elements.m_metinv, m_elements.m_metdet, m_deriv.get_dvv(), m_nu_ratio,
                              Homme::subview(m_elements.buffers.divergence_temp,kv.ie),
                              Homme::subview(m_elements.buffers.vorticity_temp,kv.ie),
                              Homme::subview(m_elements.buffers.grad_buf,kv.ie),
                              Homme::subview(m_elements.buffers.curl_buf,kv.ie),
                              m_elements.buffers.sphere_vector_buf,
                              Homme::subview(m_elements.m_v,kv.ie,m_itl),
                              Homme::subview(m_laplace_v,kv.ie));

  }
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagLaplaceTensor_T_Contra_V&, const TeamMember& team) const {
    KernelVariables kv(team);

    // Laplacian of temperature
    laplace_tensor(kv, m_elements.m_dinv, m_elements.m_spheremp, m_deriv.get_dvv(),
                   m_elements.m_tensorVisc,
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.m_t, kv.ie, m_itl),
                   m_elements.buffers.sphere_vector_buf,
                   Homme::subview(m_laplace_t,kv.ie));

    // Laplacian of velocity
    vlaplace_sphere_wk_contra(kv,m_elements.m_d,m_elements.m_dinv,m_elements.m_mp,m_elements.m_spheremp,
                              m_elements.m_metinv, m_elements.m_metdet, m_deriv.get_dvv(), m_nu_ratio,
                              Homme::subview(m_elements.buffers.divergence_temp,kv.ie),
                              Homme::subview(m_elements.buffers.vorticity_temp,kv.ie),
                              Homme::subview(m_elements.buffers.grad_buf,kv.ie),
                              Homme::subview(m_elements.buffers.curl_buf,kv.ie),
                              m_elements.buffers.sphere_vector_buf,
                              Homme::subview(m_elements.m_v,kv.ie,m_itl),
                              Homme::subview(m_laplace_v,kv.ie));
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagLaplaceTensor_T_Cartesian_V&, const TeamMember& team) const {
    KernelVariables kv(team);

    // Laplacian of temperature
    laplace_tensor(kv, m_elements.m_dinv, m_elements.m_spheremp, m_deriv.get_dvv(),
                   m_elements.m_tensorVisc,
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.m_t, kv.ie, m_itl),
                   m_elements.buffers.sphere_vector_buf,
                   Homme::subview(m_laplace_t,kv.ie));

    // Laplacian of velocity
    vlaplace_sphere_wk_cartesian_reduced(kv,m_elements.m_dinv,m_elements.m_spheremp,
                              m_elements.m_tensorVisc,m_elements.m_vec_sph2cart,m_deriv.get_dvv(),
                              Homme::subview(m_elements.buffers.grad_buf,kv.ie),
                              Homme::subview(m_elements.buffers.lapl_buf_1,kv.ie),
                              Homme::subview(m_elements.buffers.lapl_buf_2,kv.ie),
                              Homme::subview(m_elements.buffers.lapl_buf_3,kv.ie),
                              m_elements.buffers.sphere_vector_buf,
                              Homme::subview(m_elements.m_v,kv.ie,m_itl),
                              Homme::subview(m_laplace_v,kv.ie));

  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagLaplaceSimple_T_DP3D_Contra_V&, const TeamMember& team) const {
    KernelVariables kv(team);

    // Laplacian of temperature
    laplace_simple(kv, m_elements.m_dinv, m_elements.m_spheremp, m_deriv.get_dvv(),
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.m_t, kv.ie, m_itl),
                   m_elements.buffers.sphere_vector_buf,
                   Homme::subview(m_laplace_t,kv.ie));
    // Laplacian of pressure
    laplace_simple(kv, m_elements.m_dinv, m_elements.m_spheremp, m_deriv.get_dvv(),
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.m_dp3d, kv.ie, m_itl),
                   m_elements.buffers.sphere_vector_buf,
                   Homme::subview(m_laplace_dp3d,kv.ie));
    // Laplacian of velocity
    vlaplace_sphere_wk_contra(kv,m_elements.m_d,m_elements.m_dinv,m_elements.m_mp,m_elements.m_spheremp,
                              m_elements.m_metinv, m_elements.m_metdet, m_deriv.get_dvv(), m_nu_ratio,
                              Homme::subview(m_elements.buffers.divergence_temp,kv.ie),
                              Homme::subview(m_elements.buffers.vorticity_temp,kv.ie),
                              Homme::subview(m_elements.buffers.grad_buf,kv.ie),
                              Homme::subview(m_elements.buffers.curl_buf,kv.ie),
                              m_elements.buffers.sphere_vector_buf,
                              Homme::subview(m_elements.m_v,kv.ie,m_itl),
                              Homme::subview(m_laplace_v,kv.ie));

  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagLaplaceTensor_T_DP3D_Contra_V&, const TeamMember& team) const {
    KernelVariables kv(team);
    // Laplacian of temperature
    laplace_tensor(kv, m_elements.m_dinv, m_elements.m_spheremp, m_deriv.get_dvv(),
                   m_elements.m_tensorVisc,
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.m_t, kv.ie, m_itl),
                   m_elements.buffers.sphere_vector_buf,
                   Homme::subview(m_laplace_t,kv.ie));
    // Laplacian of pressure
    laplace_tensor(kv, m_elements.m_dinv, m_elements.m_spheremp, m_deriv.get_dvv(),
                   m_elements.m_tensorVisc,
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.m_dp3d, kv.ie, m_itl),
                   m_elements.buffers.sphere_vector_buf,
                   Homme::subview(m_laplace_dp3d,kv.ie));

    // Laplacian of velocity
    vlaplace_sphere_wk_contra(kv,m_elements.m_d,m_elements.m_dinv,m_elements.m_mp,m_elements.m_spheremp,
                              m_elements.m_metinv, m_elements.m_metdet, m_deriv.get_dvv(), m_nu_ratio,
                              Homme::subview(m_elements.buffers.divergence_temp,kv.ie),
                              Homme::subview(m_elements.buffers.vorticity_temp,kv.ie),
                              Homme::subview(m_elements.buffers.grad_buf,kv.ie),
                              Homme::subview(m_elements.buffers.curl_buf,kv.ie),
                              m_elements.buffers.sphere_vector_buf,
                              Homme::subview(m_elements.m_v,kv.ie,m_itl),
                              Homme::subview(m_laplace_v,kv.ie));
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagLaplaceTensor_T_DP3D_Cartesian_V&, const TeamMember& team) const {
    KernelVariables kv(team);

    // Laplacian of temperature
    laplace_tensor(kv, m_elements.m_dinv, m_elements.m_spheremp, m_deriv.get_dvv(),
                   m_elements.m_tensorVisc,
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.m_t, kv.ie, m_itl),
                   m_elements.buffers.sphere_vector_buf,
                   Homme::subview(m_laplace_t,kv.ie));
    // Laplacian of pressure
    laplace_tensor(kv, m_elements.m_dinv, m_elements.m_spheremp, m_deriv.get_dvv(),
                   m_elements.m_tensorVisc,
                   Homme::subview(m_elements.buffers.grad_buf, kv.ie),
                   Homme::subview(m_elements.m_dp3d, kv.ie, m_itl),
                   m_elements.buffers.sphere_vector_buf,
                   Homme::subview(m_laplace_dp3d,kv.ie));

    // Laplacian of velocity
    vlaplace_sphere_wk_cartesian_reduced(kv,m_elements.m_dinv,m_elements.m_spheremp,
                              m_elements.m_tensorVisc,m_elements.m_vec_sph2cart,m_deriv.get_dvv(),
                              Homme::subview(m_elements.buffers.grad_buf,kv.ie),
                              Homme::subview(m_elements.buffers.lapl_buf_1,kv.ie),
                              Homme::subview(m_elements.buffers.lapl_buf_2,kv.ie),
                              Homme::subview(m_elements.buffers.lapl_buf_3,kv.ie),
                              m_elements.buffers.sphere_vector_buf,
                              Homme::subview(m_elements.m_v,kv.ie,m_itl),
                              Homme::subview(m_laplace_v,kv.ie));

  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagApplyInvMass_T_V&, const int idx) const
  {
    // Apply inverse mass matrix
    const int ie   =  idx / (NP*NP*NUM_LEV);
    const int igp  = (idx / (NP*NUM_LEV)) % NP;
    const int jgp  = (idx / NUM_LEV) % NP;
    const int ilev =  idx % NUM_LEV;

    m_elements.m_t(ie,m_itl,igp,jgp,ilev)   *= m_elements.m_rspheremp(ie,igp,jgp);
    m_elements.m_v(ie,m_itl,0,igp,jgp,ilev) *= m_elements.m_rspheremp(ie,igp,jgp);
    m_elements.m_v(ie,m_itl,1,igp,jgp,ilev) *= m_elements.m_rspheremp(ie,igp,jgp);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagApplyInvMass_T_DP3D_V&, const int idx) const
  {
    // Apply inverse mass matrix
    const int ie   =  idx / (NP*NP*NUM_LEV);
    const int igp  = (idx / (NP*NUM_LEV)) % NP;
    const int jgp  = (idx / NUM_LEV) % NP;
    const int ilev =  idx % NUM_LEV;

    m_elements.m_dp3d(ie,m_itl,igp,jgp,ilev) *= m_elements.m_rspheremp(ie,igp,jgp);
    m_elements.m_t(ie,m_itl,igp,jgp,ilev)    *= m_elements.m_rspheremp(ie,igp,jgp);
    m_elements.m_v(ie,m_itl,0,igp,jgp,ilev)  *= m_elements.m_rspheremp(ie,igp,jgp);
    m_elements.m_v(ie,m_itl,1,igp,jgp,ilev)  *= m_elements.m_rspheremp(ie,igp,jgp);
  }

protected:

  int           m_itl;
  Real          m_nu_ratio;

  Control       m_data;
  Elements      m_elements;
  Derivative    m_deriv;
};

} // namespace Homme

#endif // HOMMEXX_HYPERVISCOSITY_FUNCTOR_HPP
