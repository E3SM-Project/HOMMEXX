#include "CaarFunctor.hpp"
#include "CaarFunctorImpl.hpp"

#include "profiling.hpp"

#include <assert.h>
#include <type_traits>


namespace Homme {

CaarFunctor::CaarFunctor()
{
  Elements&     elements  = Context::singleton().get_elements();
  Derivative&   derivtive = Context::singleton().get_derivative();
  HybridVCoord& hvcoord   = Context::singleton().get_hvcoord();
  const int rsplit = Context::singleton().get_simulation_params().rsplit;

  // Build functor impl
  m_caar_impl.reset(new CaarFunctorImpl(elements,derivtive,hvcoord,rsplit));

  // Build policy once for all
  m_policy = Homme::get_default_team_policy<ExecSpace,CaarFunctor::TagPreExchange>(e.num_elems());
}

CaarFunctor::CaarFunctor(const Elements& elements, const Derivative& derivative, const HybridVCoord& hvcoord, const int rsplit)
{
  // Build functor impl
  m_caar_impl.reset(new CaarFunctorImpl(elements,derivtive,hvcoord,rsplit));

  // Build policy once for all
  m_policy = Homme::get_default_team_policy<ExecSpace>(elements.num_elems());
}

void CaarFunctor::set_n0_qdp (const int n0_qdp)
{
  // Sanity check (should NEVER happen)
  assert (m_caar_impl);

  // Forward input to impl
  m_caar_impl->set_n0_qdp(n0_qdp);
}

void CaarFunctor::set_rk_stage_data (const int nm1, const int n0,   const int np1,
                                     const Real dt, const Real eta_ave_w,
                                     const bool compute_diagnostics)
{
  // Sanity check (should NEVER happen)
  assert (m_caar_impl);

  // Forward inputs to impl
  m_caar_impl->set_rk_stage_data(nm1,n0,np1,dt,eta_ave_w,compute_diagnostics);
}

void CaarFunctor::run ()
{
  // Sanity check (should NEVER happen)
  assert (m_caar_impl);

  // Run functor
  profiling_resume();
  Kokkos::parallel_for("caar loop pre-boundary exchange", m_policy, *m_caar_impl);
  ExecSpace::fence();
  profiling_pause();
}

void CaarFunctor::run (const int nm1, const int n0,   const int np1,
          const Real dt, const Real eta_ave_w,
          const bool compute_diagnostics)
{
  // Sanity check (should NEVER happen)
  assert (m_caar_impl);

  // Forward inputs to impl
  m_caar_impl->set_rk_stage_data(nm1,n0,np1,dt,eta_ave_w,compute_diagnostics);

  // Run functor
  profiling_resume();
  Kokkos::parallel_for("caar loop pre-boundary exchange", m_policy, *m_caar_impl);
  ExecSpace::fence();
  profiling_pause();
}

} // Namespace Homme
