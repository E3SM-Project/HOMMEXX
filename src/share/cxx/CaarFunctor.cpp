#include "CaarFunctor.hpp"
#include "CaarFunctorImpl.hpp"

#include <assert.h>

namespace Homme {

CaarFunctor::CaarFunctor() {
  if (!m_caar_impl) {
    const SimulationParams& params = Context::singleton().get_simulation_params();
    const Elements& elements = Context::singleton().get_elements();
    const Derivative& deriv = Context::singleton().get_derivative();
    const HybridVCoord& hvcoord = Context::singleton().get_hvcoord();

    m_caar_impl = std::make_shared(params.rsplit,element,deriv);
  }
}

void CaarFunctor::set_rk_stage_data (const int nm1, const int n0, const int np1, const int n0_qdp,
                                     const Real dt, const Real eta_ave_w, const bool compute_diagonstics)
{
  assert (m_caar_impl);
  m_caar_impl->set_rk_stage_data(nm1,n0,np1,n0_qdp,dt,eta_ave_w,compute_diagnostics);
}

void CaarFunctor::run_pre_exchange (Kokkos::TeamPolicy<ExecSpace,TagPreExchange>& policy)
{
  assert (m_caar_impl);
  Kokkos::parallel_for("caar loop pre-boundary exchange", policy, *m_caar_impl);
}

} // Namespace Homme
