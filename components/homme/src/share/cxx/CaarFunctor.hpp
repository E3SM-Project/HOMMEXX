#ifndef HOMMEXX_CAAR_FUNCTOR_HPP
#define HOMMEXX_CAAR_FUNCTOR_HPP

#include "Elements.hpp"
#include "Derivative.hpp"
#include "HybridVCoord.hpp"
#include "Types.hpp"
#include <memory>

namespace Homme {

struct CaarFunctorImpl;

class CaarFunctor {
public:

  CaarFunctor();

  CaarFunctor(const Elements& elements, const Derivative& derivative, const HybridVCoord& hvcoord, const int rsplit);

  void set_n0_qdp (const int n0_qdp);

  void set_rk_stage_data (const int nm1, const int n0,   const int np1,
                          const Real dt, const Real eta_ave_w,
                          const bool compute_diagnostics);

  void run ();

  void run (const int nm1, const int n0,   const int np1,
            const Real dt, const Real eta_ave_w,
            const bool compute_diagnostics);
private:

  std::unique_ptr<CaarFunctorImpl>  m_caar_impl;

  // Setup the policies
  Kokks::TeamPolicy<ExecSpace>      m_policy;

};

} // Namespace Homme

#endif // HOMMEXX_CAAR_FUNCTOR_HPP
