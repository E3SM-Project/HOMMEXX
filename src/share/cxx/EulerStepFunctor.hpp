#ifndef HOMMEXX_EULER_STEP_FUNCTOR_HPP
#define HOMMEXX_EULER_STEP_FUNCTOR_HPP

#include <memory>

#include "Types.hpp"
#include "SimulationParams.hpp"

namespace Homme {

class EulerStepFunctorImpl;

class EulerStepFunctor {
  std::shared_ptr<EulerStepFunctorImpl> p_;

public:
  EulerStepFunctor(const SimulationParams& params);

  void precompute_divdp();

  void qdp_time_avg(const int n0_qdp, const int np1_qdp);

  void euler_step(const int np1_qdp, const int n0_qdp, const Real dt,
                  const Real rhs_multiplier, const DSSOption DSSopt);
};

} // namespace Homme

#endif // HOMMEXX_EULER_STEP_FUNCTOR_HPP
