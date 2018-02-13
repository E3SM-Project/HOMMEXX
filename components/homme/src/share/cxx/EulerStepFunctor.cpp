#include "EulerStepFunctorImpl.hpp"

namespace Homme {

EulerStepFunctor
::EulerStepFunctor () {
  p_ = std::make_shared<EulerStepFunctorImpl>();
}

void EulerStepFunctor::reset (const SimulationParams& params) {
  p_->reset(params);
}

void EulerStepFunctor::precompute_divdp () {
  p_->precompute_divdp();
}

void EulerStepFunctor
::euler_step (const int np1_qdp, const int n0_qdp, const Real dt,
              const Real rhs_multiplier, const DSSOption DSSopt) {
  p_->euler_step(np1_qdp, n0_qdp, dt, rhs_multiplier, DSSopt);
}

void EulerStepFunctor
::qdp_time_avg (const int n0_qdp, const int np1_qdp) {
  p_->qdp_time_avg(n0_qdp, np1_qdp);
}

}
