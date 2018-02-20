#include "EulerStepFunctorImpl.hpp"

namespace Homme {

EulerStepFunctor
::EulerStepFunctor () {
  p_ = std::make_shared<EulerStepFunctorImpl>();
}

void EulerStepFunctor::reset (const SimulationParams& params) {
  p_->reset(params);
}

size_t EulerStepFunctor::buffers_size () const
{
  assert (p_);
  return p_->buffers_size();
}

void EulerStepFunctor::init_buffers (Real* raw_buffer, const size_t buffer_size) {
  assert (p_);
  p_->init_buffers(raw_buffer, buffer_size);
}

void EulerStepFunctor::init_boundary_exchanges () {
  p_->init_boundary_exchanges();
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
