#include "Context.hpp"

#include "BoundaryExchange.hpp"
#include "BuffersManager.hpp"
#include "CaarFunctor.hpp"
#include "Comm.hpp"
#include "Connectivity.hpp"
#include "Derivative.hpp"
#include "Elements.hpp"
#include "HybridVCoord.hpp"
#include "HyperviscosityFunctor.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "VerticalRemapManager.hpp"
#include "EulerStepFunctor.hpp"

namespace Homme {

Context::Context() {}

Context::~Context() {}

CaarFunctor& Context::get_caar_functor() {
  if ( ! caar_functor_) {
    caar_functor_.reset(new CaarFunctor());
  }
  return *caar_functor_;
}

Comm& Context::get_comm() {
  if ( ! comm_) {
    comm_.reset(new Comm());
    comm_->init();
  }
  return *comm_;
}

Elements& Context::get_elements() {
  //if ( ! elements_) elements_ = std::make_shared<Elements>();
  if ( ! elements_) elements_.reset(new Elements());
  return *elements_;
}

HybridVCoord& Context::get_hvcoord() {
  if ( ! hvcoord_) hvcoord_.reset(new HybridVCoord());
  return *hvcoord_;
}

HyperviscosityFunctor& Context::get_hyperviscosity_functor() {
  if ( ! hyperviscosity_functor_) {
    Elements& e = get_elements();
    Derivative& d = get_derivative();
    SimulationParams& p = get_simulation_params();
    hyperviscosity_functor_.reset(new HyperviscosityFunctor(p,e,d));
  }
  return *hyperviscosity_functor_;
}

Derivative& Context::get_derivative() {
  //if ( ! derivative_) derivative_ = std::make_shared<Derivative>();
  if ( ! derivative_) derivative_.reset(new Derivative());
  return *derivative_;
}

SimulationParams& Context::get_simulation_params() {
  if ( ! simulation_params_) simulation_params_.reset(new SimulationParams());
  return *simulation_params_;
}

TimeLevel& Context::get_time_level() {
  if ( ! time_level_) time_level_.reset(new TimeLevel());
  return *time_level_;
}

VerticalRemapManager& Context::get_vertical_remap_manager() {
  if ( ! vertical_remap_mgr_) vertical_remap_mgr_.reset(new VerticalRemapManager());
  return *vertical_remap_mgr_;
}

std::shared_ptr<BuffersManager> Context::get_buffers_manager(short int exchange_type) {
  if ( ! buffers_managers_) {
    buffers_managers_.reset(new BMMap());
  }

  if (!(*buffers_managers_)[exchange_type]) {
    (*buffers_managers_)[exchange_type] = std::make_shared<BuffersManager>(get_connectivity());
  }
  return (*buffers_managers_)[exchange_type];
}

std::shared_ptr<Connectivity> Context::get_connectivity() {
  if ( ! connectivity_) connectivity_.reset(new Connectivity());
  return connectivity_;
}

Context::BEMap& Context::get_boundary_exchanges() {
  if ( ! boundary_exchanges_) boundary_exchanges_.reset(new BEMap());

  return *boundary_exchanges_;
}

std::shared_ptr<BoundaryExchange> Context::get_boundary_exchange(const std::string& name) {
  if ( ! boundary_exchanges_) boundary_exchanges_.reset(new BEMap());

  // Todo: should we accept a bool param 'must_already_exist'
  //       to make sure we are not creating a new BE?
  if (!(*boundary_exchanges_)[name]) {
    (*boundary_exchanges_)[name] = std::make_shared<BoundaryExchange>();
  }
  return (*boundary_exchanges_)[name];
}

EulerStepFunctor& Context::get_euler_step_functor() {
  if ( ! euler_step_functor_) euler_step_functor_.reset(new EulerStepFunctor());
  return *euler_step_functor_;
}

void Context::clear() {
  comm_ = nullptr;
  elements_ = nullptr;
  derivative_ = nullptr;
  hvcoord_ = nullptr;
  hyperviscosity_functor_ = nullptr;
  connectivity_ = nullptr;
  boundary_exchanges_ = nullptr;
  buffers_managers_ = nullptr;
  simulation_params_ = nullptr;
  time_level_ = nullptr;
  vertical_remap_mgr_ = nullptr;
  caar_functor_ = nullptr;
  euler_step_functor_ = nullptr;
}

Context& Context::singleton() {
  static Context c;
  return c;
}

void Context::finalize_singleton() {
  singleton().clear();
}

} // namespace Homme
