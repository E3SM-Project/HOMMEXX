#include "Context.hpp"

#include "Comm.hpp"
#include "Control.hpp"
#include "Elements.hpp"
#include "Derivative.hpp"
#include "BuffersManager.hpp"
#include "Connectivity.hpp"
#include "BoundaryExchange.hpp"
#include "SimulationParams.hpp"

namespace Homme {

Context::Context() {}

Context::~Context() {}

Comm& Context::get_comm() {
  //if ( ! control_) control_ = std::make_shared<Control>();
  if ( ! comm_) {
    comm_.reset(new Comm());
    comm_->init();
  }
  return *comm_;
}

Control& Context::get_control() {
  //if ( ! control_) control_ = std::make_shared<Control>();
  if ( ! control_) control_.reset(new Control());
  return *control_;
}

Elements& Context::get_elements() {
  //if ( ! elements_) elements_ = std::make_shared<Elements>();
  if ( ! elements_) elements_.reset(new Elements());
  return *elements_;
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

void Context::clear() {
  comm_ = nullptr;
  control_ = nullptr;
  elements_ = nullptr;
  derivative_ = nullptr;
  connectivity_ = nullptr;
  boundary_exchanges_ = nullptr;
  buffers_managers_ = nullptr;
  simulation_params_ = nullptr;
}

Context& Context::singleton() {
  static Context c;
  return c;
}

void Context::finalize_singleton() {
  singleton().clear();
}

} // namespace Homme
