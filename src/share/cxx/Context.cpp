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
#include "KernelsBuffersManager.hpp"
#include "SphereOperators.hpp"
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

KernelsBuffersManager& Context::get_kernels_buffers_manager() {
  if ( ! kernels_buffers_manager_) kernels_buffers_manager_.reset(new KernelsBuffersManager());
  return *kernels_buffers_manager_;
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

SphereOperators& Context::get_sphere_operators(int qsize) {
  if ( ! sphere_operators_) {
    if (qsize<0) {
      qsize = get_simulation_params().qsize;
    }
    Elements&   elements   = get_elements();
    Derivative& derivative = get_derivative();
    sphere_operators_.reset(new SphereOperators(elements,derivative,qsize));
  }
  return *sphere_operators_;
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
  kernels_buffers_manager_ = nullptr;
  connectivity_ = nullptr;
  buffers_managers_ = nullptr;
  simulation_params_ = nullptr;
  sphere_operators_ = nullptr;
  time_level_ = nullptr;
  vertical_remap_mgr_ = nullptr;
  caar_functor_ = nullptr;
  euler_step_functor_ = nullptr;
}

void Context::init_functors_buffers () {
  // Get all functors
  CaarFunctor& caar = get_caar_functor();
  EulerStepFunctor& esf = get_euler_step_functor();
  HyperviscosityFunctor& hvf = get_hyperviscosity_functor();
  //VerticalRemapManager& vrm = get_vertical_remap_manager();
  //SphereOperators& sph = get_sphere_operators();

  // Get the KBM
  KernelsBuffersManager& kbm = get_kernels_buffers_manager();

  // Get the buffers requests from all functors
  kbm.request_size(caar.buffers_size());
  kbm.request_size(esf.buffers_size());
  kbm.request_size(hvf.buffers_size());
  //kbm.request_size(vrm.buffers_size());
  //kbm.request_size(sph.buffers_size());

  // Allocate buffer
  kbm.allocate_buffer();

  // Tell all functors to carve their buffers from the raw buffer in KBM
  caar.init_buffers (kbm.get_raw_buffer(),kbm.buffer_size());
  esf.init_buffers(kbm.get_raw_buffer(),kbm.buffer_size());
  hvf.init_buffers(kbm.get_raw_buffer(),kbm.buffer_size());
  //vrm.init_buffers(kbm.get_raw_buffer(),kbm.buffer_size());
  //sph.init_buffers(kbm.get_raw_buffer(),kbm.buffer_size());

}

Context& Context::singleton() {
  static Context c;
  return c;
}

void Context::finalize_singleton() {
  singleton().clear();
}

} // namespace Homme
