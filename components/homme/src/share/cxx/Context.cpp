#include "Context.hpp"

#include "Control.hpp"
#include "Elements.hpp"
#include "Derivative.hpp"

namespace Homme {

Context::Context() {}

Context::~Context() {}

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

void Context::clear() {
  control_ = nullptr;
  elements_ = nullptr;
  derivative_ = nullptr;
}

Context& Context::singleton() {
  static Context c;

  Context::has_singleton = true;
  return c;
}

void Context::finalize_singleton() {
  if (Context::has_singleton) {
    singleton().clear();
  }
}

bool Context::has_singleton = false;

} // namespace Homme
