#include "Context.hpp"

#include "Control.hpp"
#include "Elements.hpp"
#include "Derivative.hpp"

namespace Homme {

Context::Context() {}

Context::~Context() {}

Control& Context::get_control() {
  if ( ! control_) control_ = std::make_shared<Control>();
  return *control_;
}

Elements& Context::get_elements() {
  if ( ! elements_) elements_ = std::make_shared<Elements>();
  return *elements_;
}

Derivative& Context::get_derivative() {
  if ( ! derivative_) derivative_ = std::make_shared<Derivative>();
  return *derivative_;
}

void Context::clear() {
  control_ = nullptr;
  elements_ = nullptr;
  derivative_ = nullptr;
}

Context& Context::singleton() {
  static Context c;
  return c;
}

void Context::finalize_singleton() {
  singleton().clear();
}

}
