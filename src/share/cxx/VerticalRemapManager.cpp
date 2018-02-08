#include "VerticalRemapManager.hpp"
#include "Control.hpp"
#include "HommexxEnums.hpp"

namespace Homme {

struct VerticalRemapManager::Impl {
  Impl (Control& c, RemapAlg alg) {
  }
};

VerticalRemapManager
::VerticalRemapManager () {
  //p_ = std::unique_ptr<Impl>(new Impl(c, alg));
}

void VerticalRemapManager::run_remap () const {
}

}
