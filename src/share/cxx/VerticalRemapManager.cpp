#include "VerticalRemapManager.hpp"
#include "SimulationParams.hpp"
#include "Control.hpp"
#include "Context.hpp"
#include "HybridVCoord.hpp"
#include "HommexxEnums.hpp"
#include "RemapFunctor.hpp"

namespace Homme {

struct VerticalRemapManager::Impl {
  Impl (const Control& c, const Elements& e, const HybridVCoord& h, RemapAlg alg) {
    if (alg == RemapAlg::PPM_FIXED) {
      if (c.rsplit != 0) {
        remapper = std::make_shared<RemapFunctor<true, PpmVertRemap, PpmFixed> >(c, e, h);
      } else {
        remapper = std::make_shared<RemapFunctor<false, PpmVertRemap, PpmFixed> >(c, e, h);
      }
    } else if (alg == RemapAlg::PPM_MIRRORED) {
      if (c.rsplit != 0) {
        remapper = std::make_shared<RemapFunctor<true, PpmVertRemap, PpmMirrored> >(c, e, h);
      } else {
        remapper = std::make_shared<RemapFunctor<false, PpmVertRemap, PpmMirrored> >(c, e, h);
      }
    } else {
      Errors::runtime_abort("Error in VerticalRemapManager: unknown remap algorithm.\n",
                            Errors::err_unknown_option);
    }
  }

  std::shared_ptr<Remapper> remapper;
};

VerticalRemapManager
::VerticalRemapManager () {
  const auto& c = Context::singleton().get_control();
  const auto& h = Context::singleton().get_hvcoord();
  const auto& p = Context::singleton().get_simulation_params();
  const auto& e = Context::singleton().get_elements();
  assert(p.params_set);
  p_.reset(new Impl(c, e, h, p.remap_alg));
}

void VerticalRemapManager::run_remap (int np1, int n0_qdp, double dt) const {
  assert(p_);
  assert(p_->remapper);
  p_->remapper->run_remap(np1, n0_qdp, dt);
}

}
