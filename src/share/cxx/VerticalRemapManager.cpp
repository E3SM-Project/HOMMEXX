#include "VerticalRemapManager.hpp"
#include "SimulationParams.hpp"
#include "Context.hpp"
#include "HybridVCoord.hpp"
#include "HommexxEnums.hpp"
#include "RemapFunctor.hpp"

namespace Homme {

struct VerticalRemapManager::Impl {
  Impl (const SimulationParams& params, const Elements& e, const HybridVCoord& h) {
    if (params.remap_alg == RemapAlg::PPM_FIXED) {
      if (params.rsplit != 0) {
        remapper = std::make_shared<RemapFunctor<true, PpmVertRemap, PpmFixed> >(params.qsize, e, h);
      } else {
        remapper = std::make_shared<RemapFunctor<false, PpmVertRemap, PpmFixed> >(params.qsize, e, h);
      }
    } else if (params.remap_alg == RemapAlg::PPM_MIRRORED) {
      if (params.rsplit != 0) {
        remapper = std::make_shared<RemapFunctor<true, PpmVertRemap, PpmMirrored> >(params.qsize, e, h);
      } else {
        remapper = std::make_shared<RemapFunctor<false, PpmVertRemap, PpmMirrored> >(params.qsize, e, h);
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
  const auto& h = Context::singleton().get_hvcoord();
  const auto& p = Context::singleton().get_simulation_params();
  const auto& e = Context::singleton().get_elements();
  assert(p.params_set);
  p_.reset(new Impl(p, e, h));
}

void VerticalRemapManager::run_remap (int np1, int n0_qdp, double dt) const {
  assert(p_);
  assert(p_->remapper);
  p_->remapper->run_remap(np1, n0_qdp, dt);
}

}
