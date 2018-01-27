#include "Context.hpp"
#include "Control.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"

#include "RemapFunctor.hpp"
#include "mpi/ErrorDefs.hpp"

namespace Homme
{

template <bool rsplit, template <int, typename...> class RemapAlg,
          typename... RemapOptions>
void vertical_remap(Control &data) {
  RemapFunctor<rsplit, RemapAlg, RemapOptions...> remap(
      data, Context::singleton().get_elements());
  remap.run_remap();
}

extern "C" {

// fort_ps_v is of type Real [NUM_ELEMS][NUM_TIME_LEVELS][NP][NP]
void vertical_remap_c(const Real dt)
{
  // Get control and simulation params
  Control&          data   = Context::singleton().get_control();
  SimulationParams& params = Context::singleton().get_simulation_params();
  assert(params.params_set);

  // Get time level info and set them in the control
  TimeLevel& tl = Context::singleton().get_time_level();
  data.np1 = tl.np1;
  data.qn0 = tl.np1_qdp;
  data.dt  = dt;
  const auto rsplit = data.rsplit;
  if (params.remap_alg == RemapAlg::PPM_FIXED) {
    if (rsplit != 0) {
      vertical_remap<true, PpmVertRemap, PpmFixed>(data);
    } else {
      vertical_remap<false, PpmVertRemap, PpmFixed>(data);
    }
  } else if (params.remap_alg == RemapAlg::PPM_MIRRORED) {
    if (rsplit != 0) {
      vertical_remap<true, PpmVertRemap, PpmMirrored>(data);
    } else {
      vertical_remap<false, PpmVertRemap, PpmMirrored>(data);
    }
  } else {
    Errors::runtime_abort("Error in vertical_remap_c: unknown remap algorithm.\n",
                           Errors::unknown_option);
  }
}

} // extern "C"

} // namespace Homme
