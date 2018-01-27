#include "CaarFunctor.hpp"
#include "Control.hpp"
#include "Context.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"
#include "mpi/ErrorDefs.hpp"
#include "mpi/BoundaryExchange.hpp"
#include "mpi/BuffersManager.hpp"



namespace Homme
{

void prim_advance_hypervis_dp_c (const int np1, const Real dt, const Real eta_ave_w)
{
  // Get simulation params
  SimulationParams& params = Context::singleton().get_simulation_params();
  assert(params.params_set);

  if (params.nu==0 && params.nu_s==0 && params.nu_p==0) {
    // Nothing to do here
    return;
  }

  // Get control
  Control& data = Context::singleton().get_control();

  if (params.hypervis_order==1) {
    if (params.nu_p>0) {
      Errors::runtime_abort("Error hypervis_order=1 not coded for nu_p>0.\n",
                             Errors::unsupported_option);
    }

    for (int icycle=0; icycle<params.hypervis_subcycle; ++icycle) {
    }
  } else if (params.hypervis_order==2) {
  } else {
    Errors::runtime_abort("Error prim_advance_hypervis_dp_c: unsupported hypervis_order.\n",
                           Errors::unsupported_option);
  }
}

} // namespace Homme
