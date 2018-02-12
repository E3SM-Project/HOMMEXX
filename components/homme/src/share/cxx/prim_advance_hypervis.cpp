#include "Context.hpp"
#include "Derivative.hpp"
#include "Elements.hpp"
#include "SimulationParams.hpp"
#include "HyperviscosityFunctor.hpp"

#include "Types.hpp"

namespace Homme
{

void advance_hypervis_dp (const int np1, const Real dt, const Real eta_ave_w)
{
  // Get simulation parameters, control, elements and derivative
  SimulationParams& params   = Context::singleton().get_simulation_params();
  Elements&         elements = Context::singleton().get_elements();
  Derivative&       deriv    = Context::singleton().get_derivative();

  // Create and run the HVF
  HyperviscosityFunctor functor(params,elements,deriv);
  functor.run(np1,dt,eta_ave_w);
}

} // namespace Homme
