#include "Context.hpp"
#include "Control.hpp"
#include "SimulationParams.hpp"
#include "TimeLevel.hpp"

#include "VerticalRemapManager.hpp"
#include "ErrorDefs.hpp"

namespace Homme
{

void vertical_remap(const Real dt)
{
  // Get control and simulation params
  Control& data = Context::singleton().get_control();
  VerticalRemapManager& vrm = Context::singleton().get_vertical_remap_manager();
  TimeLevel& tl = Context::singleton().get_time_level();
  vrm.run_remap(tl.np1, tl.np1_qdp, dt);
}

} // namespace Homme
