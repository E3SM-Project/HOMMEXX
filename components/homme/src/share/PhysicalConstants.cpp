#include <PhysicalConstants.hpp>

namespace Homme
{

extern "C"
{

PhysicalConstants* get_physical_constants_c ()
{
  static PhysicalConstants pc;
  return &pc;
}

void init_physical_constants_c (const real& rearth,  const real& g,            const real& omega,
                                const real& Rgas,    const real& Cp,           const real& p0,
                                const real& MWDAIR,  const real& Rwater_vapor, const real& Cpwater_vapor,
                                const real& kappa,   const real& Rd_on_Rv,     const real& Cpd_on_Cpv,
                                const real& rrearth, const real& Lc,           const real& pi)
{
  get_physical_constants_c()->rearth         = rearth;
  get_physical_constants_c()->g              = g;
  get_physical_constants_c()->omega          = omega;
  get_physical_constants_c()->Rgas           = Rgas;
  get_physical_constants_c()->Cp             = Cp;
  get_physical_constants_c()->p0             = p0;
  get_physical_constants_c()->MWDAIR         = MWDAIR;
  get_physical_constants_c()->Rwater_vapor   = Rwater_vapor;
  get_physical_constants_c()->Cpwater_vapor  = Cpwater_vapor;
  get_physical_constants_c()->kappa          = kappa;
  get_physical_constants_c()->Rd_on_Rv       = Rd_on_Rv;
  get_physical_constants_c()->Cpd_on_Cpv     = Cpd_on_Cpv;
  get_physical_constants_c()->rrearth        = rrearth;
  get_physical_constants_c()->Lc             = Lc;
  get_physical_constants_c()->pi             = pi;
}

} // extern "C"

} // Namespace Homme
