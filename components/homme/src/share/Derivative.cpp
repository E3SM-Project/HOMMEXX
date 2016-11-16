#include <Derivative.hpp>

namespace Homme
{

extern "C"
{

Derivative* get_derivative_c ()
{
  static Derivative deriv;
  return &deriv;
}

// Initializing the static members of the Derivative struct using the values computed in f90
void init_derivative_c (real* const& Dvv,     real* const& Dvv_diag, real* const& Dvv_twt,
                        real* const& Mvv_twt, real* const& Mfvm,     real* const& Cfvm,
                        real* const& legdg)
{
  for (int igp=0; igp<np; ++igp)
  {
    for (int jgp=0; jgp<np; ++jgp)
    {
      get_derivative_c()->Dvv[igp][jgp]       = Dvv[jgp*np+igp];
      get_derivative_c()->Dvv_diag[igp][jgp]  = Dvv_diag[jgp*np+igp];
      get_derivative_c()->Dvv_twt[igp][jgp]   = Dvv_twt[jgp*np+igp];
      get_derivative_c()->legdg[igp][jgp]     = legdg[jgp*np+igp];
    }
    for (int jcp=0; jcp<nc; ++jcp)
    {
      get_derivative_c()->Mfvm[igp][jcp] = Mfvm[jcp*np+igp];
      get_derivative_c()->Cfvm[igp][jcp] = Cfvm[jcp*np+igp];
    }
    get_derivative_c()->Mfvm[igp][nc-1] = Mfvm[(nc-1)*np+igp];
  }
}

} // extern "C"

} // Namespace Homme
