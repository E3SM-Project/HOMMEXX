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
  constexpr const int np2 = np*np;
  for (int i=0; i<np2; ++i)
  {
    // Note: in homme, the pseudo-spectral differentiation matrix is stored 'transposed':
    //       instead of D_ij = d/dx(phi_j)(x_i) it is D_ij = d/dx(phi_i)(x_j)
    //       Since here we want to restore the 'classical' expression of D,
    //       and fortran stores matrices 'transposed' compared to C, we simply copy
    //       the array as it is to balance the two effects
    const int igp = i / np;
    const int jgp = i % np;

    get_derivative_c()->Dvv[igp][jgp]       = Dvv[i];
  }

  for (int i=0; i<np2; ++i)
  {
    const int igp = i % np;
    const int jgp = i / np;

    get_derivative_c()->Dvv_diag[igp][jgp]  = Dvv_diag[i];
    get_derivative_c()->Dvv_twt[igp][jgp]   = Dvv_twt[i];
    get_derivative_c()->Mvv_twt[igp][jgp]   = Mvv_twt[i];
    get_derivative_c()->legdg[igp][jgp]     = legdg[i];
  }

  constexpr const int npnc = np*nc;
  for (int i=0; i<npnc; ++i)
  {
    const int igp = i % np;
    const int jcp = i / np;

    get_derivative_c()->Mfvm[igp][jcp] = Mfvm[i];
    get_derivative_c()->Cfvm[igp][jcp] = Cfvm[i];

  }

  for (int igp=0; igp<np; ++igp)
  {
    get_derivative_c()->Mfvm[igp][nc] = Mfvm[nc*np+igp];
  }
}

} // extern "C"

} // Namespace Homme
