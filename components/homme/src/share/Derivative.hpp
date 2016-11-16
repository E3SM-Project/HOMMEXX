#ifndef HOMMEXX_DERIVATIVE_HPP
#define HOMMEXX_DERIVATIVE_HPP

#include <dimensions.hpp>
#include <kinds.hpp>

namespace Homme
{

extern "C"
{

struct Derivative
{
  real Dvv[np][np];
  real Dvv_diag[np][np];
  real Dvv_twt[np][np];
  real Mvv_twt[np][np];
  real Mfvm[np][nc + 1];
  real Cfvm[np][nc];
  real legdg[np][np];
};

Derivative* get_derivative_c ();

} // extern "C"

} // Namespace Homme

#endif // HOMMEXX_DERIVATIVE_HPP
