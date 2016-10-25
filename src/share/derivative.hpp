
#ifndef _DERIVATIVE_HPP_
#define _DERIVATIVE_HPP_

#include <dimensions.hpp>
#include <kinds.hpp>

namespace Homme {

struct derivative_t {
  real Dvv[np * np];
  real Dvv_diag[np * np];
  real Dvv_twt[np * np];
  real Mvv_twt[np * np];
  real Mfvm[np * (nc + 1)];
  real Cfvm[np * nc];
  real legdg[np * np];
};

struct derivative_stag_t {
  real D[np * np];
  real M[np * np];
  real Dpv[np * np];
  real D_twt[np * np];
  real M_twt[np * np];
  real M_t[np * np];
};

}  // namespace Homme

#endif
