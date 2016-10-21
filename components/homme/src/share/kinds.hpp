
#ifndef _KINDS_HPP_
#define _KINDS_HPP_

namespace Homme {

using real = double;

#ifdef CAM
using long_dbl = double;
#else
#if HOMME_QUAD_PREC
using long_dbl = long double;
#else
using long_dbl = double;
#endif

}  // namespace Homme

#endif
