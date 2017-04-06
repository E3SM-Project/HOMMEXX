#ifndef CAAR_HELPERS_HPP
#define CAAR_HELPERS_HPP

#include "Types.hpp"

namespace Homme
{

extern "C"
{

void caar_compute_pressure_c (const int& nets, const int& nete,
                              const int& nelemd, const int& n0, const Real& hyai_ps0,
                              CRPtr& p_ptr, CRPtr& dp_ptr);

} // extern "C"

} // namespace Homme

#endif // CAAR_HELPERS_HPP
