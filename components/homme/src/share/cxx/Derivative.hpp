#ifndef HOMMEXX_DERIVATIVE_HPP
#define HOMMEXX_DERIVATIVE_HPP

#include "Types.hpp"

#include <random>

namespace Homme {

class Derivative {
public:
  Derivative();

  void init(CF90Ptr &dvv);

  void random_init(std::mt19937_64 &engine);

  void dvv(Real *dvv);

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> get_dvv() const { return m_dvv_exec; }

private:
  ExecViewManaged<Real[NP][NP]> m_dvv_exec;
};

Derivative &get_derivative();

} // namespace Homme

#endif // HOMMEXX_DERIVATIVE_HPP
