#ifndef HOMMEXX_HYPERVISCOSITY_FUNCTOR_HPP
#define HOMMEXX_HYPERVISCOSITY_FUNCTOR_HPP

#include "Elements.hpp"
#include "Derivative.hpp"
#include "SimulationParams.hpp"
#include "Types.hpp"

#include <memory>

namespace Homme
{

class HyperviscosityFunctorImpl;

class HyperviscosityFunctor
{
public:

  HyperviscosityFunctor (const SimulationParams& params, const Elements& elements, const Derivative& deriv);

  ~HyperviscosityFunctor ();

  size_t buffers_size () const;
  void init_buffers (Real* raw_buffer, const size_t buffer_size);

  void init_boundary_exchanges();

  void run (const int np1, const Real dt, const Real eta_ave_w);

private:

  std::unique_ptr<HyperviscosityFunctorImpl>  m_hvf_impl;
};

} // namespace Homme

#endif // HOMMEXX_HYPERVISCOSITY_FUNCTOR_HPP
