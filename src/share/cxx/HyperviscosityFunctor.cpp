#include "HyperviscosityFunctor.hpp"
#include "HyperviscosityFunctorImpl.hpp"

namespace Homme
{

HyperviscosityFunctor::HyperviscosityFunctor (const SimulationParams& params, const Elements& elements, const Derivative& deriv)
{
  m_hvf_impl.reset (new HyperviscosityFunctorImpl(params,elements,deriv));
}

HyperviscosityFunctor::~HyperviscosityFunctor ()
{
  // This empty destructor (where HyperviscosityFunctorImpl type is completely known)
  // is necessary for pimpl idiom to work with unique_ptr. The issue is the
  // deleter, which needs to know the size of the stored type, and which
  // would be called from the implicitly declared default destructor, which
  // would be in the header file, where HyperviscosityFunctorImpl type is incomplete.
}

size_t HyperviscosityFunctor::buffers_size () const
{
  assert (m_hvf_impl);
  return m_hvf_impl->buffers_size();
}

void HyperviscosityFunctor::init_buffers (Real* raw_buffer, const size_t buffer_size) {
  assert (m_hvf_impl);
  m_hvf_impl->init_buffers(raw_buffer, buffer_size);
}

void HyperviscosityFunctor::init_boundary_exchanges () {
  assert (m_hvf_impl);
  m_hvf_impl->init_boundary_exchanges();
}

void HyperviscosityFunctor::run (const int np1, const Real dt, const Real eta_ave_w)
{
  // Sanity check (this should NEVER happen by design)
  assert (m_hvf_impl);

  m_hvf_impl->run(np1,dt,eta_ave_w);
}

} // namespace Homme
