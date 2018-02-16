#include "KernelsBuffersManager.hpp"

namespace Homme
{

KernelsBuffersManager::KernelsBuffersManager ()
 : m_buffer_size      (0)
 , m_buffer_allocated (false)
{
  // Nothing to be done here
}

void KernelsBuffersManager::request_size (const size_t size)
{
  // Sanity check: all functors must request their buffer size BEFORE
  // we proceed with the actual allocation. This is to avoid changing
  // the buffers once allocated, which would require a call-back system,
  // so the previous functors can re-setup their buffers. It would be
  // doable, but it's an unnecessary complication.
  assert (!m_buffer_allocated);

  m_buffer_size = std::max(m_buffer_size, size);
}

void KernelsBuffersManager::allocate_buffer ()
{
  // Sanity check: do not call this method twice!
  assert (!m_buffer_allocated);

  m_raw_buffer = ExecViewManaged<Real*>("the buffer",m_buffer_size);
}

} // namespace Homme
