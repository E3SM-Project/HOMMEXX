#ifndef HOMMEXX_KERNELS_BUFFERS_MANAGER_HPP
#define HOMMEXX_KERNELS_BUFFERS_MANAGER_HPP

#include "Types.hpp"

#include <memory>

namespace Homme
{

class KernelsBuffersManager
{
public:

  KernelsBuffersManager ();

  void request_size (const size_t size);
  Real* get_raw_buffer () const;

  size_t buffer_size () const { return m_buffer_size; }

  void allocate_buffer ();
private:

  size_t                  m_buffer_size;
  bool                    m_buffer_allocated;

  // The buffer
  ExecViewManaged<Real*>  m_raw_buffer;
};

inline Real* KernelsBuffersManager::get_raw_buffer () const
{
  // Sanity check
  assert (m_buffer_allocated);

  return m_raw_buffer.data();
}

} // namespace Homme

#endif // HOMMEXX_KERNELS_BUFFERS_MANAGER_HPP
