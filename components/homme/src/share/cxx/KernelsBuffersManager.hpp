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
  ExecViewUnmanaged<Real*> get_buffer () const;

  void allocate_buffer ();
private:

  size_t                  m_buffer_size;
  bool                    m_buffer_allocated;

  // The buffer
  ExecViewManaged<Real*>  m_raw_buffer;
};

inline ExecViewUnmanaged<Real*> KernelsBuffersManager::get_buffer () const
{
  // Sanity check
  assert (m_buffer_allocated);

  return m_raw_buffer;
}

} // namespace Homme

#endif // HOMMEXX_KERNELS_BUFFERS_MANAGER_HPP
