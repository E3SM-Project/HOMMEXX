/*********************************************************************************
 *
 * Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC
 * (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
 * Government retains certain rights in this software.
 *
 * For five (5) years from  the United States Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide
 * license in this data to reproduce, prepare derivative works, and perform
 * publicly and display publicly, by or on behalf of the Government. There is
 * provision for the possible extension of the term of this license. Subsequent
 * to that period or any extension granted, the United States Government is
 * granted for itself and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable worldwide license in this data to reproduce, prepare derivative
 * works, distribute copies to the public, perform publicly and display publicly,
 * and to permit others to do so. The specific term of the license can be
 * identified by inquiry made to National Technology and Engineering Solutions of
 * Sandia, LLC or DOE.
 *
 * NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF
 * ENERGY, NOR NATIONAL TECHNOLOGY AND ENGINEERING SOLUTIONS OF SANDIA, LLC, NOR
 * ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY
 * LEGAL RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY
 * INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS
 * USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
 *
 * Any licensee of this software has the obligation and responsibility to abide
 * by the applicable export control laws, regulations, and general prohibitions
 * relating to the export of technical data. Failure to obtain an export control
 * license or other authority from the Government may result in criminal
 * liability under U.S. laws.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 *     - Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *     - Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimers in the documentation
 *       and/or other materials provided with the distribution.
 *     - Neither the name of Sandia Corporation,
 *       nor the names of its contributors may be used to endorse or promote
 *       products derived from this Software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ********************************************************************************/

#include "BuffersManager.hpp"

#include "BoundaryExchange.hpp"
#include "Connectivity.hpp"

namespace Homme
{

BuffersManager::BuffersManager ()
 : m_num_customers     (0)
 , m_mpi_buffer_size   (0)
 , m_local_buffer_size (0)
 , m_buffers_busy      (false)
 , m_views_are_valid   (false)
{
  // The "fake" buffers used for MISSING connections. These do not depend on the requirements
  // from the custormers, so we can create them right away.
  constexpr size_t blackhole_buffer_size = 2 * NUM_LEV * VECTOR_SIZE;
  m_blackhole_send_buffer = ExecViewManaged<Real*>("blackhole array",blackhole_buffer_size);
  m_blackhole_recv_buffer = ExecViewManaged<Real*>("blackhole array",blackhole_buffer_size);
  Kokkos::deep_copy(m_blackhole_send_buffer,0.0);
  Kokkos::deep_copy(m_blackhole_recv_buffer,0.0);
}

BuffersManager::BuffersManager (std::shared_ptr<Connectivity> connectivity)
 : BuffersManager()
{
  set_connectivity(connectivity);
}

BuffersManager::~BuffersManager ()
{
  // Check that all the customers un-registered themselves
  assert (m_num_customers==0);

  // Check our buffers are not busy
  assert (!m_buffers_busy);
}

void BuffersManager::check_for_reallocation ()
{
  for (auto& it : m_customers) {
    update_requested_sizes (it);
  }
}

void BuffersManager::set_connectivity (std::shared_ptr<Connectivity> connectivity)
{
  // We don't allow a null connectivity, or a change of connectivity
  assert (connectivity && !m_connectivity);

  m_connectivity = connectivity;
}

bool BuffersManager::check_views_capacity (const int num_1d_fields, const int num_2d_fields, const int num_3d_fields) const
{
  size_t mpi_buffer_size, local_buffer_size;
  required_buffer_sizes (num_1d_fields, num_2d_fields, num_3d_fields, mpi_buffer_size, local_buffer_size);

  return (mpi_buffer_size<=m_mpi_buffer_size) &&
         (local_buffer_size<=m_local_buffer_size);
}

void BuffersManager::allocate_buffers ()
{
  // If views are marked as valid, they are already allocated, and no other
  // customer has requested a larger size
  if (m_views_are_valid) {
    return;
  }

  // The buffers used for packing/unpacking
  m_send_buffer  = ExecViewManaged<Real*>("send buffer",  m_mpi_buffer_size);
  m_recv_buffer  = ExecViewManaged<Real*>("recv buffer",  m_mpi_buffer_size);
  m_local_buffer = ExecViewManaged<Real*>("local buffer", m_local_buffer_size);

  // The buffers used in MPI calls
  m_mpi_send_buffer = Kokkos::create_mirror_view(decltype(m_mpi_send_buffer)::execution_space(),m_send_buffer);
  m_mpi_recv_buffer = Kokkos::create_mirror_view(decltype(m_mpi_recv_buffer)::execution_space(),m_recv_buffer);

  m_views_are_valid = true;

  // Tell to all our customers that they need to redo the setup of the internal buffer views
  for (auto& be_ptr : m_customers) {
    // Invalidate buffer views and requests in the customer (if none built yet, it's a no-op)
    be_ptr.first->clear_buffer_views_and_requests ();
  }
}

void BuffersManager::lock_buffers ()
{
  // Make sure we are not trying to lock buffers already locked
  assert (!m_buffers_busy);

  m_buffers_busy = true;
}

void BuffersManager::unlock_buffers ()
{
  // TODO: I am not checking if the buffers are locked. This allows to call
  //       the method twice in a row safely. Is this a bad idea?
  m_buffers_busy = false;
}

void BuffersManager::add_customer (BoundaryExchange* add_me)
{
  // We don't allow null customers (although this should never happen)
  assert (add_me!=nullptr);

  // We also don't allow re-registration
  assert (m_customers.find(add_me)==m_customers.end());

  // Add to the list of customers
  auto pair_it_bool = m_customers.emplace(add_me,CustomerNeeds{0,0});

  // Update the number of customers
  ++m_num_customers;

  // If this customer has already started the registration, we can already update the buffers sizes
  if (add_me->is_registration_started()) {
    update_requested_sizes(*pair_it_bool.first);
  }
}

void BuffersManager::remove_customer (BoundaryExchange* remove_me)
{
  // We don't allow null customers (although this should never happen)
  assert (remove_me!=nullptr);

  // Perhaps overzealous, but won't hurt: we should have customers
  assert (m_num_customers>0);

  // Find the customer
  auto it = m_customers.find(remove_me);

  // We don't allow removal of non-customers
  assert (it!=m_customers.end());

  // Remove the customer and its needs
  m_customers.erase(it);

  // Decrease number of customers
  --m_num_customers;
}

void BuffersManager::update_requested_sizes (typename std::map<BoundaryExchange*,CustomerNeeds>::value_type& customer)
{
  // Make sure connectivity is valid
  assert (m_connectivity && m_connectivity->is_finalized());

  // Make sure this is a customer
  assert (m_customers.find(customer.first)!=m_customers.end());

  // Get the number of fields that this customer has
  const int num_1d_fields = customer.first->get_num_1d_fields();
  const int num_2d_fields = customer.first->get_num_2d_fields();
  const int num_3d_fields = customer.first->get_num_3d_fields();

  // Compute the requested buffers sizes and compare with stored ones
  required_buffer_sizes (num_1d_fields, num_2d_fields, num_3d_fields, customer.second.mpi_buffer_size, customer.second.local_buffer_size);
  if (customer.second.mpi_buffer_size>m_mpi_buffer_size) {
    // Update the total
    m_mpi_buffer_size = customer.second.mpi_buffer_size;

    // Mark the views as invalid
    m_views_are_valid = false;
  }

  if(customer.second.local_buffer_size>m_local_buffer_size) {
    // Update the total
    m_local_buffer_size = customer.second.local_buffer_size;

    // Mark the views as invalid
    m_views_are_valid = false;
  }
}

void BuffersManager::required_buffer_sizes (const int num_1d_fields, const int num_2d_fields, const int num_3d_fields,
                                            size_t& mpi_buffer_size, size_t& local_buffer_size) const
{
  mpi_buffer_size = local_buffer_size = 0;

  // The buffer size for each connection kind
  // Note: for 2d/3d fields, we have 1 Real per GP (per level, in 3d). For 1d fields,
  //       we have 2 Real per level (max and min over element).
  int elem_buf_size[2];
  elem_buf_size[etoi(ConnectionKind::CORNER)] = num_1d_fields*2*NUM_LEV*VECTOR_SIZE + (num_2d_fields + num_3d_fields*NUM_LEV*VECTOR_SIZE) * 1;
  elem_buf_size[etoi(ConnectionKind::EDGE)]   = num_1d_fields*2*NUM_LEV*VECTOR_SIZE + (num_2d_fields + num_3d_fields*NUM_LEV*VECTOR_SIZE) * NP;

  // Compute the requested buffers sizes and compare with stored ones
  mpi_buffer_size += elem_buf_size[etoi(ConnectionKind::CORNER)] * m_connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::SHARED,ConnectionKind::CORNER);
  mpi_buffer_size += elem_buf_size[etoi(ConnectionKind::EDGE)]   * m_connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::SHARED,ConnectionKind::EDGE);

  local_buffer_size += elem_buf_size[etoi(ConnectionKind::CORNER)] * m_connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::LOCAL,ConnectionKind::CORNER);
  local_buffer_size += elem_buf_size[etoi(ConnectionKind::EDGE)]   * m_connectivity->get_num_connections<HostMemSpace>(ConnectionSharing::LOCAL,ConnectionKind::EDGE);
}

} // namespace Homme
