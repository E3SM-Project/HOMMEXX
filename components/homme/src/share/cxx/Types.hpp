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

#ifndef HOMMEXX_TYPES_HPP
#define HOMMEXX_TYPES_HPP

#include <Hommexx_config.h>
#include <Kokkos_Core.hpp>

#include "ExecSpaceDefs.hpp"
#include "Dimensions.hpp"

#include <vector/KokkosKernels_Vector.hpp>

#ifdef HAVE_CONFIG_H
#include "config.h.c"
#endif

#define __MACRO_STRING(MacroVal) #MacroVal
#define MACRO_STRING(MacroVal) __MACRO_STRING(MacroVal)

namespace Homme {

// Usual typedef for real scalar type
using Real = double;
using RCPtr = Real *const;
using CRCPtr = const Real *const;
using F90Ptr = Real *const; // Using this in a function signature emphasizes
                            // that the ordering is Fortran
using CF90Ptr = const Real *const; // Using this in a function signature
                                   // emphasizes that the ordering is Fortran

#if (AVX_VERSION > 0)
using VectorTagType =
    KokkosKernels::Batched::Experimental::AVX<Real, ExecSpace>;
#else
using VectorTagType =
    KokkosKernels::Batched::Experimental::SIMD<Real, ExecSpace>;
#endif // AVX_VERSION

using VectorType =
    KokkosKernels::Batched::Experimental::VectorTag<VectorTagType, VECTOR_SIZE>;

using Scalar = KokkosKernels::Batched::Experimental::Vector<VectorType>;

static_assert(sizeof(Scalar) > 0, "Vector type has 0 size");
static_assert(sizeof(Scalar) == sizeof(Real[VECTOR_SIZE]), "Vector type is not correctly defined");
static_assert(Scalar::vector_length>0, "Vector type is not correctly defined (vector_length=0)");

using MemoryManaged   = Kokkos::MemoryTraits<Kokkos::Restrict>;
using MemoryUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>;

// The memory spaces
using ExecMemSpace    = ExecSpace::memory_space;
using ScratchMemSpace = ExecSpace::scratch_memory_space;
using HostMemSpace    = Kokkos::HostSpace;
// Selecting the memory space for the MPI buffers, that is, where the MPI
// buffers will be allocated. In a CPU build, this is always going to be
// the same as ExecMemSpace, i.e., HostSpace. In a GPU build, one can choose
// whether the MPI is done on host or on device. If on device, then all MPI
// calls will be issued from device pointers.
// NOTE: this has nothing to do with pack/unpack, which ALWAYS happen on
//       device views (to be done in parallel). The difference is ONLY in
//       the location of the MPI buffer for send/receive.

#if HOMMEXX_MPI_ON_DEVICE
  using MPIMemSpace = ExecMemSpace;
#else
  using MPIMemSpace = HostMemSpace;
#endif

// A team member type
using TeamMember     = Kokkos::TeamPolicy<ExecSpace>::member_type;

// Short name for views
template <typename DataType, typename... Properties>
using ViewType = Kokkos::View<DataType, Kokkos::LayoutRight, Properties...>;

// Managed/Unmanaged view
template <typename DataType, typename... Properties>
using ViewManaged = ViewType<DataType, Properties..., MemoryManaged>;
template <typename DataType, typename... Properties>
using ViewUnmanaged = ViewType<DataType, Properties..., MemoryUnmanaged>;

// Host/Device/MPI views
template <typename DataType, typename... Properties>
using HostView = ViewType<DataType, HostMemSpace, Properties...>;
template <typename DataType, typename... Properties>
using ExecView = ViewType<DataType, ExecMemSpace, Properties...>;
// Work around Cuda 9.1.85 parse bugs.
#if CUDA_PARSE_BUG_FIXED
template <typename DataType, typename... Properties>
using MPIView = typename std::conditional<std::is_same<MPIMemSpace,ExecMemSpace>::value,
                                          ExecView<DataType,Properties...>,
                                          typename ExecView<DataType,Properties...>::HostMirror>::type;
#else
# if HOMMEXX_MPI_ON_DEVICE
template <typename DataType, typename... Properties>
using MPIView = ExecView<DataType,Properties...>;
# else
template <typename DataType, typename... Properties>
using MPIView = typename ExecView<DataType,Properties...>::HostMirror>::type;
# endif
#endif

// Further specializations for execution space and managed/unmanaged memory
template <typename DataType>
using ExecViewManaged = ExecView<DataType, MemoryManaged>;
template <typename DataType>
using ExecViewUnmanaged = ExecView<DataType, MemoryUnmanaged>;

// Further specializations for host space.
template <typename DataType>
using HostViewManaged = HostView<DataType, MemoryManaged>;
template <typename DataType>
using HostViewUnmanaged = HostView<DataType, MemoryUnmanaged>;

// Further specializations for MPI memory space.
template <typename DataType>
using MPIViewManaged = MPIView<DataType, MemoryManaged>;
template <typename DataType>
using MPIViewUnmanaged = MPIView<DataType, MemoryUnmanaged>;

// The scratch view type: always unmanaged, and always with c pointers
template <typename DataType>
using ScratchView = ViewType<DataType, ScratchMemSpace, MemoryUnmanaged>;

// To view the fully expanded name of a complicated template type T,
// just try to access some non-existent field of MyDebug<T>. E.g.:
// MyDebug<T>::type i;
template <typename T> struct MyDebug {};

} // Homme

#endif // HOMMEXX_TYPES_HPP
