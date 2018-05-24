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

#ifndef HOMMEXX_SYNC_UTILS_HPP
#define HOMMEXX_SYNC_UTILS_HPP

#include "Types.hpp"
#include "ExecSpaceDefs.hpp"

namespace Homme {

// Templates to verify at compile time that a view has the specified array type
template <typename ViewT, typename ArrayT> struct exec_view_mappable {
#if CUDA_PARSE_BUG_FIXED
  using exec_view = ExecViewUnmanaged<ArrayT>;
  static constexpr bool value = Kokkos::Impl::ViewMapping<
      typename ViewT::traits, typename exec_view::traits, void>::is_assignable;
#else
  static constexpr bool value = Kokkos::Impl::ViewMapping<
    typename ViewT::traits,
    typename Kokkos::View<ArrayT, Kokkos::LayoutRight, ExecSpace,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict> >::traits,
    void>::is_assignable;
#endif
};

template <typename ViewT, typename ArrayT> struct host_view_mappable {
#if CUDA_PARSE_BUG_FIXED
  using host_view = HostViewUnmanaged<ArrayT>;
  static constexpr bool value = Kokkos::Impl::ViewMapping<
      typename ViewT::traits, typename host_view::traits, void>::is_assignable;
#else
  static constexpr bool value = Kokkos::Impl::ViewMapping<
    typename ViewT::traits,
    typename Kokkos::View<ArrayT, Kokkos::LayoutRight, Kokkos::HostSpace,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict> >::traits,
    void>::is_assignable;  
#endif
};

// Kokkos views cannot be used to determine which overloaded function to call,
// so implement this check ourselves with enable_if.
// Despite the ugly templates, this provides much better error messages
// These functions synchronize views from the Fortran layout to the Kernel
// layout

// ===================== SYNC FROM DEVICE TO HOST ============================ //

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (exec_view_mappable<Source_T, Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::value &&
     host_view_mappable<Dest_T, Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]>::value),
    void
  >::type
sync_to_host(Source_T source, Dest_T dest)
{
  typename Source_T::HostMirror source_mirror = Kokkos::create_mirror_view(source);
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dest(ie, tl, level, igp, jgp) = source_mirror(ie, tl, igp, jgp, ilev)[ivec];
          }
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (exec_view_mappable<Source_T, Scalar * [NUM_TIME_LEVELS][2][NP][NP][NUM_LEV]>::value &&
     host_view_mappable<Dest_T, Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][2][NP][NP]>::value),
    void
  >::type
sync_to_host(Source_T source, Dest_T dest)
{
  typename Source_T::HostMirror source_mirror = Kokkos::create_mirror_view(source);
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int tl = 0; tl < NUM_TIME_LEVELS; ++tl) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dest(ie, tl, level, 0, igp, jgp) = source_mirror(ie, tl, 0, igp, jgp, ilev)[ivec];
            dest(ie, tl, level, 1, igp, jgp) = source_mirror(ie, tl, 1, igp, jgp, ilev)[ivec];
          }
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (exec_view_mappable<Source_T, Scalar * [NP][NP][NUM_LEV]>::value &&
     host_view_mappable<Dest_T, Real * [NUM_PHYSICAL_LEV][NP][NP]>::value),
    void
  >::type
sync_to_host(Source_T source, Dest_T dest)
{
  typename Source_T::HostMirror source_mirror = Kokkos::create_mirror_view(source);
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      const int ilev = level / VECTOR_SIZE;
      const int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          dest(ie, level, igp, jgp) = source_mirror(ie, igp, jgp, ilev)[ivec];
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (exec_view_mappable<Source_T, Real * [NP][NP]>::value &&
     host_view_mappable<Dest_T, Real * [NP][NP]>::value),
    void
  >::type
sync_to_host(Source_T source, Dest_T dest)
{
  typename Source_T::HostMirror source_mirror = Kokkos::create_mirror_view(source);
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        dest(ie, igp, jgp) = source_mirror(ie, igp, jgp);
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if<
    (exec_view_mappable<Source_T, Scalar[NP][NP][NUM_LEV]>::value &&
         host_view_mappable<Dest_T, Real[NUM_PHYSICAL_LEV][NP][NP]>::value),
    void>::type
sync_to_host(Source_T source, Dest_T dest) {
  typename Source_T::HostMirror source_mirror =
      Kokkos::create_mirror_view(source);
  Kokkos::deep_copy(source_mirror, source);
  for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
    const int ilev = level / VECTOR_SIZE;
    const int ivec = level % VECTOR_SIZE;
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        dest(level, igp, jgp) = source_mirror(igp, jgp, ilev)[ivec];
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (exec_view_mappable<Source_T, Scalar * [2][NP][NP][NUM_LEV]>::value &&
     host_view_mappable<Dest_T, Real * [NUM_PHYSICAL_LEV][2][NP][NP]>::value),
    void
  >::type
sync_to_host(Source_T source, Dest_T dest)
{
  typename Source_T::HostMirror source_mirror = Kokkos::create_mirror_view(source);
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      const int ilev = level / VECTOR_SIZE;
      const int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          dest(ie, level, 0, igp, jgp) = source_mirror(ie, 0, igp, jgp, ilev)[ivec];
          dest(ie, level, 1, igp, jgp) = source_mirror(ie, 1, igp, jgp, ilev)[ivec];
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (exec_view_mappable<Source_T, Scalar * [QSIZE_D][NP][NP][NUM_LEV]>::value &&
     host_view_mappable<Dest_T, Real * [QSIZE_D][NUM_PHYSICAL_LEV][NP][NP]>::value),
    void
  >::type
sync_to_host(Source_T source, Dest_T dest)
{
  typename Source_T::HostMirror source_mirror = Kokkos::create_mirror_view(source);
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int tracer = 0; tracer < QSIZE_D; ++tracer) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dest(ie, tracer, level, igp, jgp) = source_mirror(ie, tracer, igp, jgp, ilev)[ivec];
          }
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (exec_view_mappable<Source_T, Scalar * [Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]>::value &&
     host_view_mappable<Dest_T, Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP][NP]>::value),
    void
  >::type
sync_to_host(Source_T source, Dest_T dest)
{
  typename Source_T::HostMirror source_mirror = Kokkos::create_mirror_view(source);
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int time = 0; time < Q_NUM_TIME_LEVELS; ++time) {
      for (int tracer = 0; tracer < QSIZE_D; ++tracer) {
        for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
          const int ilev = level / VECTOR_SIZE;
          const int ivec = level % VECTOR_SIZE;
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp) {
              dest(ie, time, tracer, level, igp, jgp) = source_mirror(ie, time, tracer, igp, jgp, ilev)[ivec];
            }
          }
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if<
    (exec_view_mappable<Source_T, Real[10][NUM_PHYSICAL_LEV + 2]>::value &&
         host_view_mappable<Dest_T, Real[NUM_PHYSICAL_LEV + 2][10]>::value),
    void>::type
sync_to_host(Source_T source, Dest_T dest) {
  typename Source_T::HostMirror source_mirror =
      Kokkos::create_mirror_view(source);
  Kokkos::deep_copy(source_mirror, source);
  for (int i = 0; i < 10; ++i) {
    for (int level = 0; level < NUM_PHYSICAL_LEV + 2; ++level) {
      dest(level, i) = source_mirror(i, level);
    }
  }
}

// These last two sync_to_host deal with views with NUM_INTERFACE_LEV levels on host,
// and either copy the whole view to a view with NUM_LEV_P level packs (1st version) ,
// or copy only the first NUM_PHYSICAL_LEV to a view with NUM_LEV level packs (2nd version)
template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (exec_view_mappable<Source_T, Scalar * [NP][NP][NUM_LEV_P]>::value &&
     host_view_mappable<Dest_T, Real * [NUM_INTERFACE_LEV][NP][NP]>::value),
    void
  >::type
sync_to_host(Source_T source, Dest_T dest)
{
  typename Source_T::HostMirror source_mirror = Kokkos::create_mirror_view(source);
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int level = 0; level < NUM_INTERFACE_LEV; ++level) {
      const int ilev = level / VECTOR_SIZE;
      const int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          dest(ie, level, igp, jgp) = source_mirror(ie, igp, jgp, ilev)[ivec];
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (exec_view_mappable<Source_T, Scalar * [NP][NP][NUM_LEV]>::value &&
     host_view_mappable< Dest_T, Real * [NUM_INTERFACE_LEV][NP][NP]>::value),
    void
  >::type
sync_to_host_p2i(Source_T source, Dest_T dest)
{
  typename Source_T::HostMirror source_mirror = Kokkos::create_mirror_view(source);
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      const int ilev = level / VECTOR_SIZE;
      const int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          dest(ie, level, igp, jgp) = source_mirror(ie, igp, jgp, ilev)[ivec];
        }
      }
    }
  }
}

// ===================== SYNC FROM DEVICE TO HOST ============================ //

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (host_view_mappable<Source_T, Real*[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]>::value &&
     exec_view_mappable<Dest_T, Scalar*[NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::value),
    void
  >::type
sync_to_device(Source_T source, Dest_T dest) {
  typename Dest_T::HostMirror dest_mirror = Kokkos::create_mirror_view(dest);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int tl=0; tl < NUM_TIME_LEVELS; ++tl) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dest_mirror(ie, tl, igp, jgp, ilev)[ivec] = source(ie, tl, level, igp, jgp);
          }
        }
      }
    }
  }
  Kokkos::deep_copy(dest, dest_mirror);
}

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (host_view_mappable<Source_T, Real*[NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][2][NP][NP]>::value &&
     exec_view_mappable<Dest_T, Scalar*[NUM_TIME_LEVELS][2][NP][NP][NUM_LEV]>::value),
    void
  >::type
sync_to_device(Source_T source, Dest_T dest) {
  typename Dest_T::HostMirror dest_mirror = Kokkos::create_mirror_view(dest);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int tl=0; tl < NUM_TIME_LEVELS; ++tl) {
      for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
        const int ilev = level / VECTOR_SIZE;
        const int ivec = level % VECTOR_SIZE;
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dest_mirror(ie, tl, 0, igp, jgp, ilev)[ivec] = source(ie, tl, level, 0, igp, jgp);
            dest_mirror(ie, tl, 1, igp, jgp, ilev)[ivec] = source(ie, tl, level, 1, igp, jgp);
          }
        }
      }
    }
  }
  Kokkos::deep_copy(dest, dest_mirror);
}

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (host_view_mappable<Source_T, Real * [NUM_PHYSICAL_LEV][2][NP][NP]>::value &&
     exec_view_mappable<Dest_T, Scalar * [2][NP][NP][NUM_LEV]>::value),
    void
  >::type
sync_to_device(Source_T source, Dest_T dest)
{
  typename Dest_T::HostMirror dest_mirror = Kokkos::create_mirror_view(dest);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      const int ilev = level / VECTOR_SIZE;
      const int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          dest_mirror(ie, 0, igp, jgp, ilev)[ivec] = source(ie, level, 0, igp, jgp);
          dest_mirror(ie, 1, igp, jgp, ilev)[ivec] = source(ie, level, 1, igp, jgp);
        }
      }
    }
  }
  Kokkos::deep_copy(dest, dest_mirror);
}

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (host_view_mappable<Source_T, Real * [NP][NP]>::value &&
     exec_view_mappable<Dest_T, Real * [NP][NP]>::value),
    void
  >::type
sync_to_device(Source_T source, Dest_T dest)
{
  typename Dest_T::HostMirror dest_mirror = Kokkos::create_mirror_view(dest);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        dest_mirror(ie, igp, jgp) = source(ie, igp, jgp);
      }
    }
  }
  Kokkos::deep_copy(dest, dest_mirror);
}

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (host_view_mappable<Source_T,Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP][NP]>::value &&
     exec_view_mappable<Dest_T,Scalar * [Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]>::value),
    void
  >::type
sync_to_device(Source_T source, Dest_T dest)
{
  typename Dest_T::HostMirror dest_mirror = Kokkos::create_mirror_view(dest);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int q_tl = 0; q_tl < Q_NUM_TIME_LEVELS; ++q_tl) {
      for (int q = 0; q < QSIZE_D; ++q) {
        for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
          const int ilev = level / VECTOR_SIZE;
          const int ivec = level % VECTOR_SIZE;
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp) {
              dest_mirror(ie, q_tl, q, igp, jgp, ilev)[ivec] = source(ie, q_tl, q, level, igp, jgp);
            }
          }
        }
      }
    }
  }
  Kokkos::deep_copy(dest, dest_mirror);
}

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (host_view_mappable<Source_T, Real*[NUM_PHYSICAL_LEV][NP][NP]>::value &&
     exec_view_mappable<Dest_T, Scalar*[NP][NP][NUM_LEV]>::value),
    void
  >::type
sync_to_device(Source_T source, Dest_T dest)
{
  typename Dest_T::HostMirror dest_mirror = Kokkos::create_mirror_view(dest);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      const int ilev = level / VECTOR_SIZE;
      const int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          dest_mirror(ie, igp, jgp, ilev)[ivec] = source(ie, level, igp, jgp);
        }
      }
    }
  }
  Kokkos::deep_copy(dest, dest_mirror);
}

// These last two sync_to_host deal with views with NUM_INTERFACE_LEV levels on host,
// and either copy the whole view from a view with NUM_LEV_P level packs (1st version) ,
// or copy only the first NUM_PHYSICAL_LEV from a view with NUM_LEV level packs (2nd version)
template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (host_view_mappable<Source_T, Real*[NUM_INTERFACE_LEV][NP][NP]>::value &&
     exec_view_mappable<Dest_T, Scalar*[NP][NP][NUM_LEV_P]>::value),
    void
  >::type
sync_to_device(Source_T source, Dest_T dest)
{
  typename Dest_T::HostMirror dest_mirror = Kokkos::create_mirror_view(dest);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int level = 0; level < NUM_INTERFACE_LEV; ++level) {
      const int ilev = level / VECTOR_SIZE;
      const int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          dest_mirror(ie, igp, jgp, ilev)[ivec] = source(ie, level, igp, jgp);
        }
      }
    }
  }
  Kokkos::deep_copy(dest, dest_mirror);
}

template <typename Source_T, typename Dest_T>
typename std::enable_if
  <
    (host_view_mappable<Source_T, Real * [NUM_INTERFACE_LEV][NP][NP]>::value &&
     exec_view_mappable<Dest_T, Scalar * [NP][NP][NUM_LEV]>::value),
    void
  >::type
sync_to_device_i2p(Source_T source, Dest_T dest)
{
  typename Dest_T::HostMirror dest_mirror = Kokkos::create_mirror_view(dest);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int level = 0; level < NUM_PHYSICAL_LEV; ++level) {
      const int ilev = level / VECTOR_SIZE;
      const int ivec = level % VECTOR_SIZE;
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          dest_mirror(ie, igp, jgp, ilev)[ivec] = source(ie, level, igp, jgp);
        }
      }
    }
  }
  Kokkos::deep_copy(dest, dest_mirror);
}

} // namespace Homme

#endif // HOMMEXX_SYNC_UTILS_HPP
