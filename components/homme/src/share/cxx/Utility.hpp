#ifndef HOMMEXX_UTILITY_HPP
#define HOMMEXX_UTILITY_HPP

#include "Types.hpp"
#include "ExecSpaceDefs.hpp"

#include <functional>

#ifndef NDEBUG
#define DEBUG_PRINT(...)                                                       \
  { printf(__VA_ARGS__); }
#else
#define DEBUG_PRINT(...)                                                       \
  {}
#endif

namespace Homme {

// ================ Subviews of several ranks views ======================= //
// Note: we template on ScalarType to allow both Real and Scalar case, and
//       also to allow const/non-const versions.
// Note: we assume to have exactly one runtime dimension.
template <typename ScalarType, int DIM1, typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM1], MemSpace>
subview(ViewType<ScalarType * [DIM1], MemSpace, Properties...> v_in, int ie) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  return ViewUnmanaged<ScalarType[DIM1], MemSpace>(
      &v_in.implementation_map().reference(ie, 0));
}

template <typename ScalarType, int DIM1, int DIM2,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM1][DIM2], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2], MemSpace, Properties...> v_in,
        int ie) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  return ViewUnmanaged<ScalarType[DIM1][DIM2], MemSpace>(
      &v_in.implementation_map().reference(ie, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3], MemSpace, Properties...>
            v_in, int ie) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  return ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3], MemSpace>(
      &v_in.implementation_map().reference(ie, 0, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3], MemSpace>
subview(ViewType<ScalarType * [DIM2][DIM3], MemSpace, Properties...> v_in,
        int ie, int idim1) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(idim1 < v_in.extent_int(1));
  assert(idim1 >= 1);
  return ViewUnmanaged<ScalarType[DIM2][DIM3], MemSpace>(
      &v_in.implementation_map().reference(ie, idim1, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3][DIM4], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4], MemSpace, Properties...>
            v_in, int ie) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  return ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3][DIM4], MemSpace>(
      &v_in.implementation_map().reference(ie, 0, 0, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM2][DIM3][DIM4], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4], MemSpace, Properties...>
            v_in, int ie, int idim1) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(idim1 < v_in.extent_int(1));
  assert(idim1 >= 0);
  return ViewUnmanaged<ScalarType[DIM2][DIM3][DIM4], MemSpace>(
      &v_in.implementation_map().reference(ie, idim1, 0, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM3][DIM4], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4], MemSpace, Properties...>
            v_in, int ie, int idim1, int idim2) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(idim1 < v_in.extent_int(1));
  assert(idim1 >= 0);
  assert(idim2 < v_in.extent_int(2));
  assert(idim2 >= 0);
  return ViewUnmanaged<ScalarType[DIM3][DIM4], MemSpace>(
      &v_in.implementation_map().reference(ie, idim1, idim2, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM4], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4], MemSpace, Properties...>
            v_in, int ie, int idim1, int idim2, int idim3) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(idim1 < v_in.extent_int(1));
  assert(idim1 >= 0);
  assert(idim2 < v_in.extent_int(2));
  assert(idim2 >= 0);
  assert(idim3 < v_in.extent_int(3));
  assert(idim3 >= 0);
  return ViewUnmanaged<ScalarType[DIM4], MemSpace>(
      &v_in.implementation_map().reference(ie, idim1, idim2, idim3, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4, int DIM5,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3][DIM4][DIM5], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4][DIM5], MemSpace, Properties...>
            v_in, int ie) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  return ViewUnmanaged<ScalarType[DIM1][DIM2][DIM3][DIM4][DIM5], MemSpace>(
      &v_in.implementation_map().reference(ie, 0, 0, 0, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4, int DIM5,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM2][DIM3][DIM4][DIM5], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4][DIM5], MemSpace, Properties...>
            v_in, int ie, int idim1) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(idim1 < v_in.extent_int(1));
  assert(idim1 >= 0);
  return ViewUnmanaged<ScalarType[DIM2][DIM3][DIM4][DIM5], MemSpace>(
      &v_in.implementation_map().reference(ie, idim1, 0, 0, 0, 0));
}

template <typename ScalarType, int DIM1, int DIM2, int DIM3, int DIM4, int DIM5,
          typename MemSpace, typename... Properties>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[DIM3][DIM4][DIM5], MemSpace>
subview(ViewType<ScalarType * [DIM1][DIM2][DIM3][DIM4][DIM5], MemSpace, Properties...>
            v_in, int ie, int idim1, int idim2) {
  assert(v_in.data() != nullptr);
  assert(ie < v_in.extent_int(0));
  assert(ie >= 0);
  assert(idim1 < v_in.extent_int(1));
  assert(idim1 >= 0);
  assert(idim2 < v_in.extent_int(2));
  assert(idim2 >= 0);
  return ViewUnmanaged<ScalarType[DIM3][DIM4][DIM5], MemSpace>(
      &v_in.implementation_map().reference(ie, idim1, idim2, 0, 0, 0));
}

// ======================================================================== //

// Templates to verify at compile time that a view has the specified array type
template <typename ViewT, typename ArrayT> struct exec_view_mappable {
  using exec_view = ExecViewUnmanaged<ArrayT>;
  static constexpr bool value = Kokkos::Impl::ViewMapping<
      typename ViewT::traits, typename exec_view::traits, void>::is_assignable;
};

template <typename ViewT, typename ArrayT> struct host_view_mappable {
  using host_view = HostViewUnmanaged<ArrayT>;
  static constexpr bool value = Kokkos::Impl::ViewMapping<
      typename ViewT::traits, typename host_view::traits, void>::is_assignable;
};

// Kokkos views cannot be used to determine which overloaded function to call,
// so implement this check ourselves with enable_if.
// Despite the ugly templates, this provides much better error messages
// These functions synchronize views from the Fortran layout to the Kernel
// layout
template <typename Source_T, typename Dest_T>
typename std::enable_if<
    exec_view_mappable<Source_T, Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::
        value &&host_view_mappable<
            Dest_T, Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][NP][NP]>::value,
    void>::type
sync_to_host(Source_T source, Dest_T dest) {
  ExecViewUnmanaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror
  source_mirror(Kokkos::create_mirror_view(source));
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int time = 0; time < NUM_TIME_LEVELS; ++time) {
      for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
           ++vector_level) {
        for (int vector = 0; vector < VECTOR_SIZE && level < NUM_PHYSICAL_LEV;
             ++vector, ++level) {
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp) {
              dest(ie, time, level, igp, jgp) =
                  source_mirror(ie, time, igp, jgp, vector_level)[vector];
            }
          }
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if<
    exec_view_mappable<Source_T, Scalar * [NP][NP][NUM_LEV]>::value &&
        host_view_mappable<Dest_T, Real * [NUM_PHYSICAL_LEV][NP][NP]>::value,
    void>::type
sync_to_host(Source_T source, Dest_T dest) {
  ExecViewUnmanaged<Scalar * [NP][NP][NUM_LEV]>::HostMirror source_mirror(
      Kokkos::create_mirror_view(source));
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
         ++vector_level) {
      for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dest(ie, level, igp, jgp) =
                source_mirror(ie, igp, jgp, vector_level)[vector];
          }
        }
      }
    }
  }
}


//this is for sdot =  ExecViewManaged<Real* [NP][NP]> 
template <typename Source_T, typename Dest_T>
typename std::enable_if<
    exec_view_mappable<Source_T, Real * [NP][NP]>::value &&
        host_view_mappable<Dest_T, Real * [NP][NP]>::value,
    void>::type
sync_to_host(Source_T source, Dest_T dest) {
  ExecViewUnmanaged<Real * [NP][NP]>::HostMirror source_mirror(
      Kokkos::create_mirror_view(source));
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
    exec_view_mappable<Source_T, Scalar * [2][NP][NP][NUM_LEV]>::value &&
        host_view_mappable<Dest_T, Real * [NUM_PHYSICAL_LEV][2][NP][NP]>::value,
    void>::type
sync_to_host(Source_T source, Dest_T dest) {
  ExecViewUnmanaged<Scalar * [2][NP][NP][NUM_LEV]>::HostMirror source_mirror(
      Kokkos::create_mirror_view(source));
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
         ++vector_level) {
      for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
        for (int dim = 0; dim < 2; ++dim) {
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp) {
              dest(ie, level, dim, igp, jgp) =
                  source_mirror(ie, dim, igp, jgp, vector_level)[vector];
            }
          }
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if<
    exec_view_mappable<Source_T,
                       Scalar * [Q_NUM_TIME_LEVELS][QSIZE_D][NP][NP][NUM_LEV]>::
        value &&host_view_mappable<
            Dest_T, Real * [Q_NUM_TIME_LEVELS][QSIZE_D][NUM_PHYSICAL_LEV][NP]
                                              [NP]>::value,
    void>::type
sync_to_host(Source_T source, Dest_T dest) {
  typename Source_T::HostMirror source_mirror(
      Kokkos::create_mirror_view(source));
  Kokkos::deep_copy(source_mirror, source);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int time = 0; time < Q_NUM_TIME_LEVELS; ++time) {
      for (int tracer = 0; tracer < QSIZE_D; ++tracer) {
        for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
             ++vector_level) {
          for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
            for (int igp = 0; igp < NP; ++igp) {
              for (int jgp = 0; jgp < NP; ++jgp) {
                dest(ie, time, tracer, level, igp, jgp) = source_mirror(
                    ie, time, tracer, igp, jgp, vector_level)[vector];
              }
            }
          }
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if<
    exec_view_mappable<Source_T, Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::
        value &&host_view_mappable<
            Dest_T, Real * [NUM_TIME_LEVELS][NUM_PHYSICAL_LEV][2][NP][NP]>::value,
    void>::type
sync_to_host(Source_T source_1, Source_T source_2, Dest_T dest) {
  ExecViewUnmanaged<Scalar * [NUM_TIME_LEVELS][NP][NP][NUM_LEV]>::HostMirror
  source_1_mirror(Kokkos::create_mirror_view(source_1)),
      source_2_mirror(Kokkos::create_mirror_view(source_2));
  Kokkos::deep_copy(source_1_mirror, source_1);
  Kokkos::deep_copy(source_2_mirror, source_2);
  for (int ie = 0; ie < source_1.extent_int(0); ++ie) {
    for (int time = 0; time < NUM_TIME_LEVELS; ++time) {
      for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
           ++vector_level) {
        for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp) {
              dest(ie, time, level, 0, igp, jgp) =
                  source_1_mirror(ie, time, igp, jgp, vector_level)[vector];
              dest(ie, time, level, 1, igp, jgp) =
                  source_2_mirror(ie, time, igp, jgp, vector_level)[vector];
            }
          }
        }
      }
    }
  }
}

template <typename Source_T, typename Dest_T>
typename std::enable_if<
    exec_view_mappable<Source_T, Scalar * [NP][NP][NUM_LEV]>::value &&
        host_view_mappable<Dest_T, Real * [NUM_PHYSICAL_LEV][2][NP][NP]>::value,
    void>::type
sync_to_host(Source_T source_1, Source_T source_2, Dest_T dest) {
  ExecViewUnmanaged<Scalar * [NP][NP][NUM_LEV]>::HostMirror source_1_mirror(
      Kokkos::create_mirror_view(source_1)),
      source_2_mirror(Kokkos::create_mirror_view(source_2));
  Kokkos::deep_copy(source_1_mirror, source_1);
  Kokkos::deep_copy(source_2_mirror, source_2);
  for (int ie = 0; ie < source_1.extent_int(0); ++ie) {
    for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
         ++vector_level) {
      for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dest(ie, level, 0, igp, jgp) =
                source_1_mirror(ie, igp, jgp, vector_level)[vector];
            dest(ie, level, 1, igp, jgp) =
                source_2_mirror(ie, igp, jgp, vector_level)[vector];
          }
        }
      }
    }
  }
}

//this version is for using NLEV+1 variables, like eta_dot_dpdn
//source is Scalar*NUM_LEV+1, dest is Real*NUM_PHYSICAL_LEV+1
template <typename Source_T, typename Dest_T>
typename std::enable_if<
    exec_view_mappable<Source_T, Scalar * [NP][NP][NUM_LEV+1]>::
        value &&host_view_mappable< Dest_T, Real * [NUM_PHYSICAL_LEV+1][NP][NP]>::value,
    void>::type
sync_to_host(Source_T source, Dest_T dest) {

  ExecViewUnmanaged<Scalar * [NP][NP][NUM_LEV+1]>::HostMirror
      source_mirror(Kokkos::create_mirror_view(source));

  Kokkos::deep_copy(source_mirror, source);

  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int vector_level = 0, level = 0; vector_level < NUM_LEV+1; ++vector_level) {
      for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            if(level <  (NUM_PHYSICAL_LEV+1) )
              dest(ie, level, igp, jgp) = source_mirror(ie, igp, jgp, vector_level)[vector];
          }
        }
      }
    }
  }
}



template <typename Source_T, typename Dest_T>
typename std::enable_if<
    host_view_mappable<Source_T, Real * [NUM_PHYSICAL_LEV][NP][NP]>::value &&
        exec_view_mappable<Dest_T, Scalar * [NP][NP][NUM_LEV]>::value,
    void>::type
sync_to_device(Source_T source, Dest_T dest) {
  typename Dest_T::HostMirror dest_mirror = Kokkos::create_mirror_view(dest);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
         ++vector_level) {
      for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dest_mirror(ie, igp, jgp, vector_level)[vector] =
                source(ie, level, igp, jgp);
          }
        }
      }
    }
  }
  Kokkos::deep_copy(dest, dest_mirror);
}


//to copy num_phys_lev+1 to ...
template <typename Source_T, typename Dest_T>
typename std::enable_if<
    host_view_mappable<Source_T, Real * [NUM_PHYSICAL_LEV+1][NP][NP]>::value &&
        exec_view_mappable<Dest_T, Scalar * [NP][NP][NUM_LEV_P]>::value,
    void>::type
sync_to_device(Source_T source, Dest_T dest) {
  typename Dest_T::HostMirror dest_mirror = Kokkos::create_mirror_view(dest);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int vector_level = 0, level = 0; vector_level < NUM_LEV_P;
         ++vector_level) {
      for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
if( level < (NUM_PHYSICAL_LEV+1) ){
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dest_mirror(ie, igp, jgp, vector_level)[vector] =
                source(ie, level, igp, jgp);
          }
        }
}
      }
    }
  }
  Kokkos::deep_copy(dest, dest_mirror);
}



template <typename Source_T, typename Dest_T>
typename std::enable_if<
    host_view_mappable<Source_T, Real * [NUM_PHYSICAL_LEV][2][NP][NP]>::value &&
        exec_view_mappable<Dest_T, Scalar * [NP][NP][NUM_LEV]>::value,
    void>::type
sync_to_device(Source_T source, Dest_T dest_1, Dest_T dest_2) {
  typename Dest_T::HostMirror dest_1_mirror =
      Kokkos::create_mirror_view(dest_1);
  typename Dest_T::HostMirror dest_2_mirror =
      Kokkos::create_mirror_view(dest_2);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
         ++vector_level) {
      for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
        for (int igp = 0; igp < NP; ++igp) {
          for (int jgp = 0; jgp < NP; ++jgp) {
            dest_1_mirror(ie, igp, jgp, vector_level)[vector] =
                source(ie, level, 0, igp, jgp);
            dest_2_mirror(ie, igp, jgp, vector_level)[vector] =
                source(ie, level, 1, igp, jgp);
          }
        }
      }
    }
  }
  Kokkos::deep_copy(dest_1, dest_1_mirror);
  Kokkos::deep_copy(dest_2, dest_2_mirror);
}

template <typename Source_T, typename Dest_T>
typename std::enable_if<
    host_view_mappable<Source_T, Real * [NUM_PHYSICAL_LEV][2][NP][NP]>::value &&
        exec_view_mappable<Dest_T, Scalar * [2][NP][NP][NUM_LEV]>::value,
    void>::type
sync_to_device(Source_T source, Dest_T dest) {
  typename Dest_T::HostMirror dest_mirror = Kokkos::create_mirror_view(dest);
  for (int ie = 0; ie < source.extent_int(0); ++ie) {
    for (int vector_level = 0, level = 0; vector_level < NUM_LEV;
         ++vector_level) {
      for (int vector = 0; vector < VECTOR_SIZE; ++vector, ++level) {
        for (int dim = 0; dim < 2; ++dim) {
          for (int igp = 0; igp < NP; ++igp) {
            for (int jgp = 0; jgp < NP; ++jgp) {
              dest_mirror(ie, dim, igp, jgp, vector_level)[vector] =
                  source(ie, level, dim, igp, jgp);
            }
          }
        }
      }
    }
  }
  Kokkos::deep_copy(dest, dest_mirror);
}

//adding one for arrays ie,np,np without vertical index, like sdot_sum
template <typename Source_T, typename Dest_T>
typename std::enable_if<
    host_view_mappable<Source_T, Real * [NP][NP]>::value &&
        exec_view_mappable<Dest_T, Real * [NP][NP]>::value,
    void>::type
sync_to_device(Source_T source, Dest_T dest) {
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




template <typename ViewType>
typename std::enable_if<
    !std::is_same<typename ViewType::non_const_value_type, Scalar>::value, Real>::type
frobenius_norm(const ViewType view) {
  typename ViewType::pointer_type data = view.data();

  size_t length = view.size();

  // Note: use Kahan algorithm to increase accuracy
  Real norm = 0;
  Real c = 0;
  Real temp, y;
  for (size_t i = 0; i < length; ++i) {
    y = data[i] * data[i] - c;
    temp = norm + y;
    c = (temp - norm) - y;
    norm = temp;
  }

  return std::sqrt(norm);
}

template <typename ViewType>
typename std::enable_if<
    std::is_same<typename ViewType::non_const_value_type, Scalar>::value, Real>::type
frobenius_norm(const ViewType view) {
  typename ViewType::pointer_type data = view.data();

  size_t length = view.size();

  // Note: use Kahan algorithm to increase accuracy
  Real norm = 0;
  Real c = 0;
  Real temp, y;
  for (size_t i = 0; i < length; ++i) {
    for (int v = 0; v < VECTOR_SIZE; ++v) {
      y = data[i][v] * data[i][v] - c;
      temp = norm + y;
      c = (temp - norm) - y;
      norm = temp;
    }
  }

  return std::sqrt(norm);
}

template <typename rngAlg, typename PDF>
void genRandArray(Real *const x, int length, rngAlg &engine, PDF &&pdf) {
  for (int i = 0; i < length; ++i) {
    x[i] = pdf(engine);
  }
}

template <typename rngAlg, typename PDF>
void genRandArray(Scalar *const x, int length, rngAlg &engine, PDF &&pdf) {
  for (int i = 0; i < length; ++i) {
    for (int j = 0; j < VECTOR_SIZE; ++j) {
      x[i][j] = pdf(engine);
    }
  }
}

template <typename ViewType, typename rngAlg, typename PDF>
typename std::enable_if<Kokkos::is_view<ViewType>::value, void>::type
genRandArray(ViewType view, rngAlg &engine, PDF &&pdf,
             std::function<bool(typename ViewType::HostMirror)> constraint) {
  typename ViewType::HostMirror mirror = Kokkos::create_mirror_view(view);
  do {
    genRandArray(mirror.data(), view.size(), engine, pdf);
  } while (constraint(mirror) == false);
  Kokkos::deep_copy(view, mirror);
}

template <typename ViewType, typename rngAlg, typename PDF>
typename std::enable_if<Kokkos::is_view<ViewType>::value, void>::type
genRandArray(ViewType view, rngAlg &engine, PDF &&pdf) {
  genRandArray(view, engine, pdf,
               [](typename ViewType::HostMirror) { return true; });
}

template <typename FPType>
Real compare_answers(FPType target, FPType computed,
                     FPType relative_coeff = 1.0) {
  Real denom = 1.0;
  if (relative_coeff > 0.0 && target != 0.0) {
    denom = relative_coeff * std::fabs(target);
  }

  return std::fabs(target - computed) / denom;
}

} // namespace Homme

#endif // HOMMEXX_UTILITY_HPP
