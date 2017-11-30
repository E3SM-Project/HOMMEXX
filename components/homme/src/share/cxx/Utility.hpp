#ifndef HOMMEXX_UTILITY_HPP
#define HOMMEXX_UTILITY_HPP

#include "Types.hpp"
#include "ExecSpaceDefs.hpp"

#ifndef NDEBUG
#define DEBUG_PRINT(...)                                                       \
  { printf(__VA_ARGS__); }
#else
#define DEBUG_PRINT(...)                                                       \
  {}
#endif

namespace Homme {

// ================ Subviews of 2d views ======================= //
// Note: we still template on ScalarType (should always be Homme::Real here)
//       to allow const/non-const version
template<typename MemSpace, typename MemManagement, typename ScalarType>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType[NP][NP],MemSpace>
subview(ViewType<ScalarType*[NP][NP],MemSpace,MemManagement> v_in, int ie)
{
  return ViewUnmanaged<ScalarType [NP][NP],MemSpace>(&v_in(ie,0,0));
}

// Here, usually, DIM1=DIM2=2 (D and DInv)
template<typename MemSpace, typename MemManagement, typename ScalarType, int DIM1, int DIM2>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType [DIM1][DIM2][NP][NP],MemSpace>
subview(ViewType<ScalarType*[DIM1][DIM2][NP][NP],MemSpace,MemManagement> v_in, int ie)
{
  return ViewUnmanaged<ScalarType [DIM1][DIM2][NP][NP],MemSpace>(&v_in(ie,0,0,0,0));
}

// ================ Subviews of 3d views ======================= //
// Note: we still template on ScalarType (should always be Homme::Scalar here)
//       to allow const/non-const version

template<typename MemSpace, typename MemManagement, typename ScalarType>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>
subview(ViewType<ScalarType*[NP][NP][NUM_LEV],MemSpace,MemManagement> v_in, int ie)
{
  return ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>(&v_in(ie,0,0,0));
}

template<typename MemSpace, typename MemManagement, typename ScalarType, int DIM>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType [DIM][NP][NP][NUM_LEV],MemSpace>
subview(ViewType<ScalarType*[DIM][NP][NP][NUM_LEV],MemSpace,MemManagement> v_in, int ie)
{
  return ViewUnmanaged<ScalarType [DIM][NP][NP][NUM_LEV],MemSpace>(&v_in(ie,0,0,0,0));
}

template<typename MemSpace, typename MemManagement, typename ScalarType, int DIM>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>
subview(ViewType<ScalarType*[DIM][NP][NP][NUM_LEV],MemSpace,MemManagement> v_in, int ie, int idim)
{
  return ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>(&v_in(ie,idim,0,0,0));
}

template<typename MemSpace, typename MemManagement, typename ScalarType, int DIM1, int DIM2>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType [DIM1][DIM2][NP][NP][NUM_LEV],MemSpace>
subview(ViewType<ScalarType*[DIM1][DIM2][NP][NP][NUM_LEV],MemSpace,MemManagement> v_in, int ie)
{
  return ViewUnmanaged<ScalarType [DIM1][DIM2][NP][NP][NUM_LEV],MemSpace>(&v_in(ie,0,0,0,0,0));
}

template<typename MemSpace, typename MemManagement, typename ScalarType, int DIM1, int DIM2>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType [DIM2][NP][NP][NUM_LEV],MemSpace>
subview(ViewType<ScalarType*[DIM1][DIM2][NP][NP][NUM_LEV],MemSpace,MemManagement> v_in, int ie, int idim1)
{
  return ViewUnmanaged<ScalarType [DIM2][NP][NP][NUM_LEV],MemSpace>(&v_in(ie,idim1,0,0,0,0));
}

template<typename MemSpace, typename MemManagement, typename ScalarType, int DIM1, int DIM2>
KOKKOS_INLINE_FUNCTION
ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>
subview(ViewType<ScalarType*[DIM1][DIM2][NP][NP][NUM_LEV],MemSpace,MemManagement> v_in, int ie, int idim1, int idim2)
{
  return ViewUnmanaged<ScalarType [NP][NP][NUM_LEV],MemSpace>(&v_in(ie,idim1,idim2,0,0,0));
}

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
        for (int vector = 0; vector < VECTOR_SIZE && level < NUM_PHYSICAL_LEV; ++vector, ++level) {
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

if( igp==0  && jgp == 0)
std::cout << "hey and level=" << level;
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


//adding one for arrays ie,np,np withut vertical index
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
    !std::is_same<typename ViewType::value_type, Scalar>::value, Real>::type
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
    std::is_same<typename ViewType::value_type, Scalar>::value, Real>::type
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
    for(int j = 0; j < VECTOR_SIZE; ++j) {
      x[i][j] = pdf(engine);
    }
  }
}

template <typename ViewType, typename rngAlg, typename PDF>
void genRandArray(ViewType view, rngAlg &engine, PDF &&pdf) {
  genRandArray(view.data(), view.size(), engine, pdf);
}

template <typename FPType>
Real compare_answers(FPType target, FPType computed, FPType relative_coeff = 1.0) {
  Real denom = 1.0;
  if (relative_coeff > 0.0 && target != 0.0) {
    denom = relative_coeff * std::fabs(target);
  }

  return std::fabs(target - computed) / denom;
}

template <typename ExecSpace, typename Tag=void>
Kokkos::TeamPolicy<ExecSpace, Tag> get_default_team_policy(const int nelems) {
  const int threads_per_team =
    DefaultThreadsDistribution<ExecSpace>::threads_per_team(nelems);
  const int vectors_per_thread =
    DefaultThreadsDistribution<ExecSpace>::vectors_per_thread();
  return Kokkos::TeamPolicy<ExecSpace, Tag>(
    nelems, threads_per_team, vectors_per_thread);
}

} // namespace Homme

#endif // HOMMEXX_UTILITY_HPP
